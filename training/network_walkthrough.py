"""Stop-motion frames of one hand going through the trained CNN.

    python training/network_walkthrough.py                  # a test image of 3 fingers
    python training/network_walkthrough.py --label 5
    python training/network_walkthrough.py --image data/raw/ceci/2/<file>.png

Writes docs/figures/network_walkthrough/: frame_1_input.png, frame_2_conv1.png,
frame_3_conv2.png, frame_4_conv3.png, frame_5_probabilities.png (all 1280x720,
same layout, ready to sequence), plus overview.png with every stage in one image.
Each conv frame shows the 8 most active of that layer's feature maps, after
ReLU and max pooling.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from common.preprocess import MODES, preprocess  # noqa: E402
from training.data import load_samples, split_samples  # noqa: E402
from training.model import FingerCNN  # noqa: E402

OUT = ROOT / "docs" / "figures" / "network_walkthrough"
MAPS_SHOWN = 8
BLUE, GRAY, INK, INK_2, SURFACE = "#2a78d6", "#c9c8c3", "#0b0b0b", "#52514e", "#fcfcfb"
ACTIVATION = LinearSegmentedColormap.from_list("blue", ["#fcfcfb", "#86b6ef", "#2a78d6", "#104281"])
STAGES = [
    ("Layer 1: conv 7x7, 16 maps, ReLU, max pool", "64 x 64 each"),
    ("Layer 2: conv 5x5, 32 maps, ReLU, max pool", "32 x 32 each"),
    ("Layer 3: conv 3x3, 64 maps, ReLU, max pool", "16 x 16 each"),
]


def pick_image(label):
    """A correctly sized test-set image of this label, from the middle of the test block."""
    samples, _ = load_samples()
    _, _, test, _ = split_samples(samples)
    matches = sorted((s for s in test if s.label == label), key=lambda s: s.time)
    if not matches:
        raise SystemExit(f"no test images with label {label}")
    return matches[len(matches) // 2].path


def new_frame(step, title):
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor=SURFACE)
    fig.text(0.03, 0.94, f"{step}/5", color=INK_2, fontsize=16)
    fig.text(0.08, 0.94, title, color=INK, fontsize=20)
    return fig


def draw_input(fig, crop, rect):
    ax = fig.add_axes(rect)
    ax.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    ax.set_title("Input: 128 x 128 crop of the red square", color=INK_2, fontsize=11)
    ax.axis("off")


def draw_maps(fig, maps, subtitle, box, note=True):
    """The MAPS_SHOWN most active maps in a 2x4 grid inside box (left, bottom, width, height)."""
    order = np.argsort(maps.mean(axis=(1, 2)))[::-1][:MAPS_SHOWN]
    left, bottom, width, height = box
    cell_w, cell_h = width / 4, height / 2
    for k, m in enumerate(order):
        ax = fig.add_axes([left + (k % 4) * cell_w + 0.005, bottom + (1 - k // 4) * cell_h + 0.03,
                           cell_w - 0.01, cell_h - 0.06])
        ax.imshow(maps[m], cmap=ACTIVATION, vmin=0, vmax=max(maps[m].max(), 1e-6), interpolation="nearest")
        ax.set_title(f"map {m}", color=INK_2, fontsize=9)
        ax.axis("off")
    fig.text(left, bottom - 0.04, subtitle + ("  (darker = stronger response)" if note else ""),
             color=INK_2, fontsize=11)


def draw_probabilities(fig, probs, rect, fontsize=12):
    ax = fig.add_axes(rect, facecolor=SURFACE)
    best = int(probs.argmax())
    bars = ax.bar(range(6), probs, color=[BLUE if i == best else GRAY for i in range(6)], width=0.6)
    for bar, p in zip(bars, probs):
        ax.text(bar.get_x() + bar.get_width() / 2, p + 0.02, f"{p:.0%}", ha="center", color=INK, fontsize=fontsize)
    ax.set_xticks(range(6), [str(i) for i in range(6)], fontsize=fontsize)
    ax.set_xlabel("Fingers", color=INK_2, fontsize=fontsize)
    ax.set_ylim(0, 1.12)
    ax.set_yticks([])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(INK_2)
    ax.tick_params(colors=INK_2, length=0)
    return best


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--label", type=int, default=3, help="pick a test image with this many fingers")
    parser.add_argument("--image", type=Path, help="use this saved crop instead")
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "finger_cnn.pt")
    args = parser.parse_args(argv)

    path = args.image or pick_image(args.label)
    crop = cv2.imread(str(path))
    if crop is None:
        raise SystemExit(f"could not read {path}")
    checkpoint = torch.load(args.model, map_location="cpu")
    mode = checkpoint.get("preprocess_mode", "fixed")
    model = FingerCNN(in_channels=MODES[mode])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    with torch.no_grad():
        stages, logits = model.stage_outputs(torch.from_numpy(preprocess(crop, mode=mode))[None])
    stages = [s[0].numpy() for s in stages]
    probs = torch.softmax(logits, 1)[0].numpy()

    OUT.mkdir(parents=True, exist_ok=True)
    fig = new_frame(1, "A hand goes in")
    draw_input(fig, crop, [0.3, 0.1, 0.4, 0.75])
    fig.savefig(OUT / "frame_1_input.png", facecolor=SURFACE)
    plt.close(fig)

    for i, ((title, size), maps) in enumerate(zip(STAGES, stages)):
        fig = new_frame(i + 2, title)
        draw_input(fig, crop, [0.03, 0.3, 0.22, 0.4])
        draw_maps(fig, maps, f"The 8 most active of {maps.shape[0]} feature maps, {size}", [0.3, 0.12, 0.67, 0.76])
        fig.savefig(OUT / f"frame_{i + 2}_conv{i + 1}.png", facecolor=SURFACE)
        plt.close(fig)

    fig = new_frame(5, "Flatten, one linear layer, softmax: the finger count")
    draw_input(fig, crop, [0.03, 0.3, 0.22, 0.4])
    best = draw_probabilities(fig, probs, [0.33, 0.15, 0.6, 0.65], fontsize=14)
    fig.text(0.33, 0.86, f"Prediction: {best} fingers ({probs[best]:.0%})", color=INK, fontsize=16)
    fig.savefig(OUT / "frame_5_probabilities.png", facecolor=SURFACE)
    plt.close(fig)

    fig = plt.figure(figsize=(19.2, 5.4), dpi=100, facecolor=SURFACE)
    draw_input(fig, crop, [0.01, 0.2, 0.14, 0.6])
    for i, ((title, _), maps) in enumerate(zip(STAGES, stages)):
        draw_maps(fig, maps, title.split(",")[0] + ", 8 most active maps", [0.17 + i * 0.21, 0.2, 0.2, 0.65], note=False)
    fig.text(0.17, 0.06, "Darker = stronger response", color=INK_2, fontsize=11)
    draw_probabilities(fig, probs, [0.82, 0.2, 0.17, 0.55], fontsize=10)
    fig.text(0.82, 0.82, f"Prediction: {best} fingers", color=INK, fontsize=12)
    fig.savefig(OUT / "overview.png", facecolor=SURFACE)
    plt.close(fig)

    print(f"Image: {path}")
    print(f"Prediction: {best} fingers ({probs[best]:.0%}); probabilities "
          + ", ".join(f"{i}: {p:.0%}" for i, p in enumerate(probs)))
    print(f"Wrote 5 frames and overview.png to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
