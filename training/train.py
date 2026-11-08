"""Train the finger-count CNN and write the model, metrics and figures.

    python training/train.py --final          # the lock's model: every session, no test set
    python training/train.py                  # hold out a person or session and test on it
    python training/train.py --mode fixed --augment mild   # the original setup

Writes:
    models/finger_cnn.pt            weights, classes and preprocessing mode
    models/finger_cnn.onnx          the same model for the board (mode in its metadata),
                                    checked against PyTorch
    models/finger_cnn_metrics.json  split, accuracy, confusion matrix, history
    docs/figures/training_curves.png, augmentation_examples.png (final_ prefix with --final),
    and with a test set confusion_matrix.png and misclassified.png
"""
import argparse
import json
import math
import sys
import textwrap
import time
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
from sklearn.metrics import confusion_matrix  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from common.preprocess import IMAGE_SIZE, MODES, preprocess  # noqa: E402
from training.data import (COLOR_JITTER, EXCLUDE, LABELS, augment, count_by_label,  # noqa: E402
                           load_samples, split_samples)
from training.model import FingerCNN  # noqa: E402

MODELS = ROOT / "models"
FIGURES = ROOT / "docs" / "figures"
ONNX_TOLERANCE = 1e-4
DEFAULT_MODE = "fixed"
DEFAULT_AUGMENT = "strong"  # won the cross-session comparison (see training/cross_session.py)
DEFAULT_EPOCHS = 60

# Chart colors (reference palette): series 1 blue, series 2 orange, inks, sequential blue ramp.
BLUE, ORANGE, INK, INK_2, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"
SEQUENTIAL = LinearSegmentedColormap.from_list("blue", ["#fcfcfb", "#cde2fb", "#86b6ef", "#2a78d6", "#104281"])


class CropDataset(Dataset):
    def __init__(self, samples, mode, augment_strength=None, seed=0):
        self.crops = [cv2.imread(str(s.path)) for s in samples]
        self.labels = [s.label for s in samples]
        self.mode = mode
        self.augment_strength = augment_strength
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.crops)

    def __getitem__(self, i):
        crop = self.crops[i]
        if self.augment_strength:
            crop = augment(crop, self.rng, self.augment_strength)
        return torch.from_numpy(preprocess(crop, mode=self.mode)), self.labels[i]


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def run_epoch(model, loader, device, loss_fn, optimizer=None):
    """One pass over the loader; trains if an optimizer is given. Returns (loss, accuracy)."""
    model.train(optimizer is not None)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(optimizer is not None):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            n += len(y)
    return total_loss / n, correct / n


def fit(train, val, mode, augment_strength, epochs=DEFAULT_EPOCHS, seed=0, device=None,
        batch_size=32, lr=1e-3, log=print):
    """Train a FingerCNN; keep the epoch with the best validation accuracy.

    Returns (model, history, best_epoch).
    """
    device = device or pick_device()
    torch.manual_seed(seed)
    train_loader = DataLoader(CropDataset(train, mode, augment_strength, seed), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(CropDataset(val, mode), batch_size=128)
    model = FingerCNN(in_channels=MODES[mode]).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    history = {k: [] for k in ("train_loss", "train_acc", "val_loss", "val_acc")}
    best = (-1.0, float("inf"))
    best_state, best_epoch = None, 0
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, device, loss_fn, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, device, loss_fn)
        for key, value in zip(history, (train_loss, train_acc, val_loss, val_acc)):
            history[key].append(value)
        if (val_acc, -val_loss) > (best[0], -best[1]):
            best = (val_acc, val_loss)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
        if log:
            log(f"epoch {epoch:>3}: train loss {train_loss:.3f} acc {train_acc:.3f} | "
                f"val loss {val_loss:.3f} acc {val_acc:.3f}")
    model.load_state_dict(best_state)
    return model, history, best_epoch


@torch.no_grad()
def predict(model, samples, mode, device=None):
    """Class probabilities for each sample, shape (n, 6)."""
    device = device or next(model.parameters()).device
    model.eval()
    loader = DataLoader(CropDataset(samples, mode), batch_size=128)
    return torch.cat([torch.softmax(model(x.to(device)), 1).cpu() for x, _ in loader]).numpy()


def score(samples, probs):
    """Accuracy, per-class accuracy and confusion matrix (rows = true label)."""
    truth = np.array([s.label for s in samples])
    cm = confusion_matrix(truth, probs.argmax(1), labels=range(len(LABELS)))
    per_class = cm.diagonal() / np.maximum(cm.sum(1), 1)
    return float((probs.argmax(1) == truth).mean()), per_class, cm


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(INK_2)
    ax.tick_params(colors=INK_2)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_curves(history, best_epoch, path):
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor="#fcfcfb")
    for ax, key, title in ((axes[0], "loss", "Cross-entropy loss"), (axes[1], "acc", "Accuracy")):
        ax.plot(epochs, history[f"train_{key}"], color=BLUE, linewidth=2, label="Train (augmented)")
        ax.plot(epochs, history[f"val_{key}"], color=ORANGE, linewidth=2, label="Validation")
        ax.axvline(best_epoch, color=INK_2, linewidth=1, linestyle="--")
        ax.text(best_epoch, ax.get_ylim()[1], f" kept: epoch {best_epoch}", color=INK_2, va="top", fontsize=9)
        ax.set_title(title, color=INK, loc="left")
        ax.set_xlabel("Epoch", color=INK_2)
        style(ax)
    axes[1].set_ylim(0, 1.02)
    axes[1].legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion(cm, per_class, accuracy, path, title="Test set confusion matrix"):
    fig, ax = plt.subplots(figsize=(6.4, 5.6), facecolor="#fcfcfb")
    ax.imshow(cm, cmap=SEQUENTIAL, vmin=0, vmax=max(cm.max(), 1))
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            dark = cm[i, j] > 0.55 * cm.max()
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=11,
                    color="white" if dark else (INK if cm[i, j] else "#a8a7a2"))
    ax.set_xticks(range(len(LABELS)), list(LABELS))
    ax.set_yticks(range(len(LABELS)), [f"{l}   ({a:.0%})" for l, a in zip(LABELS, per_class)])
    ax.set_xlabel("Predicted fingers", color=INK_2)
    ax.set_ylabel("True fingers   (per-class accuracy)", color=INK_2)
    ax.set_title(textwrap.fill(f"{title}: {accuracy:.1%} correct", 52), color=INK, loc="left", fontsize=11)
    ax.tick_params(colors=INK_2, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_misclassified(samples, probs, path, limit=24, title="Misclassified test images"):
    wrong = [i for i, s in enumerate(samples) if probs[i].argmax() != s.label]
    columns = 6
    rows = max(1, math.ceil(min(len(wrong), limit) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(2.1 * columns, 2.8 * rows), facecolor="#fcfcfb", squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    if not wrong:
        axes[0, 0].text(0, 0.5, "No misclassified images", fontsize=13, color=INK)
    shown = wrong if len(wrong) <= limit else [wrong[int(k)] for k in np.linspace(0, len(wrong) - 1, limit)]
    for ax, i in zip(axes.flat, shown):
        ax.imshow(cv2.cvtColor(cv2.imread(str(samples[i].path)), cv2.COLOR_BGR2RGB))
        ax.set_title(f"true {samples[i].label} -> pred {probs[i].argmax()}\n{probs[i].max():.0%} sure",
                     fontsize=9, color=INK)
    suffix = f" ({limit} shown, spread evenly)" if len(wrong) > limit else ""
    fig.suptitle(f"{title}: {len(wrong)}{suffix}", color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return len(wrong)


def plot_augmentations(sample, strength, path, n=7, seed=1):
    rng = np.random.default_rng(seed)
    crop = cv2.imread(str(sample.path))
    images = [crop] + [augment(crop, rng, strength) for _ in range(n)]
    fig, axes = plt.subplots(1, n + 1, figsize=(2 * (n + 1), 2.4), facecolor="#fcfcfb")
    for k, (ax, image) in enumerate(zip(axes, images)):
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax.set_title("original" if k == 0 else f"augmented {k}", fontsize=9, color=INK)
        ax.axis("off")
    fig.suptitle(f'Augmentation: "{strength}"', color=INK, x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def export_onnx(model, mode, path, check_inputs):
    """Export with the preprocessing mode in the metadata, then compare ONNX Runtime with PyTorch.

    Returns the max logit difference.
    """
    import onnx
    import onnxruntime as ort

    model = model.cpu().eval()
    example = torch.zeros(1, MODES[mode], IMAGE_SIZE, IMAGE_SIZE)
    torch.onnx.export(model, (example,), str(path), input_names=["image"], output_names=["logits"],
                      dynamic_shapes={"x": {0: torch.export.Dim("batch")}}, dynamo=True,
                      external_data=False, verbose=False)
    proto = onnx.load(str(path))
    for key, value in {"preprocess_mode": mode, "classes": ",".join(LABELS)}.items():
        entry = proto.metadata_props.add()
        entry.key, entry.value = key, value
    onnx.save(proto, str(path))

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        expected = model(check_inputs).numpy()
    got = session.run(None, {"image": check_inputs.numpy()})[0]
    if not np.array_equal(expected.argmax(1), got.argmax(1)):
        raise RuntimeError("ONNX and PyTorch disagree on a predicted class")
    return float(np.abs(expected - got).max())


def print_counts(name_parts):
    print("            " + "  ".join(f"{l:>4}" for l in LABELS) + "   total")
    for name, part in name_parts:
        counts = count_by_label(part)
        print(f"  {name:<10}" + "  ".join(f"{c:>4}" for c in counts) + f"   {len(part):>5}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train the finger-count CNN.")
    parser.add_argument("--final", action="store_true", help="train on every session, with no test set")
    parser.add_argument("--mode", default=DEFAULT_MODE, choices=list(MODES), help="preprocessing mode")
    parser.add_argument("--augment", default=DEFAULT_AUGMENT, choices=list(COLOR_JITTER))
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--hand", default="R")
    parser.add_argument("--test-person", help="person to hold out when there is more than one")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    samples, excluded = load_samples(hand=args.hand)
    if not samples:
        print("No images found in data/raw/.")
        return 1
    sessions = sorted({s.session for s in samples})
    print(f"Dataset: {len(samples)} images ({args.hand} hand, sessions {', '.join(sessions)}), "
          f"{excluded} excluded by {EXCLUDE.relative_to(ROOT)}")
    for label, count in zip(LABELS, count_by_label(samples)):
        print(f"  {label} fingers: {count}")

    train, val, test, how = split_samples(samples, args.test_person, args.seed, holdout=not args.final)
    print(f"\nSplit: {how}")
    print_counts((("train", train), ("validation", val), ("test", test)))
    print(f"Preprocessing: {args.mode}; augmentation: {args.augment}")

    device = pick_device()
    model_size = sum(p.numel() for p in FingerCNN(in_channels=MODES[args.mode]).parameters())
    print(f"\nDevice: {device}; model: {model_size:,} parameters ({model_size * 4 / 1e6:.2f} MB as float32)")
    start = time.monotonic()
    model, history, best_epoch = fit(train, val, args.mode, args.augment, args.epochs, args.seed, device)
    print(f"Trained in {time.monotonic() - start:.0f} s; keeping epoch {best_epoch} (best validation accuracy)")

    MODELS.mkdir(exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    state = {k: v.cpu() for k, v in model.state_dict().items()}
    torch.save({"state_dict": state, "classes": list(LABELS), "image_size": IMAGE_SIZE,
                "preprocess_mode": args.mode, "augment": args.augment, "epoch": best_epoch,
                "sessions": sessions, "final": args.final}, MODELS / "finger_cnn.pt")
    check_set = test or val
    check_inputs = torch.stack([CropDataset(check_set, args.mode)[i][0] for i in range(len(check_set))])
    onnx_diff = export_onnx(model, args.mode, MODELS / "finger_cnn.onnx", check_inputs)
    model.to(device)

    prefix = "final_" if args.final else ""
    plot_curves(history, best_epoch, FIGURES / f"{prefix}training_curves.png")
    plot_augmentations(train[len(train) // 2], args.augment, FIGURES / f"{prefix}augmentation_examples.png")
    metrics = {"split": how, "sessions": sessions, "final": args.final, "preprocess_mode": args.mode,
               "augment": args.augment,
               "counts": {n: count_by_label(p) for n, p in (("train", train), ("val", val), ("test", test))},
               "excluded": excluded, "best_epoch": best_epoch, "parameters": model_size,
               "onnx_max_logit_difference": onnx_diff, "device": str(device), "history": history,
               "args": vars(args)}

    if test:
        probs = predict(model, test, args.mode, device)
        accuracy, per_class, cm = score(test, probs)
        plot_confusion(cm, per_class, accuracy, FIGURES / "confusion_matrix.png")
        plot_misclassified(test, probs, FIGURES / "misclassified.png")
        confusions = sorted(((int(cm[i, j]), LABELS[i], LABELS[j]) for i in range(6) for j in range(6)
                             if i != j and cm[i, j]), reverse=True)
        metrics.update({"test_accuracy": accuracy, "per_class_accuracy": dict(zip(LABELS, per_class.round(4).tolist())),
                        "confusion_matrix": cm.tolist(),
                        "confusions": [f"true {t} -> predicted {p}: {n}" for n, t, p in confusions]})
        print(f"\nTest accuracy: {accuracy:.1%}")
        for label, a, n in zip(LABELS, per_class, cm.sum(1)):
            print(f"  {label} fingers: {a:.1%} of {n}")
        print("Confusions (true -> predicted):", ", ".join(f"{t}->{p} x{n}" for n, t, p in confusions) or "none")
    else:
        print("\nNo test set (final model). For an honest accuracy estimate, see training/cross_session.py.")
    (MODELS / "finger_cnn_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    print(f"ONNX check on {len(check_set)} {'test' if test else 'validation'} images: same predictions, "
          f"max logit difference {onnx_diff:.1e} (tolerance {ONNX_TOLERANCE:.0e})")
    if onnx_diff > ONNX_TOLERANCE:
        print("ONNX output differs from PyTorch by more than the tolerance.")
        return 1
    print(f"Wrote {MODELS}/finger_cnn.pt, finger_cnn.onnx, finger_cnn_metrics.json and figures in {FIGURES}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
