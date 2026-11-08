"""Cross-session test: train on one session, test on the other, in both directions.

    python training/cross_session.py                 # every setup, 3 seeds each (about 20 min)
    python training/cross_session.py --setups before channel_strong --seeds 1

Uses one person's two sessions (the person who has two; --person to choose).
Other people are left out, so the numbers measure lighting and session changes only.

Validation (choosing the epoch) uses only the training session. Each setup runs
with several seeds because a 300-image training set gives noisy results.

Writes models/cross_session_results.json, docs/figures/cross_session_accuracy.png
and, for the best setup, docs/figures/cross_session_confusion_<train>_to_<test>.png.

Note: the "best" setup is picked by these same test-session results, so its
numbers are slightly optimistic. With only two sessions there is nothing else
to pick it with.
"""
import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from training.data import EXCLUDE, LABELS, _tail_split, load_samples, read_exclusions  # noqa: E402
from training.train import (BLUE, FIGURES, INK, INK_2, MODELS, ORANGE, fit, pick_device,  # noqa: E402
                            plot_confusion, plot_misclassified, predict, score, style)

# The five exclusions in use before the cut-off-finger review, to reproduce the "before" run.
ORIGINAL_EXCLUSIONS = [
    "ceci/1/ceci_R_s1_20261106-174446-420.png",
    "ceci/2/ceci_R_s1_20261106-174548-790.png",
    "ceci/4/ceci_R_s1_20261106-174739-930.png",
    "ceci/4/ceci_R_s1_20261106-174740-162.png",
    "ceci/4/ceci_R_s1_20261106-174740-391.png",
]
SETUPS = {  # name: (description, preprocessing mode, augmentation, exclusions)
    "before":         ("Before: fixed scaling, mild augmentation, original exclusions", "fixed", "mild", "original"),
    "excluded":       ("+ cut-off 4s excluded", "fixed", "mild", "current"),
    "fixed_strong":   ("+ strong augmentation", "fixed", "strong", "current"),
    "image_strong":   ("+ strong aug, per-image normalization", "image", "strong", "current"),
    "channel_strong": ("+ strong aug, per-channel normalization", "channel", "strong", "current"),
    "gray_strong":    ("+ strong aug, grayscale, per-image normalization", "gray", "strong", "current"),
}
AFTER = ["fixed_strong", "image_strong", "channel_strong", "gray_strong"]


def load(exclusions, person):
    if exclusions == "current":
        samples = load_samples(exclude=EXCLUDE)[0]
    else:
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write("\n".join(ORIGINAL_EXCLUSIONS) + "\n")
        samples = load_samples(exclude=f.name)[0]
    return [s for s in samples if s.person == person]


def pick_person(requested=None):
    """The person whose sessions are compared: the one with exactly two sessions."""
    sessions = {}
    for s in load_samples(exclude=EXCLUDE)[0]:
        sessions.setdefault(s.person, set()).add(s.session)
    if requested:
        return requested, sorted(sessions.get(requested, ()))
    two = [p for p, ss in sessions.items() if len(ss) == 2]
    if len(two) != 1:
        raise SystemExit(f"use --person: sessions per person are {dict((p, sorted(ss)) for p, ss in sessions.items())}")
    return two[0], sorted(sessions[two[0]])


def plot_summary(results, setups, directions, path):
    fig, ax = plt.subplots(figsize=(10, 0.75 * len(setups) + 1.6), facecolor="#fcfcfb")
    y = np.arange(len(setups))[::-1]
    height = 0.36
    for k, (direction, color) in enumerate(zip(directions, (BLUE, ORANGE))):
        accs = [results[s][direction]["accuracies"] for s in setups]
        means = [np.mean(a) for a in accs]
        ys = y + (0.5 - k) * height
        ax.barh(ys, means, height=height * 0.9, color=color, label=f"train {direction[0]} -> test {direction[1]}")
        for yy, a, m in zip(ys, accs, means):
            if len(a) > 1:
                ax.plot([min(a), max(a)], [yy, yy], color=INK, linewidth=1)
            ax.text(max(a) + 0.01, yy, f"{m:.0%}", va="center", fontsize=9, color=INK)
    ax.set_yticks(y, [SETUPS[s][0] for s in setups], fontsize=9)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("Test accuracy on the other session (bar = mean over seeds, line = range)", color=INK_2)
    ax.set_title("Cross-session accuracy", color=INK, loc="left")
    style(ax)
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--setups", nargs="+", default=list(SETUPS), choices=list(SETUPS))
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--person", help="whose two sessions to compare")
    parser.add_argument("--plot-only", action="store_true", help="redraw the summary figure from the saved JSON")
    args = parser.parse_args(argv)

    if args.plot_only:
        saved = json.loads((MODELS / "cross_session_results.json").read_text())
        setups = [s for s in SETUPS if s in saved["setups"]]
        keys = list(saved["setups"][setups[0]]["directions"])
        directions = [tuple(k.split("->")) for k in keys]
        results = {s: {d: {"accuracies": saved["setups"][s]["directions"][k]["accuracies"]}
                       for d, k in zip(directions, keys)} for s in setups}
        plot_summary(results, setups, directions, FIGURES / "cross_session_accuracy.png")
        best = saved["best_setup"]
        for d, k in zip(directions, keys):
            cm = np.array(saved["setups"][best]["directions"][k]["confusion_sum"])
            plot_confusion(cm, cm.diagonal() / np.maximum(cm.sum(1), 1), cm.trace() / cm.sum(),
                           FIGURES / f"cross_session_confusion_{d[0]}_to_{d[1]}.png",
                           title=f"Train {d[0]}, test {d[1]} ({best}, {saved['seeds']} seeds summed)")
        return 0

    device = pick_device()
    person, sessions = pick_person(args.person)
    if len(sessions) != 2:
        print(f"{person} has sessions {sessions}; this needs exactly two")
        return 1
    data = {name: load(name, person) for name in {SETUPS[s][3] for s in args.setups}}
    directions = [(sessions[0], sessions[1]), (sessions[1], sessions[0])]
    print(f"Device: {device}; person {person}, sessions {sessions}; {args.seeds} seed(s) per run")

    results = {}
    start = time.monotonic()
    for setup in args.setups:
        _, mode, strength, exclusions = SETUPS[setup]
        results[setup] = {}
        for train_session, test_session in directions:
            pool = [s for s in data[exclusions] if s.session == train_session]
            test = [s for s in data[exclusions] if s.session == test_session]
            train, val, _ = _tail_split(pool, 0.15)
            accs, per_class, cms, epochs, last_probs = [], [], [], [], None
            for seed in range(args.seeds):
                model, _, best_epoch = fit(train, val, mode, strength, args.epochs, seed, device, log=None)
                probs = predict(model, test, mode, device)
                acc, pc, cm = score(test, probs)
                accs.append(acc), per_class.append(pc), cms.append(cm), epochs.append(best_epoch)
                last_probs = probs
            key = f"{train_session}->{test_session}"
            results[setup][(train_session, test_session)] = {
                "accuracies": accs, "per_class_mean": np.mean(per_class, 0), "confusion_sum": np.sum(cms, 0),
                "best_epochs": epochs, "n_train": len(train), "n_val": len(val), "n_test": len(test),
                "test": test, "probs_last_seed": last_probs}
            print(f"{setup:<15} {key}: mean {np.mean(accs):.1%} (seeds: {', '.join(f'{a:.1%}' for a in accs)}) "
                  f"| per class {' '.join(f'{p:.0%}' for p in np.mean(per_class, 0))} "
                  f"| train {len(train)}, test {len(test)} | {time.monotonic() - start:.0f} s")

    after = [s for s in AFTER if s in results]
    best = max(after, key=lambda s: np.mean([np.mean(results[s][d]["accuracies"]) for d in directions])) if after else None

    FIGURES.mkdir(parents=True, exist_ok=True)
    plot_summary(results, args.setups, directions, FIGURES / "cross_session_accuracy.png")
    if best:
        for d in directions:
            r = results[best][d]
            cm = r["confusion_sum"]
            per_class = cm.diagonal() / np.maximum(cm.sum(1), 1)
            plot_confusion(cm, per_class, cm.trace() / cm.sum(), FIGURES / f"cross_session_confusion_{d[0]}_to_{d[1]}.png",
                           title=f"Train {d[0]}, test {d[1]} ({best}, {args.seeds} seeds summed)")
            plot_misclassified(r["test"], r["probs_last_seed"], FIGURES / f"cross_session_misclassified_{d[0]}_to_{d[1]}.png",
                               title=f"Train {d[0]}, test {d[1]}: misclassified")

    out = {"note": "best setup chosen on these test sessions, so its numbers are slightly optimistic",
           "person": person, "best_setup": best, "seeds": args.seeds, "epochs": args.epochs, "device": str(device), "setups": {}}
    for setup in args.setups:
        out["setups"][setup] = {"description": SETUPS[setup][0], "mode": SETUPS[setup][1],
                                "augment": SETUPS[setup][2], "exclusions": SETUPS[setup][3], "directions": {}}
        for d in directions:
            r = results[setup][d]
            out["setups"][setup]["directions"][f"{d[0]}->{d[1]}"] = {
                "mean_accuracy": float(np.mean(r["accuracies"])), "accuracies": r["accuracies"],
                "per_class_mean": dict(zip(LABELS, np.round(r["per_class_mean"], 4).tolist())),
                "confusion_sum": r["confusion_sum"].tolist(), "best_epochs": r["best_epochs"],
                "n_train": r["n_train"], "n_val": r["n_val"], "n_test": r["n_test"]}
    MODELS.mkdir(exist_ok=True)
    (MODELS / "cross_session_results.json").write_text(json.dumps(out, indent=2) + "\n")
    if best:
        print(f"\nBest setup after the changes: {best} ({SETUPS[best][0]})")
    print(f"Wrote {MODELS / 'cross_session_results.json'} and figures in {FIGURES}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
