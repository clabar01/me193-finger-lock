"""Does the model work on someone it has never seen, and does a second person help?

    python training/generalization.py                  # setup from cross_session_results.json, 3 seeds
    python training/generalization.py --mode channel --augment strong --seeds 1

Experiments (the main person is the one with the most sessions):
  new_person    train on the main person (all sessions), test on everyone else.
  add_person    for each of the main person's sessions: test on it, training on
                their other session(s) without and then with the other people.
                The difference is what adding people buys.
Validation (choosing the epoch) always comes from the training images only.

Writes models/generalization_results.json and, in docs/figures/:
  generalization_summary.png
  new_person_train-<main>_test-<others>_confusion.png / _misclassified.png
  add_person_test-<main>-<session>_{without,with}-<others>_confusion.png / _misclassified.png
Confusion matrices are summed over seeds; misclassified examples come from seed 0.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from common.preprocess import MODES  # noqa: E402
from training.data import COLOR_JITTER, LABELS, _tail_split, count_by_label, load_samples  # noqa: E402
from training.train import (BLUE, FIGURES, INK, INK_2, MODELS, ORANGE, fit, pick_device,  # noqa: E402
                            plot_confusion, plot_misclassified, predict, score, style)

GRAY = "#8f8e89"


def best_setup():
    """(mode, augment) of the best setup in models/cross_session_results.json."""
    path = MODELS / "cross_session_results.json"
    results = json.loads(path.read_text())
    setup = results["setups"][results["best_setup"]]
    return setup["mode"], setup["augment"], results["best_setup"]


def run(name, train_pool, test, mode, augment, seeds, epochs, device, title, stem):
    """Train with several seeds and test. Returns a result dict; writes the two figures."""
    train, val, _ = _tail_split(train_pool, 0.15)
    accs, cms, first_probs = [], [], None
    for seed in range(seeds):
        model, _, best_epoch = fit(train, val, mode, augment, epochs, seed, device, log=None)
        probs = predict(model, test, mode, device)
        acc, _, cm = score(test, probs)
        accs.append(acc)
        cms.append(cm)
        first_probs = probs if first_probs is None else first_probs
    cm = np.sum(cms, 0)
    per_class = cm.diagonal() / np.maximum(cm.sum(1), 1)
    plot_confusion(cm, per_class, cm.trace() / cm.sum(), FIGURES / f"{stem}_confusion.png",
                   title=f"{title} ({seeds} seeds summed)")
    plot_misclassified(test, first_probs, FIGURES / f"{stem}_misclassified.png", title=f"{title}: misclassified")
    confusions = sorted(((int(cm[i, j]), LABELS[i], LABELS[j]) for i in range(6) for j in range(6)
                         if i != j and cm[i, j]), reverse=True)
    print(f"{name:<45} mean {np.mean(accs):.1%} (seeds: {', '.join(f'{a:.1%}' for a in accs)}) "
          f"| per class {' '.join(f'{p:.0%}' for p in per_class)} | train {len(train)}, test {len(test)}")
    return {"title": title, "mean_accuracy": float(np.mean(accs)), "accuracies": accs,
            "per_class": dict(zip(LABELS, np.round(per_class, 4).tolist())), "confusion_sum": cm.tolist(),
            "top_confusions": [f"true {t} -> predicted {p}: {n}" for n, t, p in confusions[:6]],
            "n_train": len(train), "n_val": len(val), "n_test": len(test),
            "test_counts": count_by_label(test), "figures": [f"{stem}_confusion.png", f"{stem}_misclassified.png"]}


def plot_summary(rows, path):
    """rows: (label, accuracies, color)."""
    fig, ax = plt.subplots(figsize=(10, 0.6 * len(rows) + 1.8), facecolor="#fcfcfb")
    y = np.arange(len(rows))[::-1]
    for yy, (label, accs, color) in zip(y, rows):
        m = np.mean(accs)
        ax.barh(yy, m, height=0.6, color=color)
        if len(accs) > 1:
            ax.plot([min(accs), max(accs)], [yy, yy], color=INK, linewidth=1)
        ax.text(max(accs) + 0.01, yy, f"{m:.0%}", va="center", fontsize=9, color=INK)
    ax.set_yticks(y, [r[0] for r in rows], fontsize=9)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("Test accuracy (bar = mean over seeds, line = range)", color=INK_2)
    ax.set_title("Generalization: new person, and adding a person", color=INK, loc="left")
    style(ax)
    ax.grid(axis="y", visible=False)
    handles = [matplotlib.patches.Patch(color=c, label=l) for c, l in
               ((GRAY, "new person, never trained on"), (BLUE, "trained without the other person"),
                (ORANGE, "trained with the other person"))]
    ax.legend(handles=handles, frameon=False, loc="upper center", bbox_to_anchor=(0.4, -0.18), ncol=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=list(MODES))
    parser.add_argument("--augment", choices=list(COLOR_JITTER))
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--plot-only", action="store_true",
                        help="redraw the summary and confusion figures from the saved JSON")
    args = parser.parse_args(argv)

    if args.plot_only:
        saved = json.loads((MODELS / "generalization_results.json").read_text())
        runs = [(saved["new_person"], GRAY)] + [(saved["add_person"][h][k], c) for h in saved["add_person"]
                                                for k, c in (("without", BLUE), ("with", ORANGE))]
        for r, _ in runs:
            cm = np.array(r["confusion_sum"])
            plot_confusion(cm, cm.diagonal() / np.maximum(cm.sum(1), 1), cm.trace() / cm.sum(),
                           FIGURES / r["figures"][0], title=f"{r['title']} ({saved['seeds']} seeds summed)")
        plot_summary([(r["title"], r["accuracies"], c) for r, c in runs], FIGURES / "generalization_summary.png")
        return 0

    if args.mode and args.augment:
        mode, augment, source = args.mode, args.augment, "command line"
    else:
        mode, augment, source = best_setup()
        mode, augment = args.mode or mode, args.augment or augment
    samples, excluded = load_samples()
    sessions = {}
    for s in samples:
        sessions.setdefault(s.person, set()).add(s.session)
    if len(sessions) < 2:
        print(f"Only one person in data/raw ({list(sessions)}); this needs at least two.")
        return 1
    main_person = max(sessions, key=lambda p: (len(sessions[p]), sum(s.person == p for s in samples)))
    others = sorted(p for p in sessions if p != main_person)
    others_name = "+".join(others)
    mine = [s for s in samples if s.person == main_person]
    theirs = [s for s in samples if s.person != main_person]
    device = pick_device()
    print(f"Setup: mode {mode!r}, augment {augment!r} (from {source}); device {device}; {args.seeds} seeds")
    print(f"Main person: {main_person}, sessions {sorted(sessions[main_person])}, {len(mine)} images; "
          f"others: {', '.join(f'{p} ({sorted(sessions[p])})' for p in others)}, {len(theirs)} images; "
          f"{excluded} excluded\n")

    FIGURES.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    results = {"setup": {"mode": mode, "augment": augment, "source": source}, "seeds": args.seeds,
               "main_person": main_person, "others": others}
    results["new_person"] = run(
        f"new person: train {main_person}, test {others_name}", mine, theirs, mode, augment, args.seeds, args.epochs,
        device, f"Train {main_person}, test {others_name} (never seen)", f"new_person_train-{main_person}_test-{others_name}")
    rows = [(f"New person: train {main_person} (all sessions), test {others_name}",
             results["new_person"]["accuracies"], GRAY)]

    results["add_person"] = {}
    for held in sorted(sessions[main_person]):
        test = [s for s in mine if s.session == held]
        rest = [s for s in mine if s.session != held]
        if not rest:
            continue
        rest_name = "+".join(sorted({s.session for s in rest}))
        stem = f"add_person_test-{main_person}-{held}"
        without = run(f"test {main_person} {held}: without {others_name}", rest, test, mode, augment, args.seeds,
                      args.epochs, device, f"Train {main_person} {rest_name}, test {main_person} {held}",
                      f"{stem}_without-{others_name}")
        with_ = run(f"test {main_person} {held}: with {others_name}", rest + theirs, test, mode, augment, args.seeds,
                    args.epochs, device, f"Train {main_person} {rest_name} + {others_name}, test {main_person} {held}",
                    f"{stem}_with-{others_name}")
        results["add_person"][held] = {"without": without, "with": with_,
                                       "change": with_["mean_accuracy"] - without["mean_accuracy"]}
        rows += [(f"Test {main_person} {held}: train {main_person} {rest_name}", without["accuracies"], BLUE),
                 (f"Test {main_person} {held}: train {main_person} {rest_name} + {others_name}", with_["accuracies"], ORANGE)]
        print(f"   adding {others_name} changes {main_person} {held} accuracy by {results['add_person'][held]['change']:+.1%}"
              f"  ({time.monotonic() - start:.0f} s)\n")

    plot_summary(rows, FIGURES / "generalization_summary.png")
    (MODELS / "generalization_results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(f"Wrote {MODELS / 'generalization_results.json'} and figures in {FIGURES}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
