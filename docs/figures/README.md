# Figures

All accuracy numbers are on images the model never trained on. Results with
"3 seeds summed" add up three training runs with different random seeds.

## Does it work across sessions? (`training/cross_session.py`)

| File | Shows |
| --- | --- |
| `cross_session_accuracy.png` | Every setup tried, trained on one of Ceci's sessions and tested on the other, both directions |
| `cross_session_confusion_s1_to_s2.png`, `cross_session_confusion_s2_to_s1.png` | Confusion matrices for the winning setup (fixed scaling, strong augmentation) |
| `cross_session_misclassified_s1_to_s2.png`, `cross_session_misclassified_s2_to_s1.png` | Its mistakes, from seed 0 |

## Does it work on a new person? (`training/generalization.py`)

| File | Shows |
| --- | --- |
| `generalization_summary.png` | All the numbers below in one chart |
| `new_person_train-ceci_test-paige_confusion.png` | Trained on Ceci only, tested on Paige |
| `new_person_train-ceci_test-paige_misclassified.png` | Its mistakes |
| `add_person_test-ceci-s1_without-paige_*.png`, `add_person_test-ceci-s1_with-paige_*.png` | Testing on Ceci s1, training without and then with Paige |
| `add_person_test-ceci-s2_without-paige_*.png`, `add_person_test-ceci-s2_with-paige_*.png` | The same, testing on Ceci s2 |

## The final model (`training/train.py --final`, trained on everyone, no test set)

| File | Shows |
| --- | --- |
| `final_training_curves.png` | Loss and accuracy per epoch; validation comes from the same sessions, so it is optimistic |
| `final_augmentation_examples.png` | What strong augmentation does to one image |
| `network_walkthrough/` | Stop-motion frames of one hand going through the final model |

## Out of date

`confusion_matrix.png`, `misclassified.png`, `training_curves.png` and
`augmentation_examples.png` are from the earlier 45% run (train s1, test s2,
mild augmentation). They are kept for the before/after story; rerunning
`training/train.py` without `--final` replaces them.
