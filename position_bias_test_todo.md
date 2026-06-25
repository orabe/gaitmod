# Position Bias / Robustness Test TODO

Use this checklist to track implementation progress for the inter-segment `seq2seq_CNN-LSTM` and `seq2seq_LSTM` models.

- [ ] **1. Position-only baseline**
  - Train a simple classifier using only each segment's normalized position within the trial.
  - Goal: measure how much performance can be explained by trial timing alone.

- [ ] **2. Mask signal, keep position**
  - Replace segment content with noise or a constant signal while keeping the same trial lengths and segment order.
  - Goal: test whether the model still produces modulation-like predictions without meaningful signal.

- [ ] **3. Permutation without labels**
  - Shuffle segment order within each trial, but keep the original labels in place.
  - Goal: break alignment between signal and labels and test whether predictions still follow trial position.

- [ ] **4. Permutation with labels**
  - Shuffle segment order within each trial together with their labels.
  - Goal: preserve signal-label pairing but destroy the original temporal order, testing whether sequence order itself matters.

- [ ] **5. Shift labels only**
  - Move the modulation label block earlier or later within the trial, without changing the signal.
  - Goal: test whether predictions stay tied to the original signal or drift toward the shifted label position.

- [ ] **6. Shift signal and labels together**
  - Move the modulation block earlier or later within the trial together with its signal and labels.
  - Goal: test whether the model can still detect modulation when it appears at a different trial position.

- [ ] **7. Trial cropping**
  - Remove part of the trial from the start, end, or both, while keeping the remaining signal unchanged.
  - Goal: change the relative position of modulation and reduce context to test robustness.

- [ ] **8. Trial concatenation**
  - Join multiple trials into one longer continuous sequence.
  - Goal: remove the clean single-trial template and test whether the model still tracks modulation in a more continuous setting.

## Notes

- Train on the original training data unless a test explicitly requires retraining.
- Prefer evaluation on perturbed test sets first.
- Record for each test:
  - implementation status
  - affected model(s)
  - metrics used
  - main findings
