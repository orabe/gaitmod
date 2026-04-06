# Seq2VecLSTM

Source implementation:
- `gaitmod/models/seq2vec_lstm.py`

Corresponding hyperparameter config:
- `gaitmod/configs/hparams_configs/hparams_seq2vec_lstm.json`

## Input (2D only)

`Seq2VecLSTM` is a sequence-to-vector binary classifier used here in single-channel mode.

- Expected external input format: `(n_samples, n_timesteps)`
- This document treats the model interface as 2D input only.

## Network topology (from code)

Model is built as a Keras `Sequential` stack:

1. `Input(shape=input_shape)`
2. For each LSTM layer `i` in `hidden_dims`:
   - `LSTM(units=hidden_dims[i], activation=activations[i], recurrent_activation=recurrent_activations[i], return_sequences=(i < last_layer))`
   - `Dropout(rate=dropout)`
3. `Dense(units=dense_units, activation=dense_activation)`

Output is a single sigmoid unit by default (`dense_units=1`, `dense_activation='sigmoid'`).

## Compile settings (from code)

- Loss: `binary_crossentropy` (default)
- Optimizer options: `adam`, `RMSprop`, `SGD` (config currently uses `adam`)
- Metrics:
  - `accuracy`
  - `balanced_accuracy`
  - `f1_score`
  - `precision`
  - `recall`
  - `roc_auc`
  - `pr_auc`

## Config-driven architecture search space

From `hparams_seq2vec_lstm.json` under `Seq2VecLSTM.architecture_configs`:

1. `[8]` with `activations=['tanh']`, `recurrent_activations=['sigmoid']`
2. `[12]` with `activations=['tanh']`, `recurrent_activations=['sigmoid']`
3. `[16]` with `activations=['tanh']`, `recurrent_activations=['sigmoid']`
4. `[16, 8]` with `activations=['tanh','tanh']`, `recurrent_activations=['sigmoid','sigmoid']`

From `Seq2VecLSTM.other_params`:

- `dropout`: `[0.3, 0.5]`
- `dense_units`: `[1]`
- `dense_activation`: `['sigmoid']`
- `optimizer`: `['adam']`
- `lr`: `[0.001, 0.0005]`
- `patience`: `[15]`
- `epochs`: `[120]`
- `batch_size`: `[16, 32]`
- `threshold`: `[0.5]`

From `Seq2VecLSTM.feature_params`:

- `scaler__scaler_type`: `['standard']`

## Two concrete configuration cases

Case A is the simpler baseline. Case B is the more complex variant.

### Case A (Simple): compact 1-layer LSTM

Concrete config values:

- `classifier__hidden_dims`: `[8]`
- `classifier__activations`: `['tanh']`
- `classifier__recurrent_activations`: `['sigmoid']`
- `classifier__dropout`: `0.3`
- `classifier__dense_units`: `1`
- `classifier__dense_activation`: `'sigmoid'`
- `classifier__optimizer`: `'adam'`
- `classifier__lr`: `0.001`
- `classifier__batch_size`: `16`
- `classifier__epochs`: `120`
- `classifier__threshold`: `0.5`

Architecture diagram:

```mermaid
flowchart LR
    A["Input (n_samples, n_timesteps)"] --> B["LSTM(8)\nact=tanh\nrecurrent=sigmoid\nreturn_sequences=False"]
    B --> C["Dropout(0.3)"]
    C --> D["Dense(1, sigmoid)"]
    D --> E["Binary probability"]
```

### Case B (Complex): 2-layer LSTM

Concrete config values:

- `classifier__hidden_dims`: `[16, 8]`
- `classifier__activations`: `['tanh', 'tanh']`
- `classifier__recurrent_activations`: `['sigmoid', 'sigmoid']`
- `classifier__dropout`: `0.5`
- `classifier__dense_units`: `1`
- `classifier__dense_activation`: `'sigmoid'`
- `classifier__optimizer`: `'adam'`
- `classifier__lr`: `0.0005`
- `classifier__batch_size`: `32`
- `classifier__epochs`: `120`
- `classifier__threshold`: `0.5`

Architecture diagram:

```mermaid
flowchart LR
    A["Input (n_samples, n_timesteps)"] --> B["LSTM(16)\nact=tanh\nrecurrent=sigmoid\nreturn_sequences=True"]
    B --> C["Dropout(0.5)"]
    C --> D["LSTM(8)\nact=tanh\nrecurrent=sigmoid\nreturn_sequences=False"]
    D --> E["Dropout(0.5)"]
    E --> F["Dense(1, sigmoid)"]
    F --> G["Binary probability"]
```

## Effective architecture summary for current config

The configured experiments use tanh/sigmoid LSTM cells with 1-2 recurrent layers, dropout after every LSTM layer, and a 1-unit sigmoid dense head for binary segment-level prediction in single-channel mode.

## 3D-style text illustration (draw.io-like)

For sequence tensors in this illustration:

- `w`: temporal length (`T`)
- `h`: signal height, fixed to `1` for 1D inputs
- `d`: channel/feature depth (`C`)

```txt
Case A (Simple): compact 1-layer LSTM

                       [d = channels/features (C)]
                                  ^
                                  |
        [h = signal height (1)]   |

        +-----------------------------------+
       /                                   /|
      /   Input                            / |
     +-----------------------------------+  |
     | out: (B, T, 1)                    |  |
     | geom: w=T, h=1, d=1               |  |
     | source: raw 1D segment            |  +
     |                                   | /
     +-----------------------------------+/
        <-------- [w = temporal length (T)] -------->
                     |
                     v
        +-----------------------------------+
       /                                   /|
      /   LSTM(8)                          / |
     +-----------------------------------+  |
     | in:  (B, T, 1)                    |  |
     | out: (B, 8)                       |  +
     | act=tanh, recurrent=sigmoid       | /
     | return_sequences=False            | /
     +-----------------------------------+/
                     |
                     v
        +-----------------------------------+
       /                                   /|
      /   Dropout                          / |
     +-----------------------------------+  |
     | in:  (B, 8)                       |  |
     | out: (B, 8)                       |  +
     | rate=0.3                           | /
     +-----------------------------------+/
                     |
                     v
             Output Layer: 1 neuron (sigmoid)
                         (o)
           in:  (B, 8) -> out: (B, 1)

               Binary probability


Case B (Complex): 2-layer LSTM

                       [d = channels/features (C)]
                                  ^
                                  |
        [h = signal height (1)]   |

        +-----------------------------------+
       /                                   /|
      /   Input                            / |
     +-----------------------------------+  |
     | out: (B, T, 1)                    |  |
     | geom: w=T, h=1, d=1               |  |
     | source: raw 1D segment            |  +
     |                                   | /
     +-----------------------------------+/
        <-------- [w = temporal length (T)] -------->
                     |
                     v
        +-----------------------------------+
       /                                   /|
      /   LSTM(16)                         / |
     +-----------------------------------+  |
     | in:  (B, T, 1)                    |  |
     | out: (B, T, 16)                   |  +
     | act=tanh, recurrent=sigmoid       | /
     | return_sequences=True             | /
     +-----------------------------------+/
                     |
                     v
        +-----------------------------------+
       /                                   /|
      /   Dropout                          / |
     +-----------------------------------+  |
     | in:  (B, T, 16)                   |  |
     | out: (B, T, 16)                   |  +
     | rate=0.5                           | /
     +-----------------------------------+/
                     |
                     v
        +-----------------------------------+
       /                                   /|
      /   LSTM(8)                          / |
     +-----------------------------------+  |
     | in:  (B, T, 16)                   |  |
     | out: (B, 8)                       |  +
     | act=tanh, recurrent=sigmoid       | /
     | return_sequences=False            | /
     +-----------------------------------+/
                     |
                     v
        +-----------------------------------+
       /                                   /|
      /   Dropout                          / |
     +-----------------------------------+  |
     | in:  (B, 8)                       |  |
     | out: (B, 8)                       |  +
     | rate=0.5                           | /
     +-----------------------------------+/
                     |
                     v
             Output Layer: 1 neuron (sigmoid)
                         (o)
           in:  (B, 8) -> out: (B, 1)

               Binary probability
```

