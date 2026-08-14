# Results

This directory contains the numerical results reported in the paper.

## Global Performance (Test Set)

| Metric    | BETO base | BETO + DialectLex | Δ        |
|-----------|-----------|-------------------|----------|
| Accuracy  | 0.9120    | **0.9220**        | +1.0 pp  |
| Precision | 0.8992    | **0.9306**        | +3.1 pp  |
| Recall    | **0.9280**| 0.9120            | −1.6 pp  |
| F1-score  | 0.9134    | **0.9212**        | +0.8 pp  |
| Loss      | 0.4851    | **0.2654**        | −45.4%   |

## Performance by Dialectal Variant

| Country       | BETO F1 | DialectLex F1 | Δ F1    |
|---------------|---------|---------------|---------|
| 🇲🇽 México    | 0.5333  | **0.8235**    | +29.0pp |
| 🇨🇱 Chile     | 0.1818  | **0.6250**    | +44.3pp |
| 🇦🇷 Argentina | 0.4615  | **0.6667**    | +20.5pp |
| 🇨🇴 Colombia  | 0.5714  | **0.5882**    | +1.7pp  |
| 🇪🇸 España    | 0.3333  | **0.4615**    | +12.8pp |
| 🇵🇪 Perú      | 0.4615  | **0.5714**    | +10.9pp |
| 🇺🇸 EE.UU.    | 0.3333  | **0.4615**    | +12.8pp |

> Improvement in **6 of 7** dialectal variants.

## Training Curves

### BETO Base (Best checkpoint: epoch 4)

| Epoch | Accuracy | Precision | Recall | F1     |
|-------|----------|-----------|--------|--------|
| 1     | 0.9020   | 0.9140    | 0.8880 | 0.9010 |
| 2     | 0.9080   | 0.8860    | 0.9360 | 0.9110 |
| 3     | 0.9040   | 0.9010    | 0.9080 | 0.9040 |
| **4** | **0.9240**|**0.9490** |**0.8960**|**0.9220**|

### BETO + DialectLex (Best checkpoint: epoch 3)

| Epoch | Accuracy | Precision | Recall | F1     |
|-------|----------|-----------|--------|--------|
| 1     | 0.8960   | 0.9200    | 0.8680 | 0.8930 |
| 2     | 0.9100   | 0.9360    | 0.8800 | 0.9070 |
| **3** | **0.9180**|**0.9600** |**0.8720**|**0.9140**|

## How to Reproduce

Run the main notebook end-to-end:

```bash
jupyter nbconvert --to notebook --execute notebooks/train_beto.ipynb
```

Or open it in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/dialectlex-spanish/blob/main/notebooks/train_beto.ipynb)
