# Methodology

## Overview

This document describes the methodology of the **DialectLex** strategy in detail.

---

## 1. Problem Formulation

Given a Spanish text `t`, we want to classify it as toxic (1) or non-toxic (0).
We hypothesize that a model trained only on standard Spanish will exhibit
**dialectal bias**: systematic performance differences when processing
regional variants.

**Research question:** Can prepending an explicit dialect tag to the input text
reduce dialectal bias in binary toxicity classification?

---

## 2. Dataset

| Split | Toxic | Non-toxic | Total |
|-------|-------|-----------|-------|
| Train | 2,000 | 2,000 | 4,000 |
| Validation | 250 | 250 | 500 |
| Test | 250 | 250 | 500 |

Source: Multilingual Toxicity Dataset — PAN 2024 (Spanish partition).

---

## 3. Text Preprocessing

Preprocessing is **intentionally minimal** to preserve dialectal markers.
Only non-linguistic noise is removed:

| Element | Replacement |
|---|---|
| URLs (`https://...`) | `[URL]` |
| @mentions | `@USER` |
| #hashtags | `[HASHTAG]` |
| Numbers | `[NUMBER]` |
| Email addresses | `[EMAIL]` |
| Emojis | `[EMOJI]` |
| Currency symbols | `[CURRENCY]` |

We do **NOT** apply lemmatization, stemming, or stopword removal — these would
destroy dialectal markers like "tenés" (voseo), "güey" (Mexican), or "parcero" (Colombian).

---

## 4. BILMA Dialectal Vocabulary

### Source
Tellez et al. (2022) built BILMA from 795M geolocated tweets across 26 Spanish-speaking
countries. Each TSV file contains:

```
token    occs     ndocs    weight
güey     91823    47201    2.847
chido    38421    21004    2.103
```

### Dialectal Score Formula

$$\text{Score}(t, d) = \log(1 + \text{occs}(t, d)) \times \text{weight}(t, d)$$

- `occs(t, d)`: number of occurrences of token `t` in dialect `d`
- `weight(t, d)`: statistical relevance of `t` in `d` vs. other dialects

### Vocabulary Filters

| Filter | Value | Reason |
|---|---|---|
| Minimum frequency | 5 | Remove noise and rare tokens |
| Minimum length | 2 chars | Remove single characters |
| Maximum length | 30 chars | Remove malformed strings |
| Weight | > 0 | Keep only regionally distinctive tokens |
| Character set | `[a-záéíóúüñ]` | Alphabetic tokens only |

**Final vocabulary:** 1,117,825 unique terms.

---

## 5. DialectLex Enrichment

For each text, the algorithm:
1. Tokenizes the text into lowercase alphabetic tokens.
2. Looks up each token in the dialectal vocabulary.
3. Accumulates scores per dialect.
4. Prepends the tag of the dialect with the highest cumulative score.

```python
scores = {dialect: sum(vocab[token][dialect] for token in text if token in vocab)}
best_dialect = argmax(scores)
enriched_text = f"[{best_dialect}] {original_text}"
```

**Example:**
```
Input:  "ese wey llegó tarde y no se disculpó"
Tokens: ["ese", "wey", "llegó", "tarde", ...]
Scores: MX=2.847 (from "wey"), AR=0, CL=0, ...
Output: "[MX] ese wey llegó tarde y no se disculpó"
```

---

## 6. Model Training

### Architecture
- **Base model:** `dccuchile/bert-base-spanish-wwm-uncased` (BETO)
- **Task head:** `BertForSequenceClassification` with 2 labels
- **Tokenizer:** `BertTokenizerFast`, `max_length=128`

The dialect tags (`[MX]`, `[CO]`, etc.) are added as special tokens to the
tokenizer before fine-tuning, allowing BETO to learn their contextual meaning.

### Hyperparameter Optimization

Performed with `hyperparameter_search` (HuggingFace Transformers):

| Parameter | Optimal Value |
|---|---|
| Learning rate | 2.18 × 10⁻⁵ |
| Weight decay | 0.039 |
| Warmup ratio | 0.047 |
| Batch size | 32 |
| Max epochs | 11 |
| Early stopping patience | 3 |

### Experimental Design

Both models (BETO base and BETO + DialectLex) are trained with **identical
hyperparameters** on the same dataset. The only difference is the preprocessing
step that adds dialect tags. This controlled design ensures that performance
differences are attributable solely to the DialectLex enrichment.

---

## 7. Evaluation

### Standard Metrics
Evaluated on the held-out test set (500 samples, unseen during training):
- Accuracy, Precision, Recall, F1-score
- Confusion matrix (TP, FP, FN, TN)
- Validation loss

### Dialectal Evaluation
A complementary dialectal evaluation set was built with representative
comments from each of the 7 supported variants, containing clear dialectal
markers. Both models were evaluated on this set to measure performance
differences across variants.

### Bias Metric
Delta F1 per variant:

$$\Delta F1(d) = F1_{\text{DialectLex}}(d) - F1_{\text{BETO base}}(d)$$

A positive ΔF1 indicates that DialectLex reduces dialectal bias for variant `d`.
