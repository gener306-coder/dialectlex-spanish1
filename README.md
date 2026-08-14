# DialectLex — Dialectal Bias Evaluation in Spanish Toxicity Detection

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://python.org)
[![License](https://img.shields.io/badge/License-Apache_2.0-orange.svg)](LICENSE)
[![BETO](https://img.shields.io/badge/Model-BETO%20(dccuchile)-1A3A6B?logo=huggingface)](https://huggingface.co/dccuchile/bert-base-spanish-wwm-uncased)
[![Dataset](https://img.shields.io/badge/Dataset-Multilingual%20Toxicity-green)](https://huggingface.co/datasets/textdetox/multilingual_toxicity_dataset)
[![Delfín](https://img.shields.io/badge/Program-Delfín%202026-E87722)](https://programadelfin.org.mx)

**Evaluación del Sesgo Dialectal en Español para la Detección de Toxicidad mediante BETO**

*Assessment of Dialectal Bias in Spanish for Toxicity Detection Using BETO*

[📄 Paper](#citation) · [🚀 Quick Start](#quick-start) · [📊 Results](#results) · [🗂 Dataset](#dataset)

</div>

---

## 📌 Overview

This repository contains the code and resources for **DialectLex**, a simple and reproducible strategy to mitigate dialectal bias in Spanish toxicity detection models. By enriching input texts with regional dialect tags derived from the **BILMA** corpus, we improve BETO's performance across 7 Spanish-speaking variants without modifying the model's architecture.

| Model | Accuracy | Precision | Recall | F1-score | Loss |
|---|---|---|---|---|---|
| BETO (base) | 91.2% | 89.9% | **92.8%** | 91.3% | 0.485 |
| **BETO + DialectLex** | **92.2%** | **93.1%** | 91.2% | **92.1%** | **0.265** |

> 📈 Improvement across **6 of 7 dialectal variants** — largest gains in Chile (+44.3pp F1), Mexico (+29.1pp) and Argentina (+20.5pp).

---

## 🏗 Methodology

```
                    ┌─────────────────────────────────────┐
                    │         BETO Base Pipeline           │
 Dataset ──────────►│ Preprocessing ► Tokenization ► BETO │──► Predictions
                    └─────────────────────────────────────┘

                    ┌──────────────────────────────────────────────────────┐
                    │           BETO + DialectLex Pipeline                  │
 Dataset ──────────►│ BILMA Vocab ► Text Enrichment ► Tokenization ► BETO+│──► Predictions
                    └──────────────────────────────────────────────────────┘

  Text enrichment: "ese wey llegó tarde" ──► "[MX] ese wey llegó tarde"
```

### Key Components

| Component | Description |
|---|---|
| **BILMA** | Dialectal vocabulary of 1,117,825 terms from 795M geolocated tweets across 26 Spanish-speaking countries |
| **DialectLex** | Strategy that prepends regional tags (`[MX]`, `[CO]`, `[AR]`, etc.) to input texts based on detected dialectal markers |
| **BETO** | Spanish BERT model (`dccuchile/bert-base-spanish-wwm-uncased`) fine-tuned for binary toxicity classification |
| **Dialectal Score** | `Score = log(1 + occs) × weight` — combines term frequency with regional statistical relevance |

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/dialectlex-spanish.git
cd dialectlex-spanish
pip install -r requirements.txt
```

### Run the Training Notebook

Open `notebooks/train_beto.ipynb` in Jupyter or Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/dialectlex-spanish/blob/main/notebooks/train_beto.ipynb)

### Use DialectLex in Your Code

```python
from src.dialectlex import DialectLex

# Initialize with BILMA vocabulary directory
dlx = DialectLex(bilma_dir="path/to/bilma/")

# Enrich a text with dialectal tags
text = "ese wey llegó tarde"
enriched = dlx.enrich(text)
print(enriched)  # "[MX] ese wey llegó tarde"

# Batch enrichment
texts = ["che, vení un momento", "ese man llegó temprano", "qué padre estuvo la fiesta"]
enriched_texts = dlx.enrich_batch(texts)
```

---

## 📁 Project Structure

```
dialectlex-spanish/
├── README.md                    # This file
├── LICENSE                      # Apache 2.0 License
├── requirements.txt             # Python dependencies
├── CITATION.cff                 # How to cite this work
├── .gitignore
│
├── notebooks/
│   └── train_beto.ipynb         # Main training & evaluation notebook
│
├── src/
│   ├── __init__.py
│   ├── dialectlex.py            # DialectLex enrichment strategy
│   ├── preprocessing.py         # Text preprocessing pipeline
│   └── evaluation.py            # Evaluation metrics & bias analysis
│
├── scripts/
│   └── build_vocabulary.py      # Build dialectal vocabulary from BILMA files
│
├── data/
│   └── README.md                # Dataset access instructions
│
├── results/
│   └── README.md                # Results description & reproduction
│
└── docs/
    └── methodology.md           # Detailed methodology documentation
```

---

## 📊 Results

### Global Performance (Test Set)

| Metric | BETO base | BETO + DialectLex | Δ |
|---|---|---|---|
| Accuracy | 91.2% | **92.2%** | +1.0 pp ↑ |
| Precision | 89.9% | **93.1%** | +3.2 pp ↑ |
| Recall | **92.8%** | 91.2% | −1.6 pp ↓ |
| F1-score | 91.3% | **92.1%** | +0.8 pp ↑ |
| Loss | 0.485 | **0.265** | −45.4% ↓ |

### Performance by Dialectal Variant

| Country | BETO F1 | DialectLex F1 | Δ F1 |
|---|---|---|---|
| 🇲🇽 México | 0.533 | **0.824** | +29.1 pp ↑ |
| 🇨🇱 Chile | 0.182 | **0.625** | +44.3 pp ↑ |
| 🇦🇷 Argentina | 0.462 | **0.667** | +20.5 pp ↑ |
| 🇨🇴 Colombia | 0.571 | **0.588** | +1.7 pp ↑ |
| 🇪🇸 España | 0.333 | **0.462** | +12.9 pp ↑ |
| 🇵🇪 Perú | 0.462 | **0.571** | +10.9 pp ↑ |
| 🇺🇸 EE.UU. | 0.333 | **0.462** | +12.9 pp ↑ |

### Training Configuration (Optimal Hyperparameters)

| Parameter | Value |
|---|---|
| Learning rate | 2.18 × 10⁻⁵ |
| Weight decay | 0.039 |
| Warmup ratio | 0.047 |
| Batch size | 32 |
| Epochs (max) | 11 (Early Stopping patience=3) |
| Optimizer | AdamW |
| Base model | `dccuchile/bert-base-spanish-wwm-uncased` |

---

## 🗂 Dataset

This project uses the **Spanish partition** of the [Multilingual Toxicity Dataset](https://huggingface.co/datasets/textdetox/multilingual_toxicity_dataset) from PAN 2024:

- **5,000 comments** — 2,500 toxic · 2,500 non-toxic
- **Split**: 80% train · 10% validation · 10% test
- **License**: CC BY 4.0

The BILMA dialectal vocabulary is available at: https://ingeotec.github.io/regional-spanish-models/

See [`data/README.md`](data/README.md) for full dataset download instructions.

---

## 📚 References

1. Cañete, J. et al. (2020). *Spanish Pre-Trained BERT Model and Evaluation Data*. PML4DC @ ICLR 2020.
2. Tellez, E.S. et al. (2022). *Regionalized models for Spanish language variations based on Twitter*. Language Resources and Evaluation. https://doi.org/10.1007/s10579-023-09640-9
3. Dementieva, D. et al. (2024). *Overview of the Multilingual Text Detoxification Task at PAN 2024*. CLEF 2024.
4. Blodgett, S.L. et al. (2020). *Language (Technology) is Power: A Critical Survey of "Bias" in NLP*. ACL 2020.
5. Fortuna, P. et al. (2022). *How Well Do Hate Speech, Toxicity, Abusive and Offensive Language Classification Models Generalize Across Datasets?* Frontiers in AI.

---

## 📝 Citation

If you use this work in your research, please cite:

```bibtex
@article{hernandez2026dialectlex,
  title     = {Evaluación del Sesgo Dialectal en Español para la Detección
               de Toxicidad mediante BETO y Enriquecimiento Léxico Dialectal basado en BILMA},
  author    = {Hernández Hernández, Ana Lizbeth and
               Martínez Cobos, Brian Leonardo and
               López Aguilar, José Eduardo},
  journal   = {Portal de la Ciencia},
  year      = {2026},
  institution = {Universidad Politécnica de Atlacomulco},
  note      = {Programa Delfín 2026 — Grupo VerbaNexAI, UTB Cartagena}
}
```

---

## 🙏 Acknowledgments

This work was developed during the **Delfín 2026 Program** at the [Universidad Tecnológica de Bolívar](https://utb.edu.co), Cartagena, Colombia, under the supervision of **Dra. Danileth Almanza-González** (Grupo VerbaNexAI).

- Universidad Politécnica de Atlacomulco (UPA)
- Universidad Tecnológica de Bolívar (UTB) — Grupo VerbaNexAI
- Programa Interinstitucional para el Fortalecimiento de la Investigación y el Posgrado del Pacífico (Delfín)

---

## 📜 License

This project is licensed under the **Apache License 2.0** — see the [LICENSE](LICENSE) file for details.
