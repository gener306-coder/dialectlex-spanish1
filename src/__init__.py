"""
DialectLex — Dialectal Bias Evaluation in Spanish Toxicity Detection
Universidad Politécnica de Atlacomulco · Grupo VerbaNexAI · Programa Delfín 2026
"""

from .dialectlex import DialectLex
from .preprocessing import preprocess, preprocess_batch
from .evaluation import compute_metrics, compute_dialectal_bias, print_bias_report

__version__ = "1.0.0"
__authors__ = [
    "Ana Lizbeth Hernández Hernández",
    "Brian Leonardo Martínez Cobos",
    "José Eduardo López Aguilar",
]
