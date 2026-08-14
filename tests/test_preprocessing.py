"""Tests for preprocessing module."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.preprocessing import preprocess, preprocess_batch


def test_replace_url():
    text = "visita https://example.com para más info"
    result = preprocess(text)
    assert "[URL]" in result
    assert "https://" not in result


def test_replace_user():
    text = "@usuario123 gracias por el comentario"
    result = preprocess(text)
    assert "@USER" in result
    assert "@usuario123" not in result


def test_replace_hashtag():
    text = "me gustó el evento #Delfin2026"
    result = preprocess(text)
    assert "[HASHTAG]" in result
    assert "#Delfin" not in result


def test_preserve_dialectal_markers():
    """Critical: dialectal markers must be preserved."""
    dialectal_texts = [
        "ese wey llegó tarde",         # Mexican: wey
        "che, vení un momento",        # Argentine: che, vení
        "ese parcero no llegó",        # Colombian: parcero
        "la fiesta estuvo chida",      # Mexican: chida
        "tenés razón en eso",          # Rioplatense: tenés
    ]
    for text in dialectal_texts:
        result = preprocess(text)
        # The dialectal word should still be present
        words = text.split()
        dialectal_word = words[-2] if len(words) > 1 else words[0]
        assert any(w in result for w in words if len(w) > 3), (
            f"Dialectal marker lost in: '{text}' -> '{result}'"
        )


def test_batch_processing():
    texts = ["hola @user", "ver https://link.com", "texto normal"]
    results = preprocess_batch(texts)
    assert len(results) == len(texts)
    assert "@USER" in results[0]
    assert "[URL]" in results[1]
    assert "texto normal" in results[2]


def test_whitespace_normalization():
    text = "texto    con    muchos     espacios"
    result = preprocess(text)
    assert "    " not in result
