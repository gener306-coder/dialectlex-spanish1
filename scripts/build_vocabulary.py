"""
build_vocabulary.py — Build the DialectLex vocabulary from BILMA TSV files.

Usage:
    python scripts/build_vocabulary.py \
        --bilma-dir path/to/bilma/ \
        --output vocabulary.json \
        --min-freq 5

Downloads BILMA from: https://ingeotec.github.io/regional-spanish-models/
"""

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.dialectlex import DialectLex

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Build DialectLex vocabulary from BILMA files."
    )
    parser.add_argument("--bilma-dir", required=True,
                        help="Directory containing BILMA TSV files (MX.tsv, AR.tsv, ...)")
    parser.add_argument("--output",    default="vocabulary.json",
                        help="Output JSON file path (default: vocabulary.json)")
    parser.add_argument("--min-freq",  type=int, default=5,
                        help="Minimum term frequency (default: 5)")
    parser.add_argument("--min-len",   type=int, default=2,
                        help="Minimum token length (default: 2)")
    parser.add_argument("--max-len",   type=int, default=30,
                        help="Maximum token length (default: 30)")
    args = parser.parse_args()

    logger.info(f"Building vocabulary from: {args.bilma_dir}")
    logger.info(f"Filters: min_freq={args.min_freq}, "
                f"len=[{args.min_len}, {args.max_len}]")

    dlx = DialectLex(
        bilma_dir=args.bilma_dir,
        min_freq=args.min_freq,
        min_len=args.min_len,
        max_len=args.max_len,
    )

    logger.info(f"Saving vocabulary ({dlx.get_vocabulary_size():,} terms) "
                f"to: {args.output}")
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(dlx.vocabulary, f, ensure_ascii=False)

    logger.info("Done.")


if __name__ == "__main__":
    main()
