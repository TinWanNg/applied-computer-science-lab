"""
Utility functions for text processing and metadata handling.
"""
import re
from typing import Dict, Any, Optional


def clean_text(text: str) -> str:
    """Remove extra whitespace, page labels, and broken hyphenation."""
    text = re.sub(r"-\s*\n", "", text)     # remove hyphen line-breaks
    text = re.sub(r"\s+", " ", text)       # collapse whitespace
    text = re.sub(r"Page\s+\d+", "", text) # remove "Page X" footer labels
    return text.strip()


def sanitize_metadata(md: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Chroma only accepts metadata values that are:
      - str, int, float, bool, None, or SparseVector

    This function converts any list/dict/other objects to strings,
    so we don't get errors like:
      "Expected metadata value to be a str, int, float, bool, SparseVector, or None, got ['eng']"
    """
    if md is None:
        return {}

    simple: Dict[str, Any] = {}
    for k, v in md.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            simple[k] = v
        else:
            # Convert lists, dicts, etc. to string representation
            simple[k] = str(v)
    return simple
