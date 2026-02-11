"""Text normalization and tokenization for the streaming pipeline."""

from __future__ import annotations

import re

import torch

from .interfaces import TextProcessorBase

# Compiled replacements (single-pass via dict lookup)
_PUNC_REPLACEMENTS = {
    "\u2026": ", ",   # ellipsis
    ":": ",",
    "\u2014": "-",    # em-dash
    "\u2013": "-",    # en-dash
    " ,": ",",
    "\u201c": '"',    # left double smart quote
    "\u201d": '"',    # right double smart quote
    "\u2018": "'",    # left single smart quote
    "\u2019": "'",    # right single smart quote
}

_PUNC_PATTERN = re.compile("|".join(re.escape(k) for k in _PUNC_REPLACEMENTS))
_MULTI_SPACE = re.compile(r"\s+")
_SENTENCE_ENDERS = frozenset(".,!?-")
_DEFAULT_TEXT = "You need to add some text for me to talk."


class TextProcessor(TextProcessorBase):
    """Normalizes text and tokenizes via a GPT2 tokenizer."""

    def __init__(self, tokenizer):
        self._tokenizer = tokenizer

    def normalize(self, text: str) -> str:
        """Replicate upstream punc_norm() logic without importing it."""
        if not text:
            return _DEFAULT_TEXT

        # Capitalize first letter
        if text[0].islower():
            text = text[0].upper() + text[1:]

        # Collapse whitespace
        text = _MULTI_SPACE.sub(" ", text)

        # Replace uncommon punctuation
        text = _PUNC_PATTERN.sub(lambda m: _PUNC_REPLACEMENTS[m.group()], text)

        # Add period if no ending punctuation
        text = text.rstrip()
        if not any(text.endswith(p) for p in _SENTENCE_ENDERS):
            text += "."

        return text

    def process(self, text: str, device: torch.device | str = "cpu") -> torch.Tensor:
        """Normalize text and return token IDs tensor of shape (1, seq_len)."""
        normalized = self.normalize(text)
        encoded = self._tokenizer(normalized, return_tensors="pt")
        return encoded.input_ids.to(device=device, dtype=torch.long)
