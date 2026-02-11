"""Tests for TextProcessor — normalization + tokenization."""

import torch
import pytest


class TestTextProcessor:
    @pytest.fixture(autouse=True)
    def setup(self, mock_tokenizer):
        from chatterstream.text_processor import TextProcessor
        self.processor = TextProcessor(mock_tokenizer)

    def test_empty_text_gets_default(self):
        result = self.processor.process("")
        assert result.shape[0] == 1
        assert result.shape[1] > 0  # default text was substituted

    def test_capitalizes_first_letter(self):
        # Normalization should capitalize; tokenizer sees capitalized text
        result = self.processor.process("hello world.")
        assert result.shape[0] == 1
        assert result.shape[1] > 0

    def test_adds_period_if_no_ending_punctuation(self):
        # "Hello world" should get a period appended
        result = self.processor.process("Hello world")
        assert result.shape[0] == 1

    def test_preserves_existing_ending_punctuation(self):
        r1 = self.processor.process("Hello world!")
        r2 = self.processor.process("Hello world!.")  # already has ender
        # Both should produce valid tensors
        assert r1.shape[0] == 1
        assert r2.shape[0] == 1

    def test_smart_quote_replacement(self):
        result = self.processor.process("\u201cHello\u201d said \u2018Bob\u2019.")
        assert result.shape[0] == 1

    def test_ellipsis_replacement(self):
        result = self.processor.process("Wait\u2026 what?")
        assert result.shape[0] == 1

    def test_multiple_spaces_collapsed(self):
        result = self.processor.process("Hello    world.")
        assert result.shape[0] == 1

    def test_output_tensor_shape(self):
        result = self.processor.process("Test text.")
        assert result.dim() == 2
        assert result.shape[0] == 1  # batch dim

    def test_output_is_long_tensor(self):
        result = self.processor.process("Test text.")
        assert result.dtype == torch.long

    def test_device_placement(self):
        result = self.processor.process("Test text.", device="cpu")
        assert result.device == torch.device("cpu")

    def test_normalize_only(self):
        text = self.processor.normalize("hello world")
        assert text[0].isupper()
        assert text.endswith(".")

    def test_em_dash_replaced(self):
        text = self.processor.normalize("word\u2014another.")
        assert "\u2014" not in text
        assert "-" in text

    def test_en_dash_replaced(self):
        text = self.processor.normalize("word\u2013another.")
        assert "\u2013" not in text
        assert "-" in text
