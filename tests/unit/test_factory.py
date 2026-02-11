"""Tests for factory function create_streaming_pipeline."""

from unittest.mock import MagicMock

import torch
import pytest

from chatterstream.types import StreamConfig


def _make_mock_model():
    model = MagicMock()
    model.t3 = MagicMock()
    model.t3.hp = MagicMock()
    model.t3.hp.start_speech_token = 6561
    model.t3.hp.stop_speech_token = 6562
    model.t3.hp.speech_tokens_dict_size = 6563
    model.t3.device = torch.device("cpu")
    model.s3gen = MagicMock()
    model.s3gen.device = torch.device("cpu")
    model.tokenizer = MagicMock()
    model.device = "cpu"
    model.sr = 24000
    return model


class TestFactory:
    def test_default_config(self):
        from chatterstream.factory import create_streaming_pipeline
        from chatterstream.pipeline import StreamingPipeline
        model = _make_mock_model()
        pipeline = create_streaming_pipeline(model)
        assert isinstance(pipeline, StreamingPipeline)

    def test_custom_config(self):
        from chatterstream.factory import create_streaming_pipeline
        from chatterstream.pipeline import StreamingPipeline
        model = _make_mock_model()
        config = StreamConfig(first_chunk_tokens=10, subsequent_chunk_tokens=50)
        pipeline = create_streaming_pipeline(model, config=config)
        assert isinstance(pipeline, StreamingPipeline)

    def test_return_type(self):
        from chatterstream.factory import create_streaming_pipeline
        from chatterstream.interfaces import StreamingPipelineBase
        model = _make_mock_model()
        pipeline = create_streaming_pipeline(model)
        assert isinstance(pipeline, StreamingPipelineBase)
