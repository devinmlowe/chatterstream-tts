"""Token generator: replicates T3 autoregressive loop, yielding TokenChunks."""

from __future__ import annotations

import logging
import time
from typing import Iterator

import torch
import torch.nn.functional as F
from transformers import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

from .interfaces import TokenGeneratorBase
from .types import ChunkStrategy, StreamConfig, TokenChunk

# Upstream constants
_SILENCE_TOKEN = 4299
_MIN_CHUNK_FOR_SENTENCE = 3  # Minimum tokens before sentence-aligned yield


class TokenGenerator(TokenGeneratorBase):
    """Generates speech tokens autoregressively from T3, yielding chunks mid-loop.

    Replicates the logic of ``T3.inference_turbo`` (t3.py:414-490) but yields
    ``TokenChunk`` objects as tokens accumulate to the configured chunk size.
    """

    def __init__(self, t3, config: StreamConfig):
        self._t3 = t3
        self._config = config
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def _build_logits_processors(self, config: StreamConfig) -> LogitsProcessorList:
        processors = LogitsProcessorList()
        if config.temperature > 0 and config.temperature != 1.0:
            processors.append(TemperatureLogitsWarper(config.temperature))
        if config.top_k > 0:
            processors.append(TopKLogitsWarper(config.top_k))
        if config.top_p < 1.0:
            processors.append(TopPLogitsWarper(config.top_p))
        if config.repetition_penalty != 1.0:
            processors.append(RepetitionPenaltyLogitsProcessor(config.repetition_penalty))
        return processors

    def _should_yield(
        self, pending: list[int], chunk_index: int, config: StreamConfig
    ) -> bool:
        """Determine if we should yield a chunk based on strategy and counts."""
        target = config.first_chunk_tokens if chunk_index == 0 else config.subsequent_chunk_tokens

        if config.strategy == ChunkStrategy.SENTENCE_ALIGNED:
            # Yield on silence token if we have minimum tokens
            if len(pending) >= _MIN_CHUNK_FOR_SENTENCE and pending[-1] == _SILENCE_TOKEN:
                return True
            # Fallback to max threshold
            return len(pending) >= target

        # Adaptive: yield at fixed thresholds
        return len(pending) >= target

    def _filter_oov(self, tokens: list[int]) -> list[int]:
        """Remove out-of-vocabulary tokens (>= start_speech_token)."""
        threshold = self._t3.hp.start_speech_token
        return [t for t in tokens if t < threshold]

    def generate(
        self,
        text_tokens: torch.Tensor,
        t3_cond: object,
        config: StreamConfig,
    ) -> Iterator[TokenChunk]:
        self._cancelled = False
        t3 = self._t3
        device = text_tokens.device
        logits_processors = self._build_logits_processors(config)

        _log = logging.getLogger(__name__)

        # --- Prefill ---
        _t0 = time.monotonic()
        speech_start_token = t3.hp.start_speech_token * torch.ones_like(text_tokens[:, :1])
        embeds, _ = t3.prepare_input_embeds(
            t3_cond=t3_cond,
            text_tokens=text_tokens,
            speech_tokens=speech_start_token,
            cfg_weight=0.0,
        )
        _t_embed = time.monotonic() - _t0

        _t0 = time.monotonic()
        llm_outputs = t3.tfmr(inputs_embeds=embeds, use_cache=True)
        hidden_states = llm_outputs[0]
        past_key_values = llm_outputs.past_key_values
        _t_prefill = time.monotonic() - _t0

        _log.info(
            f"  prefill: embed_prep={_t_embed*1000:.0f}ms  "
            f"tfmr_prefill={_t_prefill*1000:.0f}ms  "
            f"seq_len={embeds.size(1)}"
        )

        # First token from prefill
        speech_hidden = hidden_states[:, -1:]
        speech_logits = t3.speech_head(speech_hidden)
        processed_logits = logits_processors(speech_start_token, speech_logits[:, -1, :])
        probs = F.softmax(processed_logits, dim=-1)
        next_speech_token = torch.multinomial(probs, num_samples=1)
        _ar_start = time.monotonic()

        all_generated: list[int] = []
        pending: list[int] = []
        chunk_index = 0
        current_speech_token = next_speech_token
        overlap = config.overlap_tokens

        # Check first token for stop
        token_val = next_speech_token.item()
        if token_val == t3.hp.stop_speech_token:
            # Immediate stop — yield empty final chunk with silence
            final_tokens = [_SILENCE_TOKEN] * 3
            yield TokenChunk(
                tokens=torch.tensor([final_tokens], dtype=torch.long, device=device),
                is_final=True,
                chunk_index=0,
            )
            return

        pending.append(token_val)
        all_generated.append(token_val)

        # --- Autoregressive loop ---
        for _ in range(config.max_gen_len):
            if self._cancelled:
                break

            current_speech_embed = t3.speech_emb(current_speech_token)
            llm_outputs = t3.tfmr(
                inputs_embeds=current_speech_embed,
                past_key_values=past_key_values,
                use_cache=True,
            )
            hidden_states = llm_outputs[0]
            past_key_values = llm_outputs.past_key_values
            speech_logits = t3.speech_head(hidden_states)

            input_ids = torch.tensor([all_generated], dtype=torch.long, device=device)
            processed_logits = logits_processors(input_ids, speech_logits[:, -1, :])

            if torch.all(processed_logits == -float("inf")):
                break

            probs = F.softmax(processed_logits, dim=-1)
            next_speech_token = torch.multinomial(probs, num_samples=1)
            token_val = next_speech_token.item()
            current_speech_token = next_speech_token

            # Stop token → finalize
            if token_val == t3.hp.stop_speech_token:
                break

            pending.append(token_val)
            all_generated.append(token_val)

            # Check if we should yield a chunk
            if self._should_yield(pending, chunk_index, config):
                filtered = self._filter_oov(pending)
                if filtered:
                    _ar_elapsed = time.monotonic() - _ar_start
                    _log.info(
                        f"  yield chunk {chunk_index}: "
                        f"{len(filtered)} tokens, ar={_ar_elapsed*1000:.0f}ms"
                    )
                    yield TokenChunk(
                        tokens=torch.tensor([filtered], dtype=torch.long, device=device),
                        is_final=False,
                        chunk_index=chunk_index,
                    )
                    chunk_index += 1
                    _ar_start = time.monotonic()

                # Carry overlap tokens into next pending
                if overlap > 0 and len(filtered) >= overlap:
                    pending = filtered[-overlap:]
                else:
                    pending = []

        # --- Final chunk ---
        filtered = self._filter_oov(pending)
        # Append silence suffix
        filtered.extend([_SILENCE_TOKEN] * 3)
        if filtered:
            yield TokenChunk(
                tokens=torch.tensor([filtered], dtype=torch.long, device=device),
                is_final=True,
                chunk_index=chunk_index,
            )
