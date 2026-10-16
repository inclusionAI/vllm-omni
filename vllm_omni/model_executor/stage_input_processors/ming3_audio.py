# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Stage input processors for Ming-3.0-audio-flash."""

from __future__ import annotations

from typing import Any

import regex as re
import torch
from vllm.inputs import TextPrompt
from vllm.tokenizers import cached_tokenizer_from_config

from vllm_omni.data_entry_keys import CodesStruct, MetaStruct, OmniPayloadStruct
from vllm_omni.engine import OmniEngineCoreRequest
from vllm_omni.engine.serialization import deserialize_additional_information
from vllm_omni.inputs.data import OmniTokensPrompt

from .ming_talker import (
    build_ming_talker_additional_info,
    build_ming_talker_inputs,
    strip_language_tag,
)


def thinker2talker_token_only(
    source_outputs: list[Any],
    prompt: OmniTokensPrompt | TextPrompt | list[Any] | None = None,
    _requires_multimodal_data: bool = False,
) -> list[OmniTokensPrompt]:
    """Build Ming talker inputs after the thinker has completed its answer."""
    return build_ming_talker_inputs(source_outputs, prompt)


_SENTENCE_END = re.compile(r"[。！？!?]|(?<!\d)\.(?!\d)")
_PROTECTED_SENTENCE_TEXT = re.compile(
    r"https?://[^\s。！？]+|\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr)\.|\b(?:[A-Za-z]\.)+[A-Za-z]?",
    re.IGNORECASE,
)
_CLOSING_PUNCTUATION = frozenset("\"'”’）】")


def _ready_sentence_end(text: str, start: int) -> int | None:
    """Return the last stable sentence boundary after ``start``."""
    end = None
    protected = list(_PROTECTED_SENTENCE_TEXT.finditer(text))
    for match in _SENTENCE_END.finditer(text, start):
        if any(span.start() <= match.start() < span.end() for span in protected):
            continue
        # The next token may turn a trailing period into an abbreviation or URL.
        if match.group() == "." and match.end() == len(text):
            continue
        end = match.end()
        while end < len(text) and text[end] in _CLOSING_PUNCTUATION:
            end += 1
    return end


def thinker2talker_async_chunk(
    transfer_manager: Any,
    multimodal_output: dict[str, Any] | None,
    request: OmniEngineCoreRequest,
    is_finished: bool = False,
) -> OmniPayloadStruct | None:
    """Send stable thinker sentences to Ming's self-contained talker.

    The talker retokenizes each sentence; no thinker hidden-state transport is needed.
    Full token history is requested explicitly for the v2 data plane.
    """
    del multimodal_output
    request_id = request.external_req_id
    states = getattr(transfer_manager, "_ming3_chunk_state", None)
    if states is None:
        states = {}
        transfer_manager._ming3_chunk_state = states
    state = states.setdefault(request_id, "")

    token_ids = list(request.all_token_ids)[len(request.prompt_token_ids) :]
    model_config = getattr(transfer_manager, "model_config", None) or getattr(transfer_manager, "config", None)
    if model_config is None and hasattr(transfer_manager, "_get_model_config"):
        model_config = transfer_manager._get_model_config()
    tokenizer = getattr(transfer_manager, "_ming3_text_tokenizer", None)
    if tokenizer is None:
        tokenizer = cached_tokenizer_from_config(model_config)
        if tokenizer is None:
            raise RuntimeError("Ming3 async chunk requires the thinker tokenizer")
        transfer_manager._ming3_text_tokenizer = tokenizer
    text = tokenizer.decode(token_ids, skip_special_tokens=True)
    if not text.startswith(state):
        raise RuntimeError("Ming3 thinker text changed after an audio sentence was emitted")

    end = len(text) if is_finished else _ready_sentence_end(text, len(state))
    if end is None:
        return None
    chunk = text[len(state) : end]
    states[request_id] = text[:end]
    if is_finished:
        states.pop(request_id, None)
    # The transcribe-format thinker prefixes its stream with a `<lang>\t`
    # tag (metadata for text responses). Strip it from the first chunk only
    # after the raw prefix-consistency state above has been advanced, so it
    # is never read aloud while `text.startswith(state)` stays intact.
    chunk = strip_language_tag(chunk.lstrip()).strip()
    if not chunk:
        return None

    return OmniPayloadStruct(
        # The generation runner needs a nonempty prompt to execute this chunk.
        codes=CodesStruct(audio=torch.tensor([0], dtype=torch.long)),
        additional_information=build_ming_talker_additional_info(
            chunk, deserialize_additional_information(request.additional_information)
        ),
        meta=MetaStruct(replace_runtime_additional_information=True),
    )


# Opt into dispatch on every decode step, even without an inter-stage payload.
# The v2 data plane also keeps the full token history for each snapshot,
# since sentence segmentation decodes the complete generated prefix on every step.
setattr(thinker2talker_async_chunk, "_requires_full_token_history", True)


__all__ = ["thinker2talker_async_chunk", "thinker2talker_token_only"]
