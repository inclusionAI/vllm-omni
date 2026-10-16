# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace
from typing import cast

import pytest

from vllm_omni.engine import OmniEngineCoreRequest
from vllm_omni.engine.serialization import serialize_additional_information
from vllm_omni.model_executor.stage_input_processors.ming3_audio import thinker2talker_async_chunk

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _Tokenizer:
    def decode(self, token_ids: list[int], *, skip_special_tokens: bool) -> str:
        return "".join(chr(token_id) for token_id in token_ids)


def _manager() -> SimpleNamespace:
    return SimpleNamespace(_ming3_text_tokenizer=_Tokenizer(), _ming3_chunk_state={})


def _request(text: str, info=None) -> OmniEngineCoreRequest:
    return cast(
        OmniEngineCoreRequest,
        SimpleNamespace(
            external_req_id="request",
            prompt_token_ids=[99],
            all_token_ids=[99, *map(ord, text)],
            additional_information=info,
        ),
    )


def test_async_preserves_explicit_talker_controls():
    controls = {
        "tn_mode": "upstream",
        "normalize_text": True,
        "stream_decode": False,
        "seed": 123,
        "voice_mode": "ip",
        "ip_name": "灵小柔",
    }
    info = serialize_additional_information(controls)
    output = thinker2talker_async_chunk(_manager(), None, _request("Chinese\t你好。", info))

    assert output is not None
    assert output.additional_information["text"] == "你好。"
    for key, value in controls.items():
        assert output.additional_information[key] == value


def test_terminal_language_only_chunk_is_not_sent():
    manager = _manager()
    assert thinker2talker_async_chunk(manager, None, _request("  Chinese\t  "), is_finished=True) is None
    assert manager._ming3_chunk_state == {}


def test_raw_prefix_state_survives_language_stripping_and_terminal_tail():
    manager = _manager()
    first_text = "Chinese\t你好。"
    first = thinker2talker_async_chunk(manager, None, _request(first_text))
    assert first is not None
    assert first.additional_information["text"] == "你好。"
    assert manager._ming3_chunk_state["request"] == first_text
    assert thinker2talker_async_chunk(manager, None, _request(first_text)) is None

    final = thinker2talker_async_chunk(manager, None, _request(first_text + "剩余文字"), is_finished=True)
    assert final is not None
    assert final.additional_information["text"] == "剩余文字"
    assert manager._ming3_chunk_state == {}


def test_incremental_input_keeps_numeric_url_and_abbreviation_context():
    text = "Dr. Smith gave e.g. 日期2026.10.16，金额12.5元，IP192.168.1.1，https://example.com/path?a=1 这个例子!"
    manager = _manager()
    for length in range(1, len(text)):
        assert thinker2talker_async_chunk(manager, None, _request(text[:length])) is None
    output = thinker2talker_async_chunk(manager, None, _request(text))
    assert output is not None
    assert output.additional_information["text"] == text


def test_english_period_waits_for_lookahead_and_final_period_flushes():
    manager = _manager()
    assert thinker2talker_async_chunk(manager, None, _request("Hello.")) is None
    first = thinker2talker_async_chunk(manager, None, _request("Hello. Next"))
    assert first is not None
    assert first.additional_information["text"] == "Hello."
    final = thinker2talker_async_chunk(manager, None, _request("Hello. Next."), is_finished=True)
    assert final is not None
    assert final.additional_information["text"] == "Next."
    assert manager._ming3_chunk_state == {}
