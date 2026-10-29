# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import json
from pathlib import Path

import pytest

from vllm_omni.model_executor.models.ming_flash_omni import omni_voice

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

ALIASES = json.loads((Path(omni_voice.__file__).parent / "data" / "ip_voice_aliases.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("alias, name", ALIASES.items())
def test_english_alias_uses_canonical_caption(alias, name):
    assert name in omni_voice.ip_voice_names()
    assert omni_voice.ip_voice_name(f" {alias.upper()} ") == name
    english = omni_voice.resolve_omni_voice("你好", {"voice_name": alias})
    chinese = omni_voice.resolve_omni_voice("你好", {"voice_name": name})
    assert english == chinese
    assert english["ip_name"] == name
    assert name in english["instruction"]


@pytest.mark.parametrize("name", ["DB30", "unknown_voice", "不存在的音色", "ling_not_a_voice"])
def test_other_voices_keep_reference_handling(name):
    assert omni_voice.ip_voice_name(name) is None
    result = omni_voice.resolve_omni_voice("你好", {"voice_name": name})
    assert result["voice_mode"] == "reference"
    assert result["voice_name"] == name
    assert result["instruction"] is None


@pytest.mark.parametrize("name", omni_voice.ip_voice_names())
def test_existing_canonical_names_remain_supported(name):
    assert omni_voice.ip_voice_name(name) == name
