# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from pathlib import Path

import pytest

from vllm_omni.config.stage_config import load_deploy_config
from vllm_omni.model_executor.models.ming_flash_omni import text_processing
from vllm_omni.model_executor.models.ming_flash_omni.tts_request import resolve_generation_params
from vllm_omni.model_executor.stage_input_processors.ming_talker import (
    build_ming_talker_additional_info,
    build_ming_talker_inputs,
)
from vllm_omni.transformers_utils.configs.ming_talker import MingFlashOmniTalkerConfig

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _profile_config() -> MingFlashOmniTalkerConfig:
    root = Path(__file__).resolve().parents[4]
    deploy = load_deploy_config(root / "vllm_omni/deploy/ming3_audio_flash.yaml")
    overrides = deploy.stages[1].engine_extras["hf_overrides"]
    return MingFlashOmniTalkerConfig.from_dict({"normalize_text": False}, **overrides)


def test_selected_profile_defaults_reach_talker_without_request_overrides():
    config = _profile_config()
    info = build_ming_talker_additional_info("金额12.5元。", None)
    params = resolve_generation_params(info, config)
    assert params.normalize_text is True
    assert params.tn_mode == "upstream"
    assert "tn_mode" not in info
    assert resolve_generation_params(info, MingFlashOmniTalkerConfig()).tn_mode == "legacy"


def test_explicit_legacy_and_disabled_normalization_override_profile():
    info = build_ming_talker_additional_info(
        "原始文字。", {"normalize_text": False, "tn_mode": "legacy", "stream_decode": False, "seed": 123}
    )
    params = resolve_generation_params(info, _profile_config())
    assert params.normalize_text is False
    assert params.tn_mode == "legacy"
    assert params.stream_decode is False
    assert params.seed == 123


def test_invalid_normalization_mode_is_rejected():
    with pytest.raises(ValueError, match="tn_mode"):
        resolve_generation_params({"tn_mode": "unsupported"}, _profile_config())
    with pytest.raises(ValueError, match="tn_mode"):
        MingFlashOmniTalkerConfig(tn_mode="unsupported")


def test_sync_language_only_text_does_not_leak_the_language_name():
    from types import SimpleNamespace

    outputs = [SimpleNamespace(outputs=[SimpleNamespace(text="  Chinese\t")])]
    prompts = build_ming_talker_inputs(outputs)
    assert prompts[0]["additional_information"]["text"] == ""


def test_upstream_resources_and_bilingual_number_normalization():
    pytest.importorskip("pynini")
    pytest.importorskip("inflect")
    tn = text_processing._get_upstream_tn()
    assert tn is not None, text_processing._UPSTREAM_TN_ERROR
    result = "".join(
        text_processing.segment_and_normalize("日期2026年10月16日，金额12.5元，进度50%。", tn_mode="upstream")
    )
    assert "百分之五十" in result
    assert "12.5" not in result
    assert "2026" not in result
    result = " ".join(text_processing.segment_and_normalize("I have 12 apples.", tn_mode="upstream"))
    assert "twelve" in result.lower()
