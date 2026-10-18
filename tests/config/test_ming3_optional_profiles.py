# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Optional Ming3 profiles preserve topology across both config builders."""

from pathlib import Path

import pytest

from vllm_omni.config.omni_config import VllmOmniConfig
from vllm_omni.config.pipeline_registry import resolve_pipeline_config
from vllm_omni.config.stage_config import load_deploy_config, merge_pipeline_deploy, resolve_deploy_yaml

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_DEPLOY = Path(__file__).parents[2] / "vllm_omni" / "deploy"
_VARIANTS = ("", "_asr", "_thinker_only")
_PROFILES = [f"ming3_audio_flash{variant}_128k.yaml" for variant in _VARIANTS] + [
    f"ming3_audio_tiny{variant}.yaml" for variant in _VARIANTS
]


@pytest.mark.parametrize("profile", _PROFILES)
def test_optional_ming_profile_resolves_topology_and_context(profile):
    deploy = load_deploy_config(_DEPLOY / profile)
    pipeline = resolve_pipeline_config(deploy.pipeline)
    assert pipeline is not None
    legacy = merge_pipeline_deploy(pipeline, deploy)
    structured = VllmOmniConfig.from_pipeline_config(pipeline, user_deploy_config=deploy)
    thinker = structured.stage_by_id(0)
    assert thinker.scheduler_config.max_model_len == legacy[0].yaml_engine_args["max_model_len"] == 131072
    assert (
        thinker.scheduler_config.max_num_batched_tokens == legacy[0].yaml_engine_args["max_num_batched_tokens"] == 32768
    )
    assert thinker.model_config.default_sampling_params == legacy[0].yaml_extras["default_sampling_params"]
    expected_tp = 1 if "tiny" in profile else 4
    assert thinker.parallel_config.tensor_parallel_size == expected_tp
    assert thinker.runtime_config.devices == ("0" if expected_tp == 1 else "0,1,2,3")
    assert len(structured.stage_configs) == len(legacy) == (1 if "thinker_only" in profile else 2)
    if len(legacy) == 2:
        talker = structured.stage_by_id(1)
        assert talker.parallel_config.tensor_parallel_size == 1
        assert talker.runtime_config.devices == ("0" if expected_tp == 1 else "3")
        assert talker.scheduler_config.max_num_batched_tokens == 8192

    lowered = VllmOmniConfig.from_pipeline_config(
        pipeline,
        user_deploy_config=deploy,
        cli_overrides={"stage_0_max_model_len": 4096, "stage_0_max_num_batched_tokens": 512},
    )
    assert lowered.stage_by_id(0).scheduler_config.max_model_len == 4096
    assert lowered.stage_by_id(0).scheduler_config.max_num_batched_tokens == 512
    if len(legacy) == 2:
        assert lowered.stage_by_id(1).scheduler_config.max_num_batched_tokens == 8192


@pytest.mark.parametrize("variant", _VARIANTS)
def test_flash_128k_overlay_preserves_base_execution_settings(variant):
    base = resolve_deploy_yaml(_DEPLOY / f"ming3_audio_flash{variant}.yaml")
    extended = resolve_deploy_yaml(_DEPLOY / f"ming3_audio_flash{variant}_128k.yaml")
    base["stages"][0].update(max_model_len=131072, max_num_batched_tokens=32768)
    base["stages"][0]["default_sampling_params"]["max_tokens"] = 32768 if variant == "_asr" else 65536
    assert extended == base


def test_ming3_history_cache_profile_reaches_legacy_and_structured_builders():
    deploy = load_deploy_config(_DEPLOY / "ming3_audio_flash_history_cache.yaml")
    pipeline = resolve_pipeline_config(deploy.pipeline)
    assert pipeline is not None
    legacy = merge_pipeline_deploy(pipeline, deploy)
    structured = VllmOmniConfig.from_pipeline_config(pipeline, user_deploy_config=deploy)
    thinker, talker = structured.stage_by_id(0), structured.stage_by_id(1)
    assert thinker.cache_config.enable_prefix_caching is True
    assert thinker.cache_config.mamba_cache_mode == "align"
    assert thinker.scheduler_config.enable_chunked_prefill is True
    assert thinker.model_config.limit_mm_per_prompt == {"audio": 16}
    assert talker.cache_config.enable_prefix_caching is False
    assert legacy[0].yaml_engine_args["enable_prefix_caching"] is True
    assert legacy[0].yaml_engine_args["mamba_cache_mode"] == "align"
    assert legacy[0].yaml_engine_args["limit_mm_per_prompt"] == {"audio": 16}
    assert legacy[1].yaml_engine_args["enable_prefix_caching"] is False
