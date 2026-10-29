# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import asyncio
import io

import numpy as np
import pytest
import soundfile as sf
from fastapi import UploadFile
from safetensors.torch import load_file
from starlette.datastructures import Headers

from vllm_omni.entrypoints.openai.serving_speech import OmniOpenAIServingSpeech
from vllm_omni.entrypoints.openai.tts_adapters import all_tts_model_types, resolve_adapter

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _server(model_type):
    server = OmniOpenAIServingSpeech.__new__(OmniOpenAIServingSpeech)
    server._tts_model_type = model_type
    return server


@pytest.mark.parametrize("frames", [22511, 23663, 24000, 30 * 24000, 31 * 24000])
def test_lingv3_reference_duration_preserves_audio(frames):
    waveform = np.linspace(-0.5, 0.5, frames, dtype=np.float32)
    server = _server("lingv3_tts")
    samples, sr, key, duration = server._finalize_fetched_ref_audio(waveform, 24000)
    np.testing.assert_array_equal(samples, waveform)
    assert sr == 24000
    assert duration == frames / 24000
    assert key == server._make_ref_audio_artifact_cache_key(waveform, sr)


@pytest.mark.parametrize("model_type", sorted(all_tts_model_types() - {"lingv3_tts"}) + ["generic"])
def test_other_models_keep_reference_duration_limits(model_type):
    server = _server(model_type)
    for frames, error in [(22511, "too short"), (31 * 24000, "too long")]:
        with pytest.raises(ValueError, match=error):
            server._finalize_fetched_ref_audio(np.zeros(frames, dtype=np.float32), 24000)
    for seconds in [1, 30]:
        assert server._validate_ref_audio_duration(np.zeros(seconds * 24000), 24000) == seconds


@pytest.mark.parametrize("model_type", ["lingv3_tts", "qwen3_tts", "generic"])
@pytest.mark.parametrize("frames,sr,error", [(0, 24000, "empty"), (10, 0, "sample rate"), (10, -1, "sample rate")])
def test_invalid_references_are_rejected(model_type, frames, sr, error):
    with pytest.raises(ValueError, match=error):
        _server(model_type)._finalize_fetched_ref_audio(np.zeros(frames), sr)


@pytest.mark.parametrize("model_type", ["lingv3_tts", "qwen3_tts"])
@pytest.mark.parametrize("frames", [22511, 31 * 24000])
def test_voice_upload_uses_model_duration_policy(model_type, frames, tmp_path):
    server = _server(model_type)
    server._upload_lock = asyncio.Lock()
    server.uploaded_speakers = {}
    server.uploaded_speakers_dir = tmp_path
    server._max_uploaded_speakers = 10
    server._last_upload_ts = 0
    waveform = np.linspace(-0.5, 0.5, frames, dtype=np.float32)
    data = io.BytesIO()
    sf.write(data, waveform, 24000, format="WAV", subtype="FLOAT")
    data.seek(0)
    audio_file = UploadFile(data, filename="reference.wav", headers=Headers({"content-type": "audio/wav"}))
    upload = server.upload_voice(audio_file, consent="consent", name="reference")
    if resolve_adapter(model_type).ref_audio_duration_bounds == (None, None):
        result = asyncio.run(upload)
        assert result["name"] == "reference"
        saved = load_file(server.uploaded_speakers["reference"]["file_path"])["audio"]
        np.testing.assert_array_equal(saved.numpy(), waveform)
    else:
        with pytest.raises(ValueError, match="too short" if frames < 24000 else "too long"):
            asyncio.run(upload)
        assert not list(tmp_path.iterdir())
