# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Persisted model voices and Realtime/Chat/Talker voice selection, CPU only."""

import base64
import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest
import soundfile as sf

from vllm_omni.entrypoints.openai.ming3_realtime import Ming3RealtimeConnection
from vllm_omni.entrypoints.openai.ming3_voices import register_model_voice_presets, resolve_stored_voice, voice_id
from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat
from vllm_omni.entrypoints.openai.serving_speech import OmniOpenAIServingSpeech
from vllm_omni.model_executor.stage_input_processors.ming_talker import build_ming_talker_additional_info

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.asyncio]


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("SPEAKER_SAMPLES_DIR", str(tmp_path / "stored"))
    server = object.__new__(OmniOpenAIServingSpeech)
    server._init_speaker_storage()
    server._adapter = None
    server._tts_model_type = "ming_flash_omni_tts"
    return server


@pytest.fixture
def model(tmp_path):
    root = tmp_path / "model"
    presets = root / "voice_presets"
    (presets / "voice_audio").mkdir(parents=True)
    for name in ("a", "b"):
        sf.write(presets / "voice_audio" / f"{name}.wav", np.full(20000, 0.125, dtype=np.float32), 16000)
    (presets / "voice_prompts.json").write_text(
        json.dumps(
            {
                "DB30": {"text": "感谢你的认可。", "audio": "voice_audio/a.wav"},
                "lingguang": {"text": "你好", "audio": "voice_audio/b.wav"},
            },
            ensure_ascii=False,
        )
    )
    return root


def chat_for(store):
    chat = object.__new__(OmniOpenAIServingChat)
    chat.engine_client = SimpleNamespace(
        stage_configs=[
            {"engine_args": {"model_arch": "BailingMoeV3AudioForConditionalGeneration", "model_stage": "thinker"}},
            {"engine_args": {"model_arch": "BailingTalker2", "model_stage": "ming3_tts"}},
        ]
    )
    chat._ming3_speech_service = store
    return chat


async def test_model_import_persists_audio_text_and_reuses_on_restart(store, model):
    before = {str(p.relative_to(model)): p.read_bytes() for p in model.rglob("*") if p.is_file()}
    await register_model_voice_presets(store, str(model))
    assert set(store.uploaded_speakers) == {"db30", "lingguang"}
    assert store.uploaded_speakers["db30"]["ref_text"] == "感谢你的认可。"
    files = sorted(store.uploaded_speakers_dir.glob("*.safetensors"))
    assert len(files) == 2
    timestamps = {p: p.stat().st_mtime_ns for p in files}
    restored = object.__new__(OmniOpenAIServingSpeech)
    restored._init_speaker_storage()
    restored._adapter = None
    await register_model_voice_presets(restored, str(model))
    assert {p: p.stat().st_mtime_ns for p in files} == timestamps
    audio, sr = restored._load_uploaded_audio("DB30")
    assert sr == 16000
    np.testing.assert_allclose(audio, 0.125)
    assert {str(p.relative_to(model)): p.read_bytes() for p in model.rglob("*") if p.is_file()} == before


async def test_manifest_change_refreshes_existing_preset(store, model):
    await register_model_voice_presets(store, str(model))
    previous = store.uploaded_speakers["db30"]["file_path"]
    manifest = model / "voice_presets/voice_prompts.json"
    data = json.loads(manifest.read_text())
    data["DB30"]["text"] = "新文本"
    manifest.write_text(json.dumps(data))
    await register_model_voice_presets(store, str(model))
    assert store.uploaded_speakers["db30"]["ref_text"] == "新文本"
    assert not Path(previous).exists()
    assert len(list(store.uploaded_speakers_dir.glob("*.safetensors"))) == 2


async def test_runtime_upload_takes_precedence(store, model):
    store.uploaded_speakers["db30"] = {"name": "DB30", "ref_text": "user"}
    await register_model_voice_presets(store, str(model))
    assert store.uploaded_speakers["db30"]["ref_text"] == "user"
    assert "lingguang" in store.uploaded_speakers


@pytest.mark.parametrize(
    "entry",
    [
        None,
        {},
        {"text": 1, "audio": "voice_audio/a.wav"},
        {"text": "x", "audio": "../outside.wav"},
        {"text": "x", "audio": "/tmp/outside.wav"},
        {"text": "x", "audio": "missing.wav"},
    ],
)
async def test_bad_manifest_fails_before_registration(store, model, entry):
    (model / "voice_presets/voice_prompts.json").write_text(json.dumps({"bad": entry}))
    with pytest.raises((ValueError, FileNotFoundError)):
        await register_model_voice_presets(store, str(model))
    assert not store.uploaded_speakers


async def test_duplicate_casefolded_names_rejected(store, model):
    entry = {"text": "x", "audio": "voice_audio/a.wav"}
    (model / "voice_presets/voice_prompts.json").write_text(json.dumps({"A": entry, "a": entry}))
    with pytest.raises(ValueError, match="Duplicate"):
        await register_model_voice_presets(store, str(model))
    assert not store.uploaded_speakers


async def test_missing_manifest_is_optional(store, tmp_path):
    await register_model_voice_presets(store, str(tmp_path))
    assert not store.uploaded_speakers


async def test_named_chat_voice_reaches_talker_with_audio_and_text(store, model):
    await register_model_voice_presets(store, str(model))
    chat = chat_for(store)
    prompt = {}
    request = SimpleNamespace(audio={"voice": {"id": "DB30"}})
    await chat._attach_ming3_reference_audio(prompt, request)
    info = build_ming_talker_additional_info("你好", prompt)
    assert info["voice_name"] == "db30"
    assert info["prompt_text"] == "感谢你的认可。"
    assert info["prompt_audio_mode"] == "continuation"
    assert info["speaker_conditioning"] == "prompt"
    assert info["reference_audio"][0]["sample_rate"] == 16000
    assert info["reference_audio"][0]["samples"][0] == pytest.approx(0.125)
    assert info["instruction"] is None


async def test_ip_mode_ignores_stored_selection(store):
    chat = chat_for(store)
    prompt = {"additional_information": {"voice_mode": "ip", "ip_name": "灵小柔"}}
    await chat._attach_ming3_reference_audio(prompt, SimpleNamespace(audio={"voice": "not-registered"}))
    info = build_ming_talker_additional_info("你好", prompt)
    assert json.loads(info["instruction"])["audio_sequence"][0]["IP"] == "灵小柔"
    assert info["reference_audio"] is None
    assert info["prompt_audio_mode"] == "none"


async def test_unknown_voice_and_conflicting_reference_rejected(store):
    with pytest.raises(ValueError, match="Unknown voice"):
        await resolve_stored_voice(store, {"id": "unknown"})
    with pytest.raises(ValueError, match="either a stored voice"):
        await chat_for(store)._attach_ming3_reference_audio(
            {"additional_information": {"voice_name": "x", "ref_audio": "AA=="}}
        )


@pytest.fixture
def realtime(store):
    socket = SimpleNamespace(query_params={"model": "ming"}, send_text=AsyncMock())
    chat = Mock()
    chat._is_model_supported.return_value = True
    chat._ming3_speech_service = store
    conn = Ming3RealtimeConnection(socket, chat)
    conn._is_connected = True
    return conn


def sent(conn):
    return [json.loads(call.args[0]) for call in conn.websocket.send_text.call_args_list]


async def update(conn, voice="DB30", **extra):
    await conn.handle_event(
        {
            "type": "session.update",
            "session": {
                "audio": {"output": {"voice": voice}},
                **extra,
            },
        }
    )


async def test_realtime_official_session_voice_and_unknown_error(realtime, store, model):
    await register_model_voice_presets(store, str(model))
    await update(realtime, {"id": "DB30"})
    assert sent(realtime)[-1]["type"] == "session.updated"
    assert realtime._session["audio"]["output"]["voice"] == "db30"
    await update(realtime, "unknown")
    assert sent(realtime)[-1]["type"] == "error"
    assert realtime._session["audio"]["output"]["voice"] == "db30"


async def test_realtime_voice_locked_after_audio(realtime, store, model):
    await register_model_voice_presets(store, str(model))
    await update(realtime)
    realtime._audio_started = True
    realtime._spoken_voice = ("reference", "db30")
    await update(realtime, {"id": "DB30"})
    assert sent(realtime)[-1]["type"] == "session.updated"
    await update(realtime, "lingguang")
    assert sent(realtime)[-1]["type"] == "error"


async def test_realtime_ip_mode_needs_no_reference(realtime):
    await update(realtime, "not-stored", additional_information={"voice_mode": "ip", "ip_name": "灵小柔"})
    assert sent(realtime)[-1]["type"] == "session.updated"


async def test_realtime_audio_buffer_commit_is_a_user_turn(realtime):
    pcm = np.zeros(24000, dtype="<i2").tobytes()
    await realtime.handle_event({"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm).decode()})
    await realtime.handle_event({"type": "input_audio_buffer.commit"})
    assert sent(realtime)[-1]["type"] == "input_audio_buffer.committed"
    assert not realtime._input_audio
    assert realtime.generation_task is None  # Official manual-turn commit waits for response.create.
    audio = realtime._messages[0]["content"][0]["input_audio"]
    waveform, sr = sf.read(io.BytesIO(base64.b64decode(audio["data"])))
    assert sr == 24000 and len(waveform) == 24000


async def test_realtime_response_uses_selected_voice_and_emits_pcm24(realtime, store, model):
    await register_model_voice_presets(store, str(model))
    await update(realtime)
    await realtime.handle_event(
        {
            "type": "conversation.item.create",
            "item": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "你好"}]},
        }
    )
    buffer = io.BytesIO()
    sf.write(buffer, np.full(4410, 0.125), 44100, format="WAV")
    audio = base64.b64encode(buffer.getvalue()).decode()
    captured = []
    closed = []

    async def events():
        try:
            for modality, content in [("text", "你好"), ("audio", audio)]:
                yield (
                    "data: " + json.dumps({"modality": modality, "choices": [{"delta": {"content": content}}]}) + "\n\n"
                )
            yield "data: [DONE]\n\n"
        finally:
            closed.append(True)

    async def create(request):
        captured.append(request)
        # Exercise actual voice lookup and thinker-to-talker handoff too.
        prompt = {"additional_information": dict(request.additional_information)}
        await chat_for(store)._attach_ming3_reference_audio(prompt, request)
        assert build_ming_talker_additional_info("你好", prompt)["prompt_text"] == "感谢你的认可。"
        return events()

    realtime.serving.create_chat_completion = create
    await realtime.handle_event({"type": "response.create"})
    await realtime.generation_task
    assert captured[0].additional_information["voice_name"] == "db30"
    events_sent = sent(realtime)
    delta = next(e for e in events_sent if e["type"] == "response.output_audio.delta")
    assert "audio" not in delta  # Official event uses delta.
    assert (
        sum(len(base64.b64decode(e["delta"])) for e in events_sent if e["type"] == "response.output_audio.delta")
        == 2400 * 2
    )
    assert events_sent[-1]["response"]["status"] == "completed"
    assert realtime._audio_started
    assert closed


async def test_voice_id_shapes():
    assert voice_id("DB30") == voice_id({"id": "DB30"}) == "db30"
    for value in (None, {}, {"id": ""}, 1):
        with pytest.raises(ValueError):
            voice_id(value)


async def test_registered_voice_works_with_speech_api_adapter(store, model):
    from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest
    from vllm_omni.entrypoints.openai.tts_adapters.base import SpeechServingContext
    from vllm_omni.entrypoints.openai.tts_adapters.ming3_audio_tts import Ming3AudioTTSAdapter

    await register_model_voice_presets(store, str(model))
    store._max_instructions_length = 500
    adapter = Ming3AudioTTSAdapter(SpeechServingContext(server=store))
    request = OpenAICreateSpeechRequest(input="你好", voice={"id": "DB30"})
    assert adapter.validate(request) is None
    assert request.voice is None  # Stored voice must not turn into an IP caption.
    assert request.ref_audio.startswith("data:audio/wav;base64,")
    assert request.ref_text == "感谢你的认可。"


async def test_realtime_websocket_route_and_voice_listing(store, model):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from vllm_omni.entrypoints.openai import api_server

    await register_model_voice_presets(store, str(model))
    chat = chat_for(store)
    chat._is_model_supported = lambda _: True
    app = FastAPI()
    app.include_router(api_server.router)
    app.state.openai_serving_chat = chat
    app.state.openai_serving_speech = store
    with TestClient(app) as client:
        result = client.get("/v1/audio/voices")
        assert result.status_code == 200
        assert {v["id"] for v in result.json()["uploaded_voices"]} == {"db30", "lingguang"}
        with client.websocket_connect("/v1/realtime?model=ming") as socket:
            assert socket.receive_json()["type"] == "session.created"
            socket.send_json(
                {
                    "type": "session.update",
                    "session": {
                        "audio": {"output": {"voice": {"id": "DB30"}}},
                    },
                }
            )
            updated = socket.receive_json()
            assert updated["type"] == "session.updated"
            assert updated["session"]["audio"]["output"]["voice"] == "db30"


async def test_realtime_response_override_cannot_bypass_voice_lock(realtime, store, model):
    await register_model_voice_presets(store, str(model))
    await update(realtime)
    realtime._messages = [{"role": "user", "content": "hello"}]
    realtime._audio_started = True
    realtime._spoken_voice = ("reference", "db30")
    await realtime.handle_event(
        {
            "type": "response.create",
            "response": {
                "audio": {"output": {"voice": {"id": "lingguang"}}},
            },
        }
    )
    assert sent(realtime)[-1]["type"] == "error"
    assert realtime.generation_task is None


async def test_realtime_generation_failure_has_terminal_event(realtime, store, model):
    await register_model_voice_presets(store, str(model))
    await update(realtime)
    realtime._messages = [{"role": "user", "content": "hello"}]
    realtime.serving.create_chat_completion = AsyncMock(side_effect=RuntimeError("generation failed"))
    await realtime.handle_event({"type": "response.create"})
    await realtime.generation_task
    assert any(e["type"] == "error" for e in sent(realtime))
    assert sent(realtime)[-1]["response"]["status"] == "failed"


async def test_hf_snapshot_audio_symlinks_are_supported(store, model, tmp_path):
    audio = model / "voice_presets/voice_audio/a.wav"
    blob = tmp_path / "cached-blob"
    audio.rename(blob)
    audio.symlink_to(blob)
    await register_model_voice_presets(store, str(model))
    assert "db30" in store.uploaded_speakers


async def test_hf_preset_download_uses_model_revision(store, model, monkeypatch):
    download = Mock(return_value=str(model))
    monkeypatch.setattr("huggingface_hub.snapshot_download", download)
    await register_model_voice_presets(store, "org/model", revision="commit-sha", cache_dir="/cache")
    download.assert_called_once_with(
        "org/model", revision="commit-sha", cache_dir="/cache", allow_patterns=["voice_presets/**"]
    )
    assert len(store.uploaded_speakers) == 2


async def test_realtime_pcm_format_defaults_and_invalid_format(realtime, store, model):
    await register_model_voice_presets(store, str(model))
    await realtime.handle_event(
        {
            "type": "session.update",
            "session": {
                "audio": {"output": {"voice": "DB30", "format": {"type": "audio/pcm"}}},
            },
        }
    )
    assert sent(realtime)[-1]["type"] == "session.updated"
    assert realtime._session["audio"]["output"]["format"]["rate"] == 24000
    await realtime.handle_event(
        {
            "type": "session.update",
            "session": {
                "audio": {"output": {"format": {"type": "audio/pcm", "rate": 16000}}},
            },
        }
    )
    assert sent(realtime)[-1]["type"] == "error"
    assert realtime._session["audio"]["output"]["format"]["rate"] == 24000


async def test_realtime_disconnect_closes_active_chat_generator(realtime, store, model):
    import asyncio

    await register_model_voice_presets(store, str(model))
    await update(realtime)
    realtime._messages = [{"role": "user", "content": "hello"}]
    started = asyncio.Event()
    closed = asyncio.Event()

    async def chunks():
        try:
            started.set()
            await asyncio.Event().wait()
            yield "data: [DONE]\n\n"
        finally:
            closed.set()

    realtime.serving.create_chat_completion = AsyncMock(return_value=chunks())
    await realtime.handle_event({"type": "response.create"})
    await asyncio.wait_for(started.wait(), timeout=2)
    realtime._is_connected = False
    await realtime.cleanup()
    assert realtime.generation_task.done()
    assert closed.is_set()
