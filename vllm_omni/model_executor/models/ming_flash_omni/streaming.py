# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Request-local producer/consumer transport for native Ming audio decoding."""

from queue import Empty, Full, Queue
from threading import Event, Thread
from time import monotonic

import torch


class _CancelledError(Exception):
    pass


class MingStreamingJob:
    """Generate on a separate CUDA stream and wake the decoder for ready data.

    The producer marks the final latent, so even a one-chunk utterance can
    flush the VAE without submitting an empty latent or duplicating a frame.
    By default, wait for the polling interval and then drain ready latents.
    With low_latency_streaming, decode the first audio as soon as latents are
    ready, then coalesce within the polling budget and flush final latents
    immediately. This changes VAE chunk boundaries and is an explicit opt-in.
    The bounded queue applies backpressure to slow consumers.
    """

    def __init__(self, produce, vae, device, poll_seconds=0.1, *, low_latency_streaming=False):
        self.vae = vae
        self.device = device
        self.poll_seconds = float(poll_seconds)
        self.low_latency_streaming = low_latency_streaming
        self.queue = Queue(maxsize=32)
        self.cancelled = Event()
        self.done = Event()
        self.error = None
        self.past_key_values = None
        self.stream_state = (None, None, None)
        self.finished = False
        self._has_audio_output = False
        # The caller may have cloned input tensors on its current stream.
        ready = torch.cuda.Event()
        ready.record(torch.cuda.current_stream(device))

        def run():
            try:
                with (
                    torch.inference_mode(),
                    torch.accelerator.device_index(device.index),
                    torch.cuda.stream(torch.cuda.Stream(device)),
                ):
                    torch.cuda.current_stream().wait_event(ready)
                    produce(self._publish)
            except _CancelledError:
                pass
            except BaseException as exc:
                self.error = exc
            finally:
                self.done.set()

        self.thread = Thread(target=run, name="ming-latent-producer", daemon=True)
        self.thread.start()

    def _publish(self, latent, last):
        # As in ming_flash.llm_job, publish only completed latent tensors.
        torch.cuda.current_stream().synchronize()
        while not self.cancelled.is_set():
            try:
                self.queue.put((latent, last), timeout=0.01)
                return
            except Full:
                continue
        raise _CancelledError()

    def next_audio(self):
        # Initial waiting and later coalescing share one budget per call.
        deadline = monotonic() + self.poll_seconds
        if not self.low_latency_streaming:
            # Preserve legacy Ming collection: wait once, then drain without
            # waiting for subsequent arrivals. VAE numerics can depend on the
            # resulting chunk boundaries, even when the latents are identical.
            self.cancelled.wait(self.poll_seconds)
        parts = []
        last = False
        while not last:
            if self.cancelled.is_set():
                raise _CancelledError()
            if self.error is not None:
                raise self.error
            try:
                latent, last = self.queue.get_nowait()
            except Empty:
                if self.done.is_set():
                    # Completion follows the last put. Recheck after observing
                    # done: the producer may have finished after our empty read.
                    try:
                        latent, last = self.queue.get_nowait()
                    except Empty as exc:
                        if self.cancelled.is_set():
                            raise _CancelledError()
                        if self.error is not None:
                            raise self.error
                        if not parts:
                            raise RuntimeError("Ming latent producer ended without a final latent") from exc
                        break
                else:
                    if not self.low_latency_streaming or (parts and not self._has_audio_output):
                        break
                    remaining = deadline - monotonic()
                    if remaining <= 0:
                        break
                    try:
                        # A put wakes this immediately. Once audio has started,
                        # collect subsequent latents only within this deadline.
                        latent, last = self.queue.get(timeout=remaining)
                    except Empty:
                        continue
            parts.append(latent)
        if self.cancelled.is_set():
            raise _CancelledError()
        if self.error is not None:
            raise self.error
        if not parts:
            return torch.empty(1, 1, 0, device=self.device), False
        audio, self.stream_state, self.past_key_values = self.vae.decode(
            torch.cat(parts, dim=1),
            past_key_values=self.past_key_values,
            stream_state=self.stream_state,
            use_cache=True,
            last_chunk=last,
        )
        self.finished = last
        if audio.numel():
            self._has_audio_output = True
        return audio, last

    def close(self):
        self.cancelled.set()
        # Wait for the current AR step to complete before another request
        # can reuse the CFM graph pool or the process-wide seeded RNG.
        self.thread.join()
        while not self.queue.empty():
            self.queue.get_nowait()
        self.past_key_values = None
        self.stream_state = (None, None, None)
