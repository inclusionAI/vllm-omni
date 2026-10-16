# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Request-local producer/consumer transport for native Ming audio decoding."""

from queue import Empty, Full, Queue
from threading import Event, Thread

import torch


class _CancelledError(Exception):
    pass


class MingStreamingJob:
    """Generate on a separate CUDA stream and decode new tokens every 100 ms.

    The producer marks the final latent, so even a one-chunk utterance can
    flush the VAE without submitting an empty latent or duplicating a frame.
    The bounded queue applies backpressure to slow consumers.
    """

    def __init__(self, produce, vae, device, poll_seconds=0.1):
        self.vae = vae
        self.device = device
        self.poll_seconds = poll_seconds
        self.queue = Queue(maxsize=32)
        self.cancelled = Event()
        self.done = Event()
        self.error = None
        self.past_key_values = None
        self.stream_state = (None, None, None)
        self.finished = False
        # The caller may have cloned input tensors on its current stream.
        ready = torch.cuda.Event()
        ready.record(torch.cuda.current_stream(device))

        def run():
            try:
                with torch.inference_mode(), torch.cuda.device(device), torch.cuda.stream(torch.cuda.Stream(device)):
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
        self.cancelled.wait(self.poll_seconds)
        if self.error is not None:
            raise self.error
        parts = []
        last = False
        while True:
            try:
                latent, last = self.queue.get_nowait()
            except Empty:
                break
            parts.append(latent)
            if last:
                break
        if not parts:
            if self.done.is_set():
                if self.error is not None:
                    raise self.error
                raise RuntimeError("Ming latent producer ended without a final latent")
            return torch.empty(1, 1, 0, device=self.device), False
        audio, self.stream_state, self.past_key_values = self.vae.decode(
            torch.cat(parts, dim=1),
            past_key_values=self.past_key_values,
            stream_state=self.stream_state,
            use_cache=True,
            last_chunk=last,
        )
        self.finished = last
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
