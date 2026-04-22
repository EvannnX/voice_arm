from __future__ import annotations

import asyncio
import logging

from ..config import AUDIO_CHANNELS, OUTPUT_SAMPLE_RATE_HZ

logger = logging.getLogger(__name__)


class Speaker:
    """Async playback sink for 24 kHz mono s16le PCM coming from Gemini."""

    def __init__(
        self,
        sample_rate_hz: int = OUTPUT_SAMPLE_RATE_HZ,
        channels: int = AUDIO_CHANNELS,
        queue_max: int = 256,
    ) -> None:
        self._sample_rate = sample_rate_hz
        self._channels = channels
        self._queue: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=queue_max)
        self._stream = None
        self._worker: asyncio.Task[None] | None = None

    async def __aenter__(self) -> "Speaker":
        import sounddevice as sd

        self._stream = sd.RawOutputStream(
            samplerate=self._sample_rate,
            dtype="int16",
            channels=self._channels,
        )
        self._stream.start()
        self._worker = asyncio.create_task(self._drain(), name="speaker-drain")
        logger.info("speaker started: %d Hz, %d ch", self._sample_rate, self._channels)
        return self

    async def __aexit__(self, *_exc) -> None:
        await self._queue.put(None)
        if self._worker is not None:
            await self._worker
            self._worker = None
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    async def play(self, pcm: bytes) -> None:
        await self._queue.put(pcm)

    def clear(self) -> None:
        """Drop any buffered audio — used when the user interrupts the arm's reply."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def _drain(self) -> None:
        assert self._stream is not None
        loop = asyncio.get_running_loop()
        while True:
            chunk = await self._queue.get()
            if chunk is None:
                return
            await loop.run_in_executor(None, self._stream.write, chunk)
