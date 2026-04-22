from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator, Iterator

from ..config import (
    AUDIO_CHANNELS,
    AUDIO_SAMPLE_WIDTH_BYTES,
    FRAME_SIZE_BYTES,
    INPUT_SAMPLE_RATE_HZ,
)

logger = logging.getLogger(__name__)


def chunk_pcm(buffer: bytes, frame_size: int = FRAME_SIZE_BYTES) -> Iterator[bytes]:
    """Split a PCM byte buffer into fixed-size frames. Trailing bytes are dropped."""
    if frame_size <= 0:
        raise ValueError("frame_size must be positive")
    end = (len(buffer) // frame_size) * frame_size
    for i in range(0, end, frame_size):
        yield buffer[i : i + frame_size]


class MicStream:
    """Async iterator over 16 kHz mono s16le PCM frames from the default input device.

    Uses sounddevice's RawInputStream whose callback runs on a PortAudio thread;
    frames are marshalled back to the asyncio loop via call_soon_threadsafe.
    """

    def __init__(
        self,
        sample_rate_hz: int = INPUT_SAMPLE_RATE_HZ,
        channels: int = AUDIO_CHANNELS,
        frame_size_bytes: int = FRAME_SIZE_BYTES,
        queue_max: int = 64,
    ) -> None:
        self._sample_rate = sample_rate_hz
        self._channels = channels
        self._frame_size = frame_size_bytes
        self._queue: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=queue_max)
        self._stream = None
        self._loop: asyncio.AbstractEventLoop | None = None

    async def __aenter__(self) -> "MicStream":
        import sounddevice as sd  # imported lazily so tests don't need PortAudio

        self._loop = asyncio.get_running_loop()
        blocksize_frames = self._frame_size // (self._channels * AUDIO_SAMPLE_WIDTH_BYTES)

        def _callback(indata, _frames, _time, status) -> None:
            if status:
                logger.debug("mic status: %s", status)
            if self._loop is None:
                return
            data = bytes(indata)
            self._loop.call_soon_threadsafe(self._try_put, data)

        self._stream = sd.RawInputStream(
            samplerate=self._sample_rate,
            blocksize=blocksize_frames,
            dtype="int16",
            channels=self._channels,
            callback=_callback,
        )
        self._stream.start()
        logger.info(
            "mic started: %d Hz, %d ch, %d B frames",
            self._sample_rate,
            self._channels,
            self._frame_size,
        )
        return self

    async def __aexit__(self, *_exc) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        await self._queue.put(None)

    def _try_put(self, data: bytes) -> None:
        if self._queue.full():
            logger.warning("mic queue full — dropping frame")
            return
        self._queue.put_nowait(data)

    async def frames(self) -> AsyncIterator[bytes]:
        while True:
            frame = await self._queue.get()
            if frame is None:
                return
            yield frame
