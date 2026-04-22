from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator, Awaitable, Callable

from google import genai
from google.genai import types

from ..tools import ARM_TOOL_DECLARATIONS, ToolResult
from .system_prompt import SYSTEM_PROMPT

logger = logging.getLogger(__name__)

ToolHandler = Callable[[str, dict], Awaitable[ToolResult]]


class GeminiLiveSession:
    """Thin wrapper over google-genai's Live API session.

    Responsibilities:
    - open a live connection with our system prompt + tool declarations
    - stream mic PCM frames up
    - demultiplex server messages into: audio chunks, tool calls, interruptions
    - forward tool calls to the supplied handler and return their responses
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        on_tool_call: ToolHandler,
        on_audio: Callable[[bytes], Awaitable[None]],
        on_interrupt: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._on_tool_call = on_tool_call
        self._on_audio = on_audio
        self._on_interrupt = on_interrupt
        self._session = None

    def _config(self) -> types.LiveConnectConfig:
        return types.LiveConnectConfig(
            response_modalities=[types.Modality.AUDIO],
            system_instruction=types.Content(parts=[types.Part(text=SYSTEM_PROMPT)]),
            tools=[types.Tool(function_declarations=ARM_TOOL_DECLARATIONS)],
        )

    @asynccontextmanager
    async def connect(self) -> AsyncIterator["GeminiLiveSession"]:
        logger.info("connecting to Gemini Live model=%s", self._model)
        async with self._client.aio.live.connect(
            model=self._model,
            config=self._config(),
        ) as session:
            self._session = session
            try:
                yield self
            finally:
                self._session = None

    async def send_audio_frame(self, pcm_16khz_mono_s16le: bytes) -> None:
        if self._session is None:
            raise RuntimeError("session not connected")
        await self._session.send_realtime_input(
            audio=types.Blob(
                data=pcm_16khz_mono_s16le,
                mime_type="audio/pcm;rate=16000",
            )
        )

    async def receive_loop(self) -> None:
        if self._session is None:
            raise RuntimeError("session not connected")

        async for message in self._session.receive():
            await self._handle_message(message)

    async def _handle_message(self, message: types.LiveServerMessage) -> None:
        if message.data:
            await self._on_audio(message.data)

        server_content = getattr(message, "server_content", None)
        if server_content is not None and getattr(server_content, "interrupted", False):
            logger.debug("server signalled interruption")
            if self._on_interrupt is not None:
                await self._on_interrupt()

        tool_call = getattr(message, "tool_call", None)
        if tool_call is not None and tool_call.function_calls:
            await self._handle_tool_calls(tool_call.function_calls)

    async def _handle_tool_calls(
        self, function_calls: list[types.FunctionCall]
    ) -> None:
        responses = await asyncio.gather(
            *[self._run_tool(fc) for fc in function_calls]
        )
        assert self._session is not None
        await self._session.send_tool_response(function_responses=responses)

    async def _run_tool(self, fc: types.FunctionCall) -> types.FunctionResponse:
        args = dict(fc.args or {})
        logger.info("tool_call %s args=%s", fc.name, json.dumps(args, default=str))
        result = await self._on_tool_call(fc.name, args)
        return types.FunctionResponse(
            id=fc.id,
            name=fc.name,
            response=result.as_response(),
        )
