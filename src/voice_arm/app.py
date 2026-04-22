from __future__ import annotations

import argparse
import asyncio
import logging
import signal
from typing import Any

from .arm import ArmController, MockArm
from .audio import MicStream, Speaker
from .config import Settings, load_settings
from .llm import GeminiLiveSession
from .tools import ToolDispatcher, ToolResult

logger = logging.getLogger(__name__)


def _make_arm(settings: Settings) -> ArmController:
    if settings.arm_backend == "mock":
        return MockArm()
    if settings.arm_backend == "so101":
        from .arm.so101 import SO101Arm

        return SO101Arm()
    raise ValueError(f"unknown arm backend: {settings.arm_backend}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="voice-arm")
    parser.add_argument(
        "--arm",
        choices=["mock", "so101"],
        default=None,
        help="Override ARM_BACKEND from env.",
    )
    parser.add_argument("--log-level", default=None)
    return parser.parse_args()


async def _run(settings: Settings) -> None:
    arm = _make_arm(settings)
    dispatcher = ToolDispatcher(arm)

    async with Speaker() as speaker, MicStream() as mic:

        async def on_tool_call(name: str, args: dict[str, Any]) -> ToolResult:
            return await dispatcher.dispatch(name, args)

        async def on_audio(pcm: bytes) -> None:
            await speaker.play(pcm)

        async def on_interrupt() -> None:
            speaker.clear()

        session = GeminiLiveSession(
            api_key=settings.gemini_api_key.get_secret_value(),
            model=settings.gemini_model,
            on_tool_call=on_tool_call,
            on_audio=on_audio,
            on_interrupt=on_interrupt,
        )

        async with session.connect():

            async def mic_to_gemini() -> None:
                async for frame in mic.frames():
                    await session.send_audio_frame(frame)

            send_task = asyncio.create_task(mic_to_gemini(), name="mic_to_gemini")
            recv_task = asyncio.create_task(session.receive_loop(), name="recv_loop")

            done, pending = await asyncio.wait(
                {send_task, recv_task},
                return_when=asyncio.FIRST_EXCEPTION,
            )
            for task in pending:
                task.cancel()
            for task in done:
                if (exc := task.exception()) is not None:
                    raise exc

    # Safety: on exit, stop and home the arm.
    await arm.stop()
    await arm.home()


def main() -> None:
    args = _parse_args()
    settings = load_settings()
    if args.arm is not None:
        settings = settings.model_copy(update={"arm_backend": args.arm})
    if args.log_level is not None:
        settings = settings.model_copy(update={"log_level": args.log_level})

    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
    )
    logger.info("voice_arm starting (arm=%s)", settings.arm_backend)

    loop = asyncio.new_event_loop()

    def _request_shutdown() -> None:
        logger.info("shutdown requested")
        for task in asyncio.all_tasks(loop):
            task.cancel()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _request_shutdown)
        except NotImplementedError:  # pragma: no cover — Windows
            pass

    try:
        loop.run_until_complete(_run(settings))
    except asyncio.CancelledError:
        logger.info("cancelled — exiting cleanly")
    finally:
        loop.close()


if __name__ == "__main__":
    main()
