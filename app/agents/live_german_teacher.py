import asyncio
import json
import logging

from fastapi import WebSocket, WebSocketDisconnect
from google import genai
from google.genai import types
from nicegui import app
from websockets.exceptions import ConnectionClosed

from .german_teacher import SYSTEM_PROMPT


LIVE_MODEL = "gemini-3.1-flash-live-preview"
AUDIO_INPUT_MIME = "audio/pcm;rate=16000"
LOGGER = logging.getLogger("projekt_grimm.live_agent")


class LiveSessionClosed(Exception):
    """Raised when Gemini Live closes the upstream websocket."""


class LiveGermanTeacherAgent:
    """Bridge browser audio streams to a Gemini Live teacher session."""

    def __init__(self, api_key: str):
        self._client = genai.Client(api_key=api_key)
        self._route_registered = False
        LOGGER.info("Live German teacher agent initialized")

    def register_routes(self):
        """Register the websocket endpoint used by the live browser client."""
        if self._route_registered:
            return
        app.add_api_websocket_route("/api/live-audio", self._handle_websocket)
        self._route_registered = True
        LOGGER.info("Live audio websocket route registered")

    async def _handle_websocket(self, websocket: WebSocket):
        """Run one live audio session for a connected browser client."""
        await websocket.accept()
        receiver_task = None
        try:
            async with self._connect_session() as session:
                await self._send_status(websocket, "active", "Live-Unterhaltung verbunden.")
                receiver_task = asyncio.create_task(
                    self._relay_model_output(session, websocket)
                )
                await self._relay_browser_input(websocket, session)
        except LiveSessionClosed:
            LOGGER.info("Gemini Live session closed while audio was streaming")
        except WebSocketDisconnect:
            LOGGER.info("Live audio websocket disconnected")
        except Exception as exc:
            LOGGER.exception("Live audio session failed")
            await self._safe_send(websocket, self._error_payload(str(exc)))
        finally:
            await self._close_task(receiver_task)
            await self._safe_close(websocket)
            LOGGER.info("Live audio websocket closed")

    def _connect_session(self):
        """Create a configured Gemini Live session context manager."""
        return self._client.aio.live.connect(
            model=LIVE_MODEL,
            config=self._build_config(),
        )

    def _build_config(self) -> types.LiveConnectConfig:
        """Build the Gemini Live configuration for the teacher persona."""
        return types.LiveConnectConfig(
            response_modalities=[types.Modality.AUDIO],
            system_instruction=SYSTEM_PROMPT,
            input_audio_transcription=types.AudioTranscriptionConfig(),
            output_audio_transcription=types.AudioTranscriptionConfig(),
            realtime_input_config=types.RealtimeInputConfig(
                automatic_activity_detection=types.AutomaticActivityDetection(
                    disabled=True
                ),
                activity_handling=types.ActivityHandling.START_OF_ACTIVITY_INTERRUPTS,
                turn_coverage=types.TurnCoverage.TURN_INCLUDES_ALL_INPUT,
            ),
        )

    async def _relay_browser_input(self, websocket: WebSocket, session):
        """Forward browser audio chunks to the Gemini Live session."""
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                raise WebSocketDisconnect()
            if message.get("bytes"):
                await self._send_audio_chunk(session, message["bytes"])
                continue
            if message.get("text") == "stop":
                await self._finish_audio_stream(session)
                return
            if message.get("text"):
                await self._handle_browser_event(session, message["text"])

    async def _send_audio_chunk(self, session, chunk: bytes):
        """Send one browser PCM chunk to Gemini Live."""
        try:
            await session.send_realtime_input(
                audio=types.Blob(data=chunk, mime_type=AUDIO_INPUT_MIME)
            )
        except ConnectionClosed as exc:
            raise LiveSessionClosed() from exc

    async def _finish_audio_stream(self, session):
        """Tell Gemini Live that the browser stopped sending audio."""
        try:
            await session.send_realtime_input(audio_stream_end=True)
        except ConnectionClosed:
            return

    async def _handle_browser_event(self, session, payload: str):
        """Forward browser turn markers to Gemini Live."""
        event = json.loads(payload)
        event_type = event.get("type")
        if event_type == "activity_start":
            await self._send_activity_start(session)
        if event_type == "activity_end":
            await self._send_activity_end(session)

    async def _send_activity_start(self, session):
        """Mark the beginning of a user speech turn."""
        try:
            await session.send_realtime_input(activity_start=types.ActivityStart())
        except ConnectionClosed as exc:
            raise LiveSessionClosed() from exc

    async def _send_activity_end(self, session):
        """Mark the end of a user speech turn."""
        try:
            await session.send_realtime_input(activity_end=types.ActivityEnd())
        except ConnectionClosed as exc:
            raise LiveSessionClosed() from exc

    async def _relay_model_output(self, session, websocket: WebSocket):
        """Forward Gemini Live audio and transcripts back to the browser."""
        try:
            while True:
                received_response = False
                async for response in session.receive():
                    received_response = True
                    await self._forward_transcripts(response, websocket)
                    await self._forward_audio(response, websocket)
                if not received_response:
                    return
        except ConnectionClosed:
            return

    async def _forward_transcripts(self, response, websocket: WebSocket):
        """Send user and model transcripts to the browser UI."""
        content = response.server_content
        if not content:
            return
        await self._forward_transcript(websocket, "user", content.input_transcription)
        await self._forward_transcript(websocket, "model", content.output_transcription)

    async def _forward_transcript(self, websocket: WebSocket, speaker: str, transcript):
        """Send a non-empty transcript message to the browser."""
        if not transcript or not transcript.text:
            return
        payload = {"type": "transcript", "speaker": speaker, "text": transcript.text}
        await self._safe_send(websocket, payload)

    async def _forward_audio(self, response, websocket: WebSocket):
        """Send model audio chunks to the browser as websocket bytes."""
        server_content = response.server_content
        if not server_content or not server_content.model_turn:
            return
        for part in server_content.model_turn.parts:
            if part.inline_data and part.inline_data.data:
                await websocket.send_bytes(part.inline_data.data)

    async def _send_status(self, websocket: WebSocket, state: str, message: str):
        """Send a status update to the browser UI."""
        await self._safe_send(
            websocket,
            {"type": "status", "state": state, "message": message},
        )

    def _error_payload(self, message: str) -> dict[str, str]:
        """Build a consistent browser error payload."""
        return {"type": "error", "message": message}

    async def _safe_send(self, websocket: WebSocket, payload: dict[str, str]):
        """Send a JSON payload unless the websocket is already closing."""
        try:
            await websocket.send_text(json.dumps(payload, ensure_ascii=True))
        except RuntimeError:
            LOGGER.info("Skipped websocket send because the socket is closing")

    async def _close_task(self, task: asyncio.Task | None):
        """Cancel and await a background task if one exists."""
        if not task:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            return

    async def _safe_close(self, websocket: WebSocket):
        """Close the websocket unless the client already disconnected."""
        try:
            await websocket.close()
        except RuntimeError:
            LOGGER.info("Skipped websocket close because the socket is already closed")