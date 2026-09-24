# Realtime transcription

`GET /v1/realtime` is a WebSocket that speaks vLLM's realtime transcription dialect. The
event shapes are the ones in vLLM's `vllm/entrypoints/speech_to_text/realtime/protocol.py`,
so a client written for a vLLM speech server works against vadsa. Eneo's live preview is
such a client.

## Connecting

```
ws://<host>:8000/v1/realtime
Authorization: Bearer <key>
```

The key is checked against `VADSA_API_KEYS`, as for the HTTP endpoints. A query string such
as `?intent=transcription` is accepted and ignored.

## A session

1. The server accepts the connection and sends `session.created`.
2. The client sends `session.update` with the model name, then a commit without `final`.
   vLLM clients send both before any audio; vadsa checks the model and otherwise
   ignores the commit.
3. The client sends `input_audio_buffer.append` events, for example one per 100 ms of
   audio. The server sends `transcription.delta` events as text is decoded.
4. The client sends `input_audio_buffer.commit` with `"final": true` when the audio ends.
   The server decodes what is left, sends the remaining deltas, then
   `transcription.done`, and closes the connection with code 1000. Until then it still
   reads the connection and ignores any further events; a disconnect, or no final text
   within `VADSA_FINALIZE_SECONDS` of the final commit, ends the session and drops what
   was not yet decoded.

A session is limited by the audio it sends, not by how long it stays open: at most
`VADSA_MAX_SESSION_SECONDS` of audio (5 h), however long the recording pauses in between.
Only a session that stays open for `VADSA_MAX_SESSION_WALL_SECONDS` (11 h) without its
final commit is ended regardless. A client's own audio limit should be at most
`VADSA_MAX_SESSION_SECONDS`, so its own error comes first.

After the final commit, `transcription.done` has to be sent within
`VADSA_FINALIZE_SECONDS` (45 s), however long the session was open. What is left then is
the audio still waiting for the GPU, at most `VADSA_MAX_PENDING_SECONDS`: one realtime step
per frame of it. While transcription requests wait as well, the GPU alternates, so each of
those steps can first wait for one transcription window (up to `VADSA_WINDOW_SECONDS` of
audio), after whatever the GPU is running at the commit. A session whose audio the GPU
keeps up with has one or two steps left.

The grace starts when vadsa reads the final commit and ends when `transcription.done` is
sent. The client's wait starts earlier, when it sends the commit, and ends later, when the
text arrives. `VADSA_FINALIZE_SECONDS` plus both of those trips must fit in the client's
wait: Eneo waits 60 s, which leaves 15 s with the default.

Audio is 16 kHz mono PCM16, little endian, base64 encoded. The server decodes it in frames
of about one second (1.04 s with the default settings, the requested
`VADSA_STREAM_CHUNK_SECONDS` rounded up to whole 80 ms model frames). A frame is decoded
once the next frame has started to arrive, with one more frame of audio after it as context
(`VADSA_STREAM_RIGHT_PADDING_SECONDS`). That buffering puts text about two seconds behind
speech. The time a step waits for the GPU and takes to run comes on top; while a long file
is transcribed, a step can wait for one whole transcription window.

Deltas only append: text that has been sent is never revised. Joined together they equal
the `text` of `transcription.done`. The text is a preview; the transcription endpoint
decodes a recording with the whole recording as context and is the better transcript.

Each session's audio starts with half a second of silence that the server adds. In a test
on CPU with the default sizes, NeMo's streaming lost the first word of a recording that
opens with speech; with the silence ahead of it, the preview of that recording had the same
text as the transcription endpoint.

## Client events

| Event | Fields | Meaning |
| --- | --- | --- |
| `session.update` | `model` | Must be the served model name. |
| `input_audio_buffer.append` | `audio` | Base64 PCM16 LE mono 16 kHz, at most 1 MiB of audio per event. |
| `input_audio_buffer.commit` | `final` (default `false`) | `true` ends the audio; events after it are ignored. Without it the event has no effect. |

## Server events

```json
{"type": "session.created", "id": "sess-3f1c9b0e2d8a4b7c9e6f5a4b3c2d1e0f", "created": 1790157305}
{"type": "transcription.delta", "delta": "Hej och", "audio_start": 0.0, "audio_end": 0.54}
{"type": "transcription.delta", "delta": " välkomna.", "audio_start": 0.54, "audio_end": 2.5}
{"type": "transcription.done", "text": "Hej och välkomna.", "usage": null, "audio_seconds": 2.5}
{"type": "error", "error": "Too many realtime sessions.", "code": "capacity_exceeded"}
```

`usage` is always null. `error` is a message for people; clients should act on `code`.

`audio_start` and `audio_end` describe the delta's token-commit window in float seconds
of submitted audio. The clock starts at the first submitted sample and excludes
the server's lead-in silence; a new session starts a new clock at 0. Both bounds are
clamped to the submitted audio length and never decrease across deltas. The start has
millisecond precision. The delta from the final step ends at the submitted audio length.
Silent steps send no delta, so a session without speech sends only `transcription.done`.

For frame duration `F`, effective commit delay `D`, and lead-in `L`, step `k` covers
`[(k - 1) * F - D - L, k * F - D - L)`, clamped to the submitted audio. NeMo's delay
is its right padding rounded up to whole model frames, independently of the chunk size.
The fake engine commits its current frame and has no delay.

With the default 1.04 s frames, 1.04 s effective right padding and 0.5 s lead-in, this is
`[(k - 2) * 1.04 - 0.5, (k - 1) * 1.04 - 0.5)`, clamped to the submitted audio. The
final step also covers the remaining tail.

These are token-commit windows, not word-level timestamps or a guarantee that every
whole word lies within one delta's window. A word can be split across consecutive
deltas; a reconstructed word spans the combined windows of all deltas containing its
tokens. For example, the Swedish CPU probe split `ekonomiavdelningen` across steps
18–20. Consumers reconstructing words must retain those combined windows.

`audio_seconds` on `transcription.done` is the number of received samples divided by
16000, including silence and excluding the server's lead-in. It records the audio
covered even when no text was produced. These timing fields extend vLLM's event shapes;
clients that ignore unknown fields continue to work.

## Errors and close codes

Every refusal is an `error` event followed by a close. A session never continues after
an error.

| `code` | Close | When |
| --- | --- | --- |
| `invalid_api_key` | 1008 | The key is missing or wrong. Sent instead of `session.created`. |
| `model_not_found` | 1008 | `session.update` names another model. |
| `invalid_json`, `invalid_event`, `unknown_event`, `invalid_audio` | 1008 | An event could not be read: not JSON, a missing or wrong field, an unknown type, or audio that is not base64 PCM16. |
| `frame_too_large` | 1009 | One append carries more than 1 MiB of audio. |
| `capacity_exceeded` | 1013 | `VADSA_MAX_SESSIONS` sessions are open. Sent instead of `session.created`. |
| `model_loading` | 1013 | The model has not finished loading. Sent instead of `session.created`. |
| `falling_behind` | 1013 | More than `VADSA_MAX_PENDING_SECONDS` of the session's audio waits for the GPU. |
| `internal_error` | 1011 | Decoding failed on the server. |
| `session_too_long` | 1000 | An append would take the session past `VADSA_MAX_SESSION_SECONDS` of audio, or the session was open for `VADSA_MAX_SESSION_WALL_SECONDS` without its final commit. |
| `finalize_timeout` | 1013 | `transcription.done` could not be sent within `VADSA_FINALIZE_SECONDS` of the final commit: the text was not ready, or the client did not take it. |
| `idle_timeout` | 1000 | No append arrived for `VADSA_IDLE_TIMEOUT_SECONDS` before the final commit. |

Close code 1013 means try again later. A client that disconnects, or a session that ends
with an error, loses the audio that was not yet decoded; nothing is kept on the server.

## Differences from vLLM

- vadsa closes the connection after `transcription.done`; vLLM keeps it open for another
  utterance. Open a new session for the next recording.
- vadsa ends the session on any error; vLLM keeps the connection open after some of them.
- vadsa does not require `session.update` before audio, but checks it when it comes.

## Example client

This streams a file at speaking pace, as a microphone would, and prints the text. It uses
the `websockets` package, which `uvicorn[standard]` already installs, and ffmpeg to turn
any audio file into PCM16.

```python
import asyncio
import base64
import json
import os
import subprocess
import sys

import websockets

URL = "ws://127.0.0.1:8000/v1/realtime"
MODEL = "KlangAI/pianissimo-sv"
CHUNK = 3200  # 100 ms of 16 kHz PCM16


async def main(path: str) -> None:
    pcm = subprocess.run(
        ["ffmpeg", "-i", path, "-ac", "1", "-ar", "16000", "-f", "s16le", "-"],
        capture_output=True,
        check=True,
    ).stdout
    headers = {"Authorization": f"Bearer {os.environ['VADSA_KEY']}"}
    async with websockets.connect(URL, additional_headers=headers) as ws:
        print(json.loads(await ws.recv()), file=sys.stderr)  # session.created
        await ws.send(json.dumps({"type": "session.update", "model": MODEL}))
        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

        async def send_audio() -> None:
            for start in range(0, len(pcm), CHUNK):
                audio = base64.b64encode(pcm[start : start + CHUNK]).decode()
                await ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": audio}))
                await asyncio.sleep(0.1)
            await ws.send(json.dumps({"type": "input_audio_buffer.commit", "final": True}))

        sender = asyncio.create_task(send_audio())
        async for message in ws:
            event = json.loads(message)
            if event["type"] == "transcription.delta":
                print(event["delta"], end="", flush=True)
            elif event["type"] == "transcription.done":
                print()
            elif event["type"] == "error":
                print(f"{event['code']}: {event['error']}", file=sys.stderr)
        sender.cancel()


asyncio.run(main(sys.argv[1]))
```

```bash
VADSA_KEY=... uv run python client.py meeting.m4a
```
