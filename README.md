# vadsa

vadsa (from Swedish *"vad sa?"*, "what was said?") serves Swedish speech recognition with
the NeMo checkpoint [KlangAI/pianissimo-sv](https://huggingface.co/KlangAI/pianissimo-sv) on
an NVIDIA GPU. It is a stateless model server that presents itself like a vLLM speech server:
an OpenAI-compatible transcription endpoint, a model list and vLLM's realtime WebSocket.

[Vemsa](https://github.com/eneo-ai/vemsa) (*"vem sa?"*, "who said?") is the transcription
and speaker-diarization pipeline; vadsa only turns speech into text. It keeps no jobs, files
or database, and it does no diarization or alignment. Eneo registers vadsa as a vLLM
transcription model, both for flows and for their live preview, and Vemsa can use it as the
text source of its hybrid tier.

## Endpoints

| Route | Auth | Purpose |
| --- | --- | --- |
| `GET /health` | none | 200 once the model is loaded, 503 while it loads |
| `GET /v1/models` | key | The served model, marked as a transcription model |
| `POST /v1/audio/transcriptions` | key | OpenAI-compatible transcription of an uploaded file |
| `GET /v1/realtime` | key | WebSocket in vLLM's realtime dialect |

Keys are sent as `Authorization: Bearer <key>` and checked against `VADSA_API_KEYS`. The HTTP
contract is in [`openapi.json`](openapi.json); a test fails when it no longer matches the app.

```bash
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8000/health

curl -s http://127.0.0.1:8000/v1/models -H "Authorization: Bearer $VADSA_KEY"
# {"object":"list","data":[{"id":"KlangAI/pianissimo-sv","object":"model",
#   "created":1790157985,"owned_by":"vadsa","model_type":"transcription"}]}
```

### Transcriptions

`POST /v1/audio/transcriptions` takes multipart form fields like the OpenAI API:

- `file` (required): audio in any format ffmpeg reads, at most `VADSA_MAX_UPLOAD_BYTES`.
- `model` (required): the served model name, `KlangAI/pianissimo-sv` by default.
- `language`: `sv`. `auto` or no value also means `sv`.
- `response_format`: `json` (default), `text` or `verbose_json`.
- `timestamp_granularities[]`: `word`, `segment` or both, for `verbose_json` only; the
  default is `segment`.
- `prompt` and `temperature` are accepted for compatibility and ignored: the model takes no
  prompt and decoding is greedy.

The form carries one file and at most 32 other fields of 64 KiB together; fields not listed
here are ignored.

```bash
curl -s http://127.0.0.1:8000/v1/audio/transcriptions \
  -H "Authorization: Bearer $VADSA_KEY" \
  -F file=@recording.wav \
  -F model=KlangAI/pianissimo-sv \
  -F response_format=verbose_json \
  -F 'timestamp_granularities[]=word' \
  -F 'timestamp_granularities[]=segment'
```

```json
{
  "task": "transcribe",
  "language": "sv",
  "duration": 3.816,
  "text": "Hej, jag är 57 år gammal och min mamma är 102.",
  "segments": [{"id": 0, "start": 0.16, "end": 3.36, "text": "Hej, jag är 57 år gammal och min mamma är 102."}],
  "words": [{"word": "Hej,", "start": 0.16, "end": 0.48}, {"word": "jag", "start": 0.64, "end": 0.88}]
}
```

(The word list is cut short here.) `json` returns `{"text": ...}` and `text` returns the text
as `text/plain`. Times are seconds from the start of the file. The upload is streamed to a
temporary file, decoded with ffmpeg to 16 kHz mono and deleted before transcription starts.
Audio longer than `VADSA_WINDOW_SECONDS` is transcribed in windows, each cut at the quietest
200 ms in the 10 seconds before the window would end. If the client disconnects, the request
stops: a window still waiting for the GPU leaves the queue, one already on the GPU finishes
first, and no further window is queued.

Errors have the OpenAI shape, `{"error": {"message", "type", "param", "code"}}`:

| Status | `code` | When |
| --- | --- | --- |
| 400 | `invalid_audio` | ffmpeg cannot decode the file, or it is empty |
| 400 | `audio_too_long` | The audio is longer than `VADSA_MAX_AUDIO_SECONDS` |
| 400 | `unsupported_language` | `language` is not `sv` or `auto` |
| 400 | `unsupported_response_format` | `response_format` is not `json`, `text` or `verbose_json` |
| 400 | `null` | No `file` or a second file, more fields than the form takes, a malformed multipart body or one without its closing boundary, or an unknown granularity |
| 401 | `invalid_api_key` | The key is missing or wrong |
| 404 | `model_not_found` | `model` is not the served name |
| 413 | `file_too_large` | The upload is larger than `VADSA_MAX_UPLOAD_BYTES` |
| 503 | `model_loading` | The model is still loading; the response has `Retry-After` |
| 503 | `server_busy` | `VADSA_MAX_BATCH_REQUESTS` requests are in progress or queued; the response has `Retry-After` |

## Realtime

`GET /v1/realtime` takes base64 PCM16 audio at 16 kHz in `input_audio_buffer.append` events
and sends `transcription.delta` events while the audio is decoded, then `transcription.done`
after the client's final commit, and closes. The event shapes are vLLM's, so vLLM clients
such as Eneo's live preview work unchanged. Deltas only ever append and include
`audio_start`/`audio_end` in seconds of submitted audio; `transcription.done` includes
`audio_seconds` for coverage. The times describe step windows, not individual words, and
each new session starts at 0. The text trails speech by
about two seconds of model buffering, plus the time each step waits for the GPU and takes to
run. It is a preview: the transcription endpoint sees the whole recording and gives the better
transcript. Every refusal (a wrong key or model, too many sessions, a client sending audio
faster than the GPU decodes it, a time limit) is an `error` event followed by a close.
[`docs/realtime.md`](docs/realtime.md) has the event reference, the close codes and an example
client.

## Configuration

Settings are environment variables. [`.env.example`](.env.example) lists them with their
defaults.

| Variable | Default | Meaning |
| --- | --- | --- |
| `VADSA_ENVIRONMENT` | `production` | `production` refuses to start without `VADSA_API_KEYS` and with the fake engine; `development` allows both |
| `VADSA_API_KEYS` | none | Comma-separated bearer keys for `/v1/*` |
| `VADSA_ENGINE` | `nemo` | `nemo`, or `fake`: a deterministic stand-in without a model, for development |
| `VADSA_MODEL` | `KlangAI/pianissimo-sv` | Hugging Face name or local `.nemo` path of the checkpoint |
| `VADSA_SERVED_MODEL_NAME` | `KlangAI/pianissimo-sv` | The name clients send as `model` and `/v1/models` lists |
| `VADSA_MAX_UPLOAD_BYTES` | `1073741824` (1 GiB) | Largest upload |
| `VADSA_MAX_AUDIO_SECONDS` | `18000` (5 h) | Longest audio in a transcription request |
| `VADSA_WINDOW_SECONDS` | `600` | Longer audio is transcribed in windows of at most this length |
| `VADSA_MAX_BATCH_REQUESTS` | `4` | Transcription requests in progress or queued; more get 503 |
| `VADSA_MAX_SESSIONS` | `32` | Open realtime sessions; more are refused |
| `VADSA_MAX_SESSION_SECONDS` | `18000` (5 h) | Audio a realtime session may send; more is refused. Keep it at or above the client's own limit (Eneo's is 5 h), so the client's limit and error come first |
| `VADSA_MAX_SESSION_WALL_SECONDS` | `39600` (11 h) | Longest a realtime session may stay open before its final commit: a backstop for a client that never finishes. Must be longer than `VADSA_MAX_SESSION_SECONDS` |
| `VADSA_FINALIZE_SECONDS` | `45` | Longest from reading a session's final commit to sending its `transcription.done`, however long the session was open. Keep it below the client's own wait for the final text, with room for the commit's way here and the text's way back (Eneo waits 60 s) |
| `VADSA_IDLE_TIMEOUT_SECONDS` | `300` | A realtime session with no audio for this long before its final commit is closed |
| `VADSA_MAX_PENDING_SECONDS` | `30` | Audio a realtime session may have waiting for the GPU before it is ended |
| `VADSA_STREAM_CHUNK_SECONDS` | `1.0` | Realtime frame length; NeMo rounds it up to whole 80 ms model frames (1.04 s) |
| `VADSA_STREAM_LEFT_PADDING_SECONDS` | `6.0` | Audio before each frame that the model sees as context |
| `VADSA_STREAM_RIGHT_PADDING_SECONDS` | `1.0` | Audio after each frame that the model sees as context (1.04 s after rounding) |
| `HF_TOKEN` | none | Hugging Face token; the model is public, so this only raises download rate limits |

The compose files also read `VADSA_VERSION` (image tag, `latest`), `VADSA_BIND`
(`127.0.0.1`), `VADSA_PORT` (`8000`) and `VADSA_SHARED_NETWORK` (`eneo-shared`).

## Running with Docker Compose or Podman

The host needs an NVIDIA driver. Docker also needs the NVIDIA Container Toolkit; Podman
needs a CDI specification, generated once per host and again after each driver upgrade:

```bash
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
nvidia-ctk cdi list    # should list nvidia.com/gpu=all
```

Then, with either tool:

```bash
cp .env.example .env     # set VADSA_API_KEYS
docker compose up -d     # or: podman compose up -d, or: podman-compose up -d
```

`compose.yaml` reserves the GPU with `deploy.resources.reservations.devices`. Docker Compose
hands that to the NVIDIA runtime; podman-compose 1.5 turns it into `--device
nvidia.com/gpu=all` and adds `--security-opt label=disable`. The service listens on
`127.0.0.1:8000`; set `VADSA_BIND` and `VADSA_PORT` to change that. The container runs as a
non-root user with a read-only root file system, no capabilities and no privilege
escalation.

The first start downloads the checkpoint (about 2.5 GB) into the `models` volume; later
starts load it from there. `/health` answers 503 until both model instances are loaded. The
container's `/tmp` is a 4 GB tmpfs: NeMo unpacks the checkpoint there while loading, and
uploads wait there while ffmpeg decodes them. Each transcription request also holds its
decoded audio in memory, 64 KB per second of audio (about 1.2 GB for 5 hours).

To reach vadsa from other containers by name instead of a published port, add the shared
network overlay:

```bash
docker network create eneo-shared    # once, or reuse the network Eneo already uses
docker compose -f compose.yaml -f compose.eneo.yaml up -d
```

Other containers on that network reach it at `http://vadsa:8000`.

The `-cpu` images run without a GPU, slowly, for trying vadsa out. `compose.yaml` reserves a
GPU, so start a CPU image directly:

```bash
docker run --rm -p 127.0.0.1:8000:8000 --env-file .env -v vadsa-models:/models \
  ghcr.io/eneo-ai/vadsa:latest-cpu
```

## Running next to vLLM

vadsa is meant to share a GPU with a vLLM server and with Vemsa. vLLM claims the share of GPU
memory given by `--gpu-memory-utilization` (0.9 by default) when it starts, so lower that
value to leave room for vadsa. vadsa holds two instances of the model (one for realtime, one
for transcription requests) plus the working memory of the current step or window. Its peak
depends on `VADSA_WINDOW_SECONDS` and on how many realtime sessions run at once, so measure it
with both workloads running, for example a long upload while realtime sessions are open,
and watch `nvidia-smi --query-gpu=memory.used --format=csv -l 1`. This README promises no
number.

Inside vadsa one thread makes every model call. When realtime sessions and transcription
requests wait at the same time, it alternates one realtime step (every session with a frame
ready, decoded together) and one transcription window. Transcription always progresses, and
a realtime step waits at most one window, which is why `VADSA_WINDOW_SECONDS` also bounds how
far live text can fall behind while a long file is transcribed.

## Using it from Eneo

1. Add a model provider of type vLLM. The endpoint is vadsa's root URL without `/v1`, for
   example `http://vadsa:8000`, and the provider key is one of `VADSA_API_KEYS`.
2. Add a transcription model named `KlangAI/pianissimo-sv` (the served model name). Eneo's
   model listing reads `/v1/models`, where vadsa marks the model with
   `"model_type": "transcription"`.
3. Tick "Livetranskribering" on the model to let flows show text while someone records.
4. Live preview also needs Eneo's transcription service in mode `diarize`, or no
   transcription service at all. In `full` mode Vemsa transcribes the recording itself, so
   the final text would come from another model than the preview.

Eneo sends transcription requests through LiteLLM's `hosted_vllm` provider, which posts
OpenAI multipart to `{endpoint}/v1/audio/transcriptions`, and connects the live preview to
`{endpoint}/v1/realtime`.

## Using it as Vemsa's text source

Vemsa's hybrid tier takes its text from an OpenAI-compatible endpoint and aligns it locally.
Point it at vadsa:

```bash
VEMSA_WHISPER_API_BASE=http://vadsa:8000/v1
VEMSA_WHISPER_API_KEY=<one of VADSA_API_KEYS>
VEMSA_DEFAULT_MODEL=KlangAI/pianissimo-sv
```

Unlike Eneo's endpoint, Vemsa's base URL includes `/v1`. Vemsa asks for `verbose_json` with
word and segment timestamps, which vadsa returns.

## Development

Requires [uv](https://docs.astral.sh/uv/) and ffmpeg.

```bash
uv sync --group dev
uv run pytest
uv run ruff format --check . && uv run ruff check .

# a local server on the fake engine: no torch, no GPU, no model download
VADSA_ENVIRONMENT=development VADSA_ENGINE=fake uv run uvicorn vadsa.main:create_app --factory
```

The fake engine hears one word per second of non-silent audio (`ord0 ord1 ...`), so every
endpoint and the realtime protocol can be tested without torch. The tests use it throughout;
they decode real uploads with ffmpeg. After an intended change to the HTTP API, rewrite
`openapi.json` with `UPDATE_OPENAPI=1 uv run pytest tests/test_openapi.py`.

To run the real model locally on CPU, slowly, install NeMo with CPU torch. Always pair the
`nemo` extra with exactly one of `cpu` and `gpu`, which route torch to the right package
index:

```bash
uv sync --group dev --extra nemo --extra cpu
VADSA_ENVIRONMENT=development uv run uvicorn vadsa.main:create_app --factory
```

The code:

- `src/vadsa/main.py`: the app factory; the model loads in the background.
- `src/vadsa/api/`: one module per endpoint.
- `src/vadsa/engine/scheduler.py`: the thread that owns the GPU, and the admission limits.
- `src/vadsa/engine/nemo.py` and `fake.py`: the two engines behind `engine/base.py`.
- `src/vadsa/audio.py`: ffmpeg decoding, PCM conversion and window cuts.
- `src/vadsa/conf/buffered_tdt.yaml`: the NeMo streaming configuration.

## Releases

CI runs ruff and the tests on every push and pull request. A push to `main` or a `v*` tag
then publishes images to `ghcr.io/eneo-ai/vadsa`:

- `latest` and `main` from `main`, `X.Y.Z` and `X.Y` from a tag `vX.Y.Z`, and `sha-<commit>`
  for every build, all with CUDA torch;
- the same tags with a `-cpu` suffix (`latest-cpu`, `0.1.0-cpu`) with CPU torch.

The images are for `linux/amd64`. Set `VADSA_VERSION` in `.env` to pin a tag.

## Licence and model attribution

vadsa is licensed under AGPL-3.0-or-later, © Sundsvalls Kommun. See [`LICENSE`](LICENSE).

The model, [KlangAI/pianissimo-sv](https://huggingface.co/KlangAI/pianissimo-sv), is made by
Klang and licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). It is a
fine-tune of NVIDIA's [Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3).
vadsa downloads it at run time; the model is not part of this repository or its images.
`src/vadsa/conf/buffered_tdt.yaml` is adapted from NVIDIA NeMo, licensed under Apache 2.0.
