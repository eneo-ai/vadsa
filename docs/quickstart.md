# vadsa quick start

vadsa is a transcription server. It writes the text of Swedish speech with the
[KlangAI/pianissimo-sv](https://huggingface.co/KlangAI/pianissimo-sv) model, for a finished
recording and live while someone records. It does not say who spoke: in Eneo that is the job
of a speaker identification service such as [Vemsa](https://github.com/eneo-ai/vemsa).

## What you need

- A `linux/amd64` host with an NVIDIA GPU and its driver.
- Docker with the NVIDIA Container Toolkit, or Podman with a CDI specification. See
  [Running with Docker Compose or Podman](../README.md#running-with-docker-compose-or-podman).
- Internet access on the first start, which downloads the model (about 2.5 GB). The model is
  public, so you need no Hugging Face token.

## Start it

`VADSA_API_KEYS` is the only setting you must set. Compose reads it from `.env`.

```bash
git clone https://github.com/eneo-ai/vadsa.git
cd vadsa
echo "VADSA_API_KEYS=change-me-long-random-value" > .env
docker compose up -d     # or: podman compose up -d
```

This publishes vadsa on `127.0.0.1:8000`. To let Eneo's containers on the same host reach it
by name instead, start it on a shared Docker network:

```bash
docker network create eneo-shared    # once, or reuse the network Eneo already uses
docker compose -f compose.yaml -f compose.eneo.yaml up -d
```

To try vadsa on a host without a GPU, run the CPU image. It is slow.

```bash
docker run --rm -p 127.0.0.1:8000:8000 --env-file .env -v vadsa-models:/models \
  ghcr.io/eneo-ai/vadsa:latest-cpu
```

## Check that it runs

```bash
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/v1/models -H "Authorization: Bearer change-me-long-random-value"
```

The first command prints `200` when the model is loaded, and `503` while it loads. The second
answers with the served model, which shows that the key works:

```json
{"object":"list","data":[{"id":"KlangAI/pianissimo-sv","object":"model","created":1790157985,"owned_by":"vadsa","model_type":"transcription"}]}
```

## Connect it to Eneo

Eneo's backend and workers must reach vadsa. As an administrator in Eneo, enter this address:

| Setup | Address |
| --- | --- |
| Shared Docker network (`compose.eneo.yaml`) | `http://vadsa:8000` |
| Published port | `http://<VADSA_BIND>:<VADSA_PORT>`. The defaults give `http://127.0.0.1:8000`, which only the host itself reaches, so set `VADSA_BIND` to a private address for an Eneo on another host. |

1. In **Admin → Models** (**Administratör → Modeller**), add a provider of type **vLLM**. Set
   **Endpoint** to the address from the table, without `/v1`, and **API Key** to the key in
   `.env`.
2. Add a transcription model on that provider, with the identifier `KlangAI/pianissimo-sv`.
3. In **Admin → Models → Transcription models** (**Transkription**), open the model's actions
   menu (the **⋯** button) and choose **Edit model** (**Redigera modell**). Select
   **Live transcription** (**Livetranskribering**) and choose **Save**.
4. Open the space's **Settings** (**Inställningar**). Under **Advanced Settings**
   (**Avancerade inställningar**), switch the model on in **Transcription models**.
5. In the flow's **Transcription** stage (**Transkribering**), turn transcription on, choose
   the model in **Model**, set **Language** to **Svenska** or **Detect automatically**, and
   publish the flow. vadsa transcribes Swedish only.
6. Check the result. Run the flow with a short Swedish recording. The run shows the
   transcript. For live text, request `GET /api/v1/flows/{id}/run-contract/` from Eneo:
   `transcription.live.available` is `true`.

## Use it

A person who runs the flow with a recording gets its text as the transcript. An app that
records for the flow can show the text while the person speaks, about two seconds behind the
speech and more when the GPU is busy. vadsa adds no speaker labels. For lines that start
with `SPEAKER_00`, also connect Vemsa: see the
[Vemsa quick start](https://github.com/eneo-ai/vemsa/blob/main/docs/QUICKSTART.md).

## Next

- [Endpoints](../README.md#endpoints) and [Configuration](../README.md#configuration): every
  route, limit and setting.
- [Realtime protocol](realtime.md): the live events, limits and close codes.
- [Running next to vLLM](../README.md#running-next-to-vllm): sharing a GPU with other servers.
- [Using it from Eneo](../README.md#using-it-from-eneo): the requests Eneo sends.
- [Development](../README.md#development): a fake engine that needs no GPU and no model.
