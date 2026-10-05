# Passive Liveness v2

Server-side passive face presentation attack detection (PAD) API.
A frozen DINOv2 (+registers) backbone with a PAD head classifies a
single face frame as **live** or **spoof**; the client uploads the
sharpest frame of a capture burst (see `architecture.md`).

## Run

```bash
uv run uvicorn passive_liveness_v2.main:app --reload   # dev
uv run passive-liveness-v2                             # serve
```

The model checkpoint is loaded once at startup. On first run, the
DINOv2 backbone config is fetched from the Hugging Face Hub
(`facebook/dinov2-with-registers-base`); weights come from the local
checkpoint, so no model download is needed.

## API

| Method | Route | Description |
| ------ | ----- | ----------- |
| GET | `/v1/health` | Liveness probe (no model check) |
| GET | `/v1/ready` | Readiness + model metadata (503 until loaded) |
| POST | `/v1/liveness` | Multipart image upload -> live/spoof decision |

Example:

```bash
curl -F "file=@face_frame.jpg" http://localhost:8000/v1/liveness
# {"request_id": "...", "model": "vitb_pad_v1", "label": "live",
#  "score": 0.9387, "threshold": 0.9353}
```

Every error is returned as `{"error": {"code", "message", "request_id"}}`
and each response carries an `X-Request-ID` header.

## Configuration

All settings use the `LIVENESS_` env prefix (or a `.env` file):

| Variable | Default | Description |
| -------- | ------- |-------------|
| `LIVENESS_CHECKPOINT_PATH` | `pl_checkpoints/vitb_pad_v1.pt` | Model checkpoint |
| `LIVENESS_BACKBONE_MODEL_ID` | `facebook/dinov2-with-registers-base` | Backbone architecture (HF id) |
| `LIVENESS_THRESHOLD` | `0.9353` | Live/spoof decision threshold (from `eval_output/scores.txt`) |
| `LIVENESS_DEVICE` | `auto` | `auto` / `cuda` / `cpu` |
| `LIVENESS_TARGET_SIZE` | `224` | Input resolution |
| `LIVENESS_MAX_UPLOAD_BYTES` | `10485760` | Max upload size |
| `LIVENESS_HOST`, `LIVENESS_PORT` | `0.0.0.0`, `8000` | Bind address |

## Project structure

```
src/passive_liveness_v2/
├── main.py                # app factory, lifespan, uvicorn entrypoint
├── api/                   # HTTP layer
│   ├── deps.py            # dependency injection
│   └── v1/                # versioned routes
│       ├── router.py
│       ├── health.py      #   GET /v1/health, GET /v1/ready
│       ├── liveness.py    #   POST /v1/liveness
│       └── schemas.py     #   Pydantic request/response models
├── core/
│   ├── config.py          # pydantic-settings (env-driven)
│   └── errors.py          # error envelope, handlers, request-ID middleware
├── inference/
│   ├── model.py           # LivenessModel (ported from notebook.py)
│   ├── preprocessing.py   # decode -> resize 224 -> ImageNet normalize
│   └── service.py         # checkpoint loading + prediction
├── static/ + templates/   # optional demo UI (future)
```

## Known gaps (serve-time)

- **Geometry branch**: the PAD head expects 7 landmark-ratio features;
  until the landmark extraction pipeline (see `preprocess.py`) is
  ported, these are zeros. Scores are real but approximate.
- **Depth branch**: falls back to zeros, matching the notebook's own
  no-depth-model path. `Depth-Anything-V2-Small-hf` is cached locally
  in the HF hub cache if wiring it later.

A `POST /v1/liveness/clip` route for the 8-frame temporal cascade
(architecture.md section 6) is planned for when the temporal branch is
built.
