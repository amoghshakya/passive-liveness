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

Then open http://localhost:8000 for the demo frontend: a live
camera viewfinder that captures a short burst, picks the sharpest
frame on-device, and shows the server's live/spoof verdict.

On first run, the DINOv2 backbone config is fetched from the
Hugging Face Hub and InsightFace downloads its detection weights
(~16 MB); both are cached locally. Model weights themselves come
from the local checkpoint, so no model download is needed.

## API

| Method | Route | Description |
| ------ | ----- | ----------- |
| GET | `/` | Demo frontend (camera capture + verdicts) |
| GET | `/health` | Liveness probe, unversioned alias of `/v1/health` |
| GET | `/ready` | Readiness, unversioned alias of `/v1/ready` |
| GET | `/v1/health` | Liveness probe (no model check) |
| GET | `/v1/ready` | Readiness + model metadata (503 until loaded) |
| POST | `/v1/liveness` | Multipart image upload -> live/spoof decision |

Example:

```bash
curl -F "file=@face_frame.jpg" http://localhost:8000/v1/liveness
# {"request_id": "...", "model": "vitb_pad_v1", "label": "live",
#  "score": 0.9387, "threshold": 0.9353,
#  "face": {"bbox": [x1, y1, x2, y2], "det_score": 0.99}}

# Frames where the detector misses a visible face:
curl -F "file=@face_frame.jpg" -F "skip_detection=true" http://localhost:8000/v1/liveness
# "face": null — center-crop classification, no quality gate
```

Every error is returned as `{"error": {"code", "message", "request_id"}}`
and each response carries an `X-Request-ID` header. Quality-gate
failures (no face, low confidence, too small, blurry) return 422
with codes `no_face_detected`, `low_confidence`, `face_too_small`,
`low_quality`, `extreme_pose`.

**`skip_detection`** (form field, default `false`) bypasses face
detection, the quality gate and alignment: a center square crop is
classified directly with zeroed geometry. Use it for frames that
visibly contain a face but defeat the detector — scores are
approximate because the model was trained on aligned face crops.

## Serving pipeline

Mirrors the extraction pipeline in `preprocess.py` so train and
serve stay numerically consistent:

```
decode -> RetinaFace detection -> primary face (confidence x area)
-> quality gate (confidence, min size, Laplacian blur)
-> 5-point alignment (ArcFace ref, adaptive margin, 512px)
-> 7 geometry ratios -> resize 224 -> ImageNet normalize -> PAD head
```

With `skip_detection=true`: `decode -> center square crop ->
resize 512 -> resize 224 -> normalize -> PAD head` (zeroed
geometry; no detection, quality gate or alignment).

## Configuration

All settings use the `LIVENESS_` env prefix (or a `.env` file):

| Variable | Default | Description |
| -------- | ------- |-------------|
| `LIVENESS_CHECKPOINT_PATH` | `pl_checkpoints/vitb_pad_v1.pt` | Model checkpoint |
| `LIVENESS_BACKBONE_MODEL_ID` | `facebook/dinov2-with-registers-base` | Backbone architecture (HF id) |
| `LIVENESS_THRESHOLD` | `0.9353` | Live/spoof decision threshold (from `eval_output/scores.txt`) |
| `LIVENESS_DEVICE` | `auto` | `auto` / `cuda` / `cpu` |
| `LIVENESS_TARGET_SIZE` | `224` | Model input resolution |
| `LIVENESS_FACE_SIZE` | `512` | Aligned crop side length |
| `LIVENESS_ADAPTIVE_CROP_MARGIN` | `false` | Derive crop margin from face-to-frame area |
| `LIVENESS_BLUR_THRESHOLD` | `15.0` | Laplacian-variance blur gate (extraction used 40.0) |
| `LIVENESS_MAX_UPLOAD_BYTES` | `none` | Max upload size in bytes (`none` = unlimited) |
| `LIVENESS_HOST`, `LIVENESS_PORT` | `0.0.0.0`, `8000` | Bind address |

## Project structure

```
src/passive_liveness_v2/
├── main.py                # app factory, lifespan, uvicorn entrypoint
├── api/                   # HTTP layer
│   ├── deps.py            # dependency injection
│   ├── demo.py            # GET / demo frontend
│   ├── probes.py          # GET /health, GET /ready (unversioned aliases)
│   └── v1/                # versioned routes
│       ├── router.py
│       ├── health.py      #   GET /v1/health, GET /v1/ready
│       ├── liveness.py    #   POST /v1/liveness
│       └── schemas.py     #   Pydantic request/response models
├── core/
│   ├── config.py          # pydantic-settings (env-driven)
│   └── errors.py          # error envelope, handlers, request-ID middleware
├── inference/
│   ├── detection.py       # RetinaFace detection (InsightFace)
│   ├── alignment.py       # 5-point ArcFace alignment
│   ├── geometry.py        # 7 landmark ratios
│   ├── quality.py         # quality gate (non-response path)
│   ├── model.py           # LivenessModel (ported from notebook.py)
│   ├── preprocessing.py   # resize 224 -> ImageNet normalize
│   └── service.py         # pipeline orchestration + checkpoint loading
├── static/ + templates/   # demo frontend (index.html)
```

## Known gaps (serve-time)

- **Depth branch**: falls back to zeros, matching the notebook's
  own no-depth-model path. `Depth-Anything-V2-Small-hf` is cached
  locally in the HF hub cache if wiring it later.

A `POST /v1/liveness/clip` route for the 8-frame temporal cascade
(architecture.md section 6) is planned for when the temporal branch is
built.

## Using on your phone

The in-browser camera needs HTTPS. Over plain HTTP
(`http://<computer-ip>:8000` from your phone):

- **Take photo** opens your phone's native camera directly
  (works over HTTP); **Upload photo** picks from the gallery.
- For the live viewfinder on the phone, tunnel the server:
  `ngrok http 8000`, then open the `https://` URL on the phone.
