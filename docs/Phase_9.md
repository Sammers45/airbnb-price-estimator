# Phase 9: Price service, web page and monitoring (CPU)

Everything from Phases 3-8 behind one web page:

```
web page ──POST /predict──► FastAPI (src/serve.py)
                              1. LightGBM baseline (Phase 3)           -> base price
                              2. fusion network (Phase 6) + description (Phase 4 text model)
                                 + photo (CLIP, Phase 5)               -> final price
                              3. calibrator (Phase 8)                  -> 80% range, review flag
                              4. SHAP (Phase 8)                        -> top reasons
                              5. append to data/monitoring/requests.jsonl
                     ◄── price, range, review flag, reasons
python -m src.monitor  ──►  reports/monitoring.md  (also GET /monitoring, shown on the page)
```

| File                          | What it does                                                                      |
| ----------------------------- | --------------------------------------------------------------------------------- |
| `src/serve.py`                | `build`: form options + a typical listing per market. `run`: the API and web page |
| `src/monitor.py`              | reference data, drift (PSI), pipeline health, outcomes, a replay simulator        |
| `web/index.html`              | the page: form, photo upload, result, monitoring panel (plain HTML + JavaScript)  |
| `Dockerfile`, `.dockerignore` | one CPU container with the models and CLIP inside                                 |

## Design decisions

- **A form can't ask for everything.** The models use ~50 features (review scores, host
  details, coordinates, ...). The page asks for what a host knows about a new listing: market,
  neighbourhood, room and property type, guests, bedrooms, bathrooms, amenities, title,
  description and photo. Everything else takes the **typical value for that market** (the
  median, or the most common category), stored by `python -m src.serve build`. Picking a
  neighbourhood sets its typical coordinates.
- **Which model is served:** `serve.model: fusion` (the Phase 7 best, which the Phase 8 ranges
  were calibrated for). If its files are missing, the service falls back to the LightGBM
  baseline and says so in `/health`. Phase 7 found no significant difference, so
  `model: baseline` is a valid, simpler choice.
- **Missing inputs are normal.** No description, or no text model: the text input is masked,
  exactly as modality dropout did in training. No photo: the photo input is masked. If a photo
  can't be processed (unreadable file, CLIP can't download), the price still comes back
  without it, with a note.
- **Input validation:** guests 1-30, bedrooms 0-20, and so on; an unknown market or a broken
  image gives a clear 422 error instead of a wrong price.

## Monitoring

`python -m src.monitor` reads the request log and writes `reports/monitoring.md`; the page
shows the same summary.

| Check            | How                                                               | Alert when                                |
| ---------------- | ----------------------------------------------------------------- | ----------------------------------------- |
| Pipeline health  | errors, latency p50/p95, % with photo / description               | error rate > 1%, p95 > 2 s                |
| Input drift      | PSI of guests, bedrooms, bathrooms, market, room type vs training | PSI > 0.1 (moderate), > 0.2 (major)       |
| Prediction drift | PSI of predicted prices; review rate vs Phase 8 (20%)             | PSI > 0.2; review rate off by > 10 points |
| Outcomes         | real prices sent to `POST /feedback`: MdAPE, range coverage       | 80% range covers < 75% (after 30 prices)  |

Drift is only checked once 50 requests are logged; with fewer, PSI is mostly noise.

**PSI (population stability index)** compares the share of requests in each bin (the
training data's deciles, or each category) with training:
`PSI = Σ (live − train) × ln(live / train)`.

**Simulator.** `python -m src.monitor simulate` replays real listings through the same
`Predictor` the API uses, and sends back their real price as feedback. Use it to:

- check the whole pipeline end to end (`--fresh --n 300`);
- see a drift alert fire (`--market "Rio De Janeiro"`: a week of only one market);
- measure how much accuracy a host loses by filling in only the form (`--form-only`).

Replayed listings were in the training data, so their accuracy is optimistic. Real
monitoring uses new listings.

## Run

```powershell
git checkout -b phase-9
# save src\serve.py, src\monitor.py, web\index.html, tests\test_serve.py, Dockerfile,
# .dockerignore, docs\PHASE_9.md; append the config sections and requirements
pip install fastapi uvicorn python-multipart httpx pillow
pytest                                   # 108 tests
python -m src.serve build                # form options + monitoring reference
python -m src.serve run                  # open http://127.0.0.1:8000  (Ctrl+C to stop)
```

In a second terminal, while the service is running or not:

```powershell
python -m src.monitor simulate --n 300 --fresh               # healthy week
python -m src.monitor simulate --n 300 --fresh --form-only   # form fields only
python -m src.monitor simulate --n 200 --market "Rio De Janeiro"   # adds a shifted week
python -m src.monitor                                        # report only
```

## Docker (optional)

Needs Docker Desktop. Build it after `python -m src.serve build`, because the image copies
`models/`.

```powershell
docker build -t airbnb-price .
docker run -p 8000:8000 -v ${PWD}/data/monitoring:/app/data/monitoring airbnb-price
```

The image downloads CLIP at build time, so it works offline. The volume keeps the request log
on your disk, so the monitoring outlives the container.

## API

| Method | Path          | Body                                                  | Returns                                          |
| ------ | ------------- | ----------------------------------------------------- | ------------------------------------------------ |
| GET    | `/`           |                                                       | the web page                                     |
| GET    | `/form`       |                                                       | markets, neighbourhoods, typical values, options |
| GET    | `/health`     |                                                       | model served, text model / photo on or off       |
| POST   | `/predict`    | form-data: `listing` (JSON), optional `photo` (image) | price, range, review flag, reasons, id           |
| POST   | `/feedback`   | JSON `{"id": ..., "actual_price": ...}`               | saved                                            |
| GET    | `/monitoring` |                                                       | the monitoring summary                           |

Interactive API docs: http://127.0.0.1:8000/docs (FastAPI generates them).
