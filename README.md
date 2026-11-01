# Conversational EoT — Text-based End-of-Turn Detector

**Conversational AI prototype:** trained classifier + practical silence-aware turn-taking policy + FastAPI + Docker + offline evaluation + repeatable latency/stress tests + monitoring plan.

> **Important:** All reported accuracy metrics come from a small, hand-authored and weakly labeled English dataset. They do **not** establish production ASR accuracy. This is a runnable demonstration to discuss engineering tradeoffs, not a trained-on-real-calls system.

## 1. Problem and design

In a voice agent, **VAD** reports whether acoustic speech is present; **ASR/STT** provides evolving text; the **turn detector** predicts whether the phrase is linguistically complete. We combine the latter with the elapsed VAD silence to decide when to send the transcript to the LLM.

```text
WebRTC inbound audio
       |
       v
   VAD + STT --------> Partial/final transcript
       |                          |
       |                          v
       |                Word + character n-gram TF-IDF
       |                + 9 simple linguistic end features
       |                          |
       |                     Logistic regression
       |                          |
       +-- active / silence_ms -->+--> Dynamic pause policy
                                             |
                                    WAIT / END_OF_TURN
                                             |
                                             v
                                            LLM --> TTS
```

**Why text first?** This minimizes requirements and cost: roughly a 1 ms CPU classifier evaluation, no GPU, transparent lexical diagnostics, and easy deployment. Text-only cannot capture pitch, breaths, rapid self-correction, ASR delay/errors, overlapping voices, or contextual intent. A future production version should fuse acoustic features, ASR stability, timing and conversational context.

**Output:** `p_complete` is a *linguistic completion score*, **not** "the speaker has stopped." If VAD says speech is active, we always wait. For silent spans, higher confidence enables a shorter pause: at `threshold+0.23` wait 160 ms; at the validation-tuned `threshold` wait 300 ms; above `threshold-0.20` wait 650 ms; otherwise a 1300 ms fallback. Times are illustrative engineering policy parameters, **not** optimized against real speech or calibrated uncertainty. (Latency target below concerns service processing, not intentional VAD pause.)

## 2. Quick start

Python 3.11+ suggested. From **the unzipped project root**:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m scripts.train                 # optional: reproduce packaged model
python -m pytest -q
uvicorn turn_detector.api:app --host 0.0.0.0 --port 8000
```

In a second terminal:

```bash
curl -s http://localhost:8000/healthz
curl -s http://localhost:8000/predict -H 'Content-Type: application/json' \
  -d '{"text":"Can you send the invoice today","silence_ms":700,"speech_active":false}'
python -m scripts.demo_client
python -m scripts.benchmark --n 500
python -m scripts.stress_test --n 500 --concurrency 10
```

Or containerize the **pretrained artifact** (training is not part of the runtime container):

```bash
docker build -t conversational-turn-detection .
docker run --rm -p 8000:8000 conversational-turn-detection
```

Interactive API schema: **http://localhost:8000/docs**. Only load `joblib` artifacts you produced or trust; untrusted pickle/joblib objects are unsafe.

### Request / response example

```json
POST /predict
{"text":"Yes, please go ahead", "silence_ms":400, "speech_active":false}
```

```json
{
  "end_of_turn": true,
  "p_complete": 0.8,
  "reason": "linguistic_endpoint",
  "recommended_pause_ms": 300,
  "normalized_text": "yes please go ahead",
  "model_version": "charword-logreg-v1"
}
```

*The JSON response above illustrates the format; score and end decision should be taken from a live request, not assumed to be exactly 0.8.*

The endpoint is intentionally stateless: caller supplies the current **entire** utterance transcript and the VAD silence elapsed, not a single last token or raw audio. In a real streaming service, maintain utterance state by connection/session ID, debounce STT partial revisions, flush state at end-of-turn, and avoid sending full transcripts/PII to ordinary metrics logs.

## 3. Data and reproducibility

- `data/source_utterances.txt`: 230+ original English support/logistics, booking, billing, and software-support *complete turns*; includes short confirmations. Human-written for this project, **not real calls**.
- `data/explicit_partial.txt`: 70+ deliberately incomplete/continuing expressions.
- `scripts/prepare_data.py`: makes additional *weak negatives* by taking the first ~40% and ~70% of long complete turns. Some prefixes may in reality be acceptable utterance endings; weak-label noise is an explicit limitation.
- Split reproducibly at the **source-utterance ID** level: train 70%, validation 15%, held-out test 15%. Prefixes never cross splits from their parent utterance.
- `data/challenge.jsonl`: a **separate 36-example, hand-labeled challenge set**, not used in training or threshold selection (short utterances, contextual ambiguity, dependent clauses). Small and manually constructed.
- Preprocess by lowercasing and stripping punctuation to simulate punctuation-free ASR; unknown languages and real ASR errors are not represented.

**Model:** `TfidfVectorizer` word 1–2 grams + character 2–5 grams + lightweight linguistic-ending features -> class-weighted logistic regression. Validation selects a completion threshold from 0.25 to 0.90 with a 1.5× cost for a false "turn complete" prediction. We did **not** tune on test/challenge samples.

Generate splits/retrain/re-evaluate with `python -m scripts.train`; the trained artifact is at `artifacts/turn_model.joblib`. The script overwrites reports and weights.

## 4. Results from the packaged model

| Split | Examples | Accuracy | Completion precision | Completion recall | False-early rate |
|---|---:|---:|---:|---:|---:|
| Validation | 92 | 91.3% | 91.2% | 86.1% | 5.4% |
| Held-out source groups | 94 | **91.5%** | 86.5% | 91.4% | 8.5% |
| Unseen, hand-authored challenge | 36 | **69.4%** | 100.0% | 38.9% | 0.0% |

The validation-selected threshold is **0.63**. The held-out confusion matrix is TN=54, FP=5, FN=3, TP=32. The challenge recall is poor (7/18 complete turns identified): its short, out-of-pattern phrases often score too low. No audio-based ground truth or call-level interruption outcomes are available; thus the 8.5% *text-label* false-early rate is **not** a measured user interruption rate. See `artifacts/evaluation.json` and `artifacts/challenge_errors.json`.

The challenge score is the clearest sign this should not be deployed without diverse, realistic conversation data. The long-pause policy can eventually handle uncertain completed turns, at the cost of 650–1300 ms delays.

### Local latency and load measurements

Measured here on **Linux, Python 3.13.5, loopback HTTP**, one Uvicorn worker. Measurements will differ across hardware, Python/container builds and traffic mixes:

| Test | Traffic | p50 | p95 | p99 | Throughput |
|---|---|---:|---:|---:|---:|
| In-process sequential model | 350 evaluations | 0.870 ms | 1.209 ms | 1.875 ms | ~1069 req/s |
| Sequential POST /predict | 350 requests | 2.179 ms | **3.193 ms** | 4.058 ms | ~430 req/s |
| Concurrent stress | 250 requests, concurrency 10 | 18.87 ms | **33.46 ms** | 36.82 ms | ~494 req/s, 0 HTTP errors |

These results meet the **under-100 ms per inference request** target on this local setup, including concurrency 10. They do **not** include ASR, VAD, network hops, LLM, TTS, and the policy's intentional end-of-speech wait. Real end-to-end turn latency is greater. Files: `artifacts/benchmark.json`, `artifacts/stress_test.json`.

### Verification

`python -m pytest -q` — **6 passed** locally, covering group-split leakage, normalization, speaking guard, policy timeouts, validation and HTTP input/output. `scripts/demo_client.py` simulates a four-step transcript stream. Docker configuration is supplied; a real Docker image/container test is not claimed here.

## 5. Production monitoring plan (~1 million calls/month context)

**Core product metrics (by hour/day and slice):** interruption/agent-barge-in rate (human resumes during/just after TTS), assistant response gap after genuinely completed speech (p50/p95/p99), false waiting/stall percentage, caller reprompts ("hello?"), conversation abandonment, escalation, customer feedback and retention. Use *human review* on opt-in/redacted samples because absence of barging does not guarantee good turn-taking.

**ML/data quality:** completion score and requested pause-time distributions, ASR partial-vs-final revisions, token count, VAD speech fraction, silence-gap histogram, share hitting 1300 ms fallback, estimated disagreements with human adjudication, drift by language/accent/channel/noise/call-intent (only where safe and lawful). Measure model across genuine holdouts by **conversation/customer/time** to avoid leakage.

**Platform:** /predict p50/p95/p99, CPU/RAM, QPS/bursts, model version, model load failures, queueing/timeouts, error codes; trace VAD→ASR→classifier→LLM handoff timings. Build alerts on sustained high interruption rate, excessive 1300 ms fallbacks, growing response lag, or inference p95 exceeding budget. One million calls/month is ~0.4 calls/sec **average**, but sizing must use concurrent audio frames/real peak request rates, not this average.

**Improvement loop:** obtain consent where needed, securely redact PII, limit access/retention; sample rare and bad cases, human-label actual endings with adjacent audio and conversation context, create time-separated and customer-separated train/validation/test sets, retrain/tune thresholds, shadow-test, A/B rollout and automatic rollback. Store anonymized aggregates by default; do not persist full transcripts in model-service access logs. Offline metrics are insufficient; measure real speech interruptions.

## 6. Discussion / choices

- **Why not VAD alone?** Silence can mean thinking, a breath or an unfinished sentence. Lexical completion adds semantics, but VAD remains necessary to prevent talking over speech.
- **Text versus audio?** Text is cheap, interpretable, and works with STT output. Audio may distinguish final intonation and hesitations, at the cost of data labeling, compute, timing, and noise robustness. A fused model using text, prosody, VAD, ASR stability, pause duration and dialogue context is likely stronger.
- **ASR as detector?** Fine-tune a transcription encoder/decoder (Whisper/Parakeet-style) with an auxiliary end-of-turn head; train against annotated natural speaker-turn boundaries, not just end-of-utterance ASR segmentation. Distinguish stable ASR finals from true pragmatic turn completion.
- **Architecture:** VAD and ASR consume streaming audio; provisional text scores update as words arrive; gate with speech activity; emit END exactly once (idempotent session event), forward stable transcript to LLM, start TTS, allow barge-in cancellation; audit timing of handoff.
- **Limits:** curated domain, no multilingual data, no real audio or timestamps, no prosody/interruptions, no background-noise testing, noisy prefix labels, brittle to atypical/short phrases, thresholds hand-picked from simple heuristics, and untested service under realistic cloud conditions.

## 7. Repository contents

```text
turn_detector/model.py       # TF-IDF + lexical end features + trained-model loader
turn_detector/policy.py      # VAD/pause-aware turn-taking decision
turn_detector/api.py         # Pydantic FastAPI /predict and /healthz
scripts/prepare_data.py     # Group-aware reproducible dataset splits
scripts/train.py            # Train, tune validation threshold, held-out evaluation
scripts/demo_client.py      # Simulated incremental transcriptions
scripts/benchmark.py        # Warmed in-process and serial HTTP p50/95/99
scripts/stress_test.py      # Async bounded-concurrency HTTP load test
data/                       # Human-authored seeds, splits and unseen challenge
artifacts/                  # Trained joblib model, scores, errors and benchmark logs
tests/                      # Model, policy, validation and endpoint smoke tests
docs/                       # Markdown project summary and video walkthrough script
Dockerfile                  # CPU inference container with non-root user/healthcheck
```

Optional: generate a PDF case study after training with `pip install -r requirements-docs.txt` and `python -m scripts.make_summary_pdf`.


## Contributing

Create a virtual environment, install the requirements, and run `python -m pytest -q` before submitting a pull request. Keep related source utterances in the same data split and document any changes to evaluation results. Do not commit credentials or private conversation data.

## License

No open-source license has been selected. Contact the repository owner for permission to reuse or redistribute this project.
