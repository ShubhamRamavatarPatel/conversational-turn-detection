# Conversational Project Overview: End-of-Turn Detection

## The problem
A voice agent needs to know when a user is done speaking so it does not interrupt them or wait too long. I built an English **text-based linguistic completion classifier**, then combined it with upstream **VAD silence** in a simple streaming decision policy.

## Approach
- **Data:** hand-authored logistics/support/billing/scheduling utterances, plus realistic incomplete fragments and automatically truncated prefixes. Split by *source utterance*, so related examples cannot cross train/validation/test. All labels are synthetic/weak, not from production calls.
- **Model:** TF-IDF word 1–2grams, character 2–5grams and small interpretable lexical end features; class-weighted logistic regression, CPU inference; threshold selected using validation with 1.5× cost for premature completion.
- **Serving:** stateless FastAPI `/predict` accepting latest transcript, `silence_ms`, `speech_active`. VAD speech always overrides the model. Silence gate chooses 160/300/650/1300 ms depending on model confidence. Includes Dockerfile, model artifact, tests, demo client and concurrent HTTP load test.

## Offline quality
| Set | N | Accuracy | Completion recall | False-early rate |
|---|---:|---:|---:|---:|
| Validation | 92 | 91.3% | 86.1% | 5.4% |
| Held-out groups | 94 | **91.5%** | 91.4% | 8.5% |
| Hard challenge | 36 | **69.4%** | 38.9% | 0% |

**Important caveat:** This is not a real-world ASR or interruption-rate evaluation. The hard set exposes weak generalization to short/context-dependent responses. Some truncated-prefix labels may be ambiguous. These are discussion points, not results to hide.

## Serving and performance
Local Linux, loopback HTTP and one worker, warmed requests: sequential API **p95 3.19 ms** (350 requests), concurrent stress test at 10 in flight **p95 33.46 ms** (250 requests; zero HTTP errors). Six local automated tests passed. This meets the *model API processing* budget of 100 ms in this test environment; VAD waiting, ASR, LLM and TTS are excluded. Dockerfile supplied but not verified via an actual Docker daemon.

## Monitoring at scale
Measure actual **interruptions/barge-ins**, median and tail **time-to-first-response after completion**, long-pause fallbacks, reprompts, customer outcomes; plus ASR revisions, data drift, language/accent/noise slices where consent permits, model score distribution, API p95/p99, exceptions and CPU. At ~1M calls/month, size for concurrent STT updates and burst peaks. Label consented/redacted real user turns, separate by account and time, shadow deploy, A/B compare, and maintain fast rollback. Avoid logging raw PII.

## What I would improve
Use real speech and holdout conversations; human-label true turn endings (rather than weak truncation labels); add audio/prosody, streaming ASR stability, silence history and dialogue context; calibrate and tune pause thresholds for the business tradeoff between interruptions and response delay; validate multilingual performance separately.

**Run:** `pip install -r requirements.txt` → `python -m pytest -q` → `uvicorn turn_detector.api:app --port 8000` → `python -m scripts.demo_client`. Full reproducible training, evaluation, bench and Docker commands are in `README.md`.
