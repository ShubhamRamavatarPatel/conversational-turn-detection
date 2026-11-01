# 7–9 minute video walkthrough (record this yourself)

**0:00–0:50 — Problem and architecture**
"I treated end-of-turn as more than silence detection. VAD tells me whether somebody is talking; ASR gives the words. The model estimates whether those words look like a complete turn; a small pause policy combines them so the assistant doesn't jump in too early."
Show the architecture block in README.

**0:50–2:15 — Data**
"I started with original support and logistics phrases because I didn't have consented real call recordings. Complete utterances get positive labels; manually incomplete phrases and generated prefixes get negative labels. I split by original utterance ID, so near-identical prefixes can't land in a different split. The labels are weak: even a prefix might be a valid stop depending on context."
Show `data/source_utterances.txt`, `data/explicit_partial.txt`, `scripts/prepare_data.py`.

**2:15–3:25 — Model**
"For a fast CPU baseline, I used word- and character-level TF-IDF, a few general end-word features, and logistic regression. The validation threshold penalizes premature completions more heavily than waiting. It's lightweight and explainable; I didn't spend time on hyperparameter search."
Show `turn_detector/model.py`, `scripts/train.py`.

**3:25–4:35 — Live inference demo**
Run `uvicorn turn_detector.api:app --port 8000` and in another terminal `python -m scripts.demo_client`. Point out that a partial phrase stays WAIT and a plausible full question after 750ms silence changes to END. Also show `/docs` and `/healthz`. Demonstrate that `speech_active=true` never ends.

**4:35–5:55 — Honest evaluation**
"Held-out accuracy is about 91.5%, but that number is from hand-authored data, not real ASR. On a deliberately harder 36-phrase challenge, accuracy drops to about 69.4%, and completion recall is under 40%. That tells me the model still hesitates on unfamiliar short phrases. The main next step is labeling real conversations, not adding more layers."
Show `artifacts/evaluation.json` and `artifacts/challenge_errors.json`.

**5:55–6:45 — Speed/serving**
"On this CPU/Linux machine, local API p95 was 3.19 milliseconds sequentially and 33.46 milliseconds under ten-way concurrent load. These measure only the API call; they don't include VAD silence, transcription, network or speech synthesis. I included a Dockerfile and tests, but did not claim a production soak test."
Show `artifacts/benchmark.json`, `artifacts/stress_test.json`, `Dockerfile`.

**6:45–8:15 — Production monitoring and future model**
"If this were handling the roughly one million monthly calls mentioned in the task, I'd track barging and false waits as product outcomes, not just offline F1. I would label consented/redacted production cases, watch performance by noise and language group, shadow deploy and A/B test pause thresholds. An audio-plus-text approach could use intonation, timing, ASR stability and dialogue history to decide more accurately."
Show `README.md` section 5 and short `docs/PROJECT_SUMMARY.md`.

**8:15–8:45 — Wrap up**
"I optimized for a transparent, deployable baseline, measurable errors, and a plan to improve it with real-world data. The weakest area is short and ambiguous turns; that is where I would focus next."

**Recording tips:** Run the commands before recording, show **your own** project edits/results, don't claim the trained-on-real-calls evaluation, keep the video under the stated 10-minute limit. Record with Loom/Zoom/OBS if available; this folder supplies the script, not a video recording.
