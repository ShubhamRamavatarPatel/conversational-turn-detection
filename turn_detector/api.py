"""HTTP scoring interface. Pass latest finalized ASR text plus upstream VAD state."""
from __future__ import annotations
from contextlib import asynccontextmanager
from pathlib import Path
import os

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from .model import DEFAULT_MODEL, load_artifact, normalize, probability
from .policy import decide


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = os.getenv('TURN_MODEL_PATH', str(DEFAULT_MODEL))
    app.state.artifact = load_artifact(str(Path(model_path).resolve()))
    yield


app = FastAPI(title='Conversational Turn Detector - Demonstration', version='0.1.0', lifespan=lifespan)


class PredictRequest(BaseModel):
    text: str = Field(..., max_length=4000, description='Latest transcript of user utterance; partial or final.')
    silence_ms: int = Field(0, ge=0, le=300000, description='Milliseconds since VAD last detected speech.')
    speech_active: bool = Field(False, description='Whether upstream VAD currently sees human speech.')


class PredictResponse(BaseModel):
    end_of_turn: bool
    p_complete: float
    reason: str
    recommended_pause_ms: int
    normalized_text: str
    model_version: str


@app.get('/healthz')
def healthz(request: Request):
    return {'status': 'ok', 'model_version': request.app.state.artifact['model_version']}


@app.post('/predict', response_model=PredictResponse)
def predict(request: PredictRequest, raw_request: Request):
    artifact = raw_request.app.state.artifact
    text = normalize(request.text)
    if not text:
        return PredictResponse(end_of_turn=False, p_complete=0., reason='empty_transcript',
                               recommended_pause_ms=0, normalized_text='', model_version=artifact['model_version'])
    p = probability(text, artifact)
    result = decide(p, request.silence_ms, request.speech_active, threshold=artifact['validation_threshold'])
    return PredictResponse(end_of_turn=result.end_of_turn, p_complete=round(p, 5),
                           reason=result.reason, recommended_pause_ms=result.recommended_pause_ms,
                           normalized_text=text, model_version=artifact['model_version'])
