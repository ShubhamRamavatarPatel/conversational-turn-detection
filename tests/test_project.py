from fastapi.testclient import TestClient
import pytest
from scripts.prepare_data import build
from turn_detector.model import DEFAULT_MODEL, normalize
from turn_detector.policy import decide
from turn_detector.api import app


def test_data_group_leakage():
    splits=build()
    groups=[{e['group'] for e in splits[x]} for x in ('train','val','test')]
    assert not groups[0]&groups[1]
    assert not groups[1]&groups[2]
    assert not groups[0]&groups[2]


def test_normalization():
    assert normalize('Hello,  WORLD!') == 'hello world'


def test_no_early_endpoint_while_speaking():
    assert not decide(.999,10000,True).end_of_turn
    assert not decide(.99,50,False).end_of_turn


def test_policy_positive_negative():
    assert decide(.98,200,False).end_of_turn
    assert not decide(.2,700,False).end_of_turn
    assert decide(.2,1400,False).end_of_turn


def test_negative_silence_rejected():
    with pytest.raises(ValueError): decide(.4,-1)


def test_api():
    assert DEFAULT_MODEL.exists(), 'Run python -m scripts.train before tests'
    with TestClient(app) as client:
        r=client.get('/healthz')
        assert r.status_code == 200
        r=client.post('/predict', json={'text':'Can you help me?','silence_ms':1400})
        assert r.status_code==200
        assert r.json()['end_of_turn'] is True
        assert 0 <= r.json()['p_complete'] <= 1
        assert r.json()['model_version']
        r=client.post('/predict',json={'text':'', 'silence_ms':10000})
        assert r.json()['end_of_turn'] is False
        r=client.post('/predict',json={'text':'Hello','silence_ms':-1})
        assert r.status_code==422
        r=client.post('/predict',json={'text':'Hello','silence_ms':1000,'speech_active':True})
        assert r.json()['end_of_turn'] is False
