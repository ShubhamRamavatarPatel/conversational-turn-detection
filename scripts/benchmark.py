"""Benchmark in-process inference and HTTP endpoint, report CPU and request latency.

Usage: python -m scripts.benchmark --n 500 --url http://127.0.0.1:8000
"""
import argparse
import json
import platform
import time
from statistics import mean
import httpx
import numpy as np
from turn_detector.model import ROOT, load_artifact, probability


def summary(ms):
    ms=np.asarray(ms,dtype=float)
    return {'requests':len(ms),'p50_ms':round(float(np.percentile(ms,50)),3),
            'p95_ms':round(float(np.percentile(ms,95)),3),
            'p99_ms':round(float(np.percentile(ms,99)),3),
            'mean_ms':round(float(np.mean(ms)),3),
            'throughput_rps':round(float(1000/np.mean(ms)),2)}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--n',type=int,default=500)
    parser.add_argument('--url',default='http://127.0.0.1:8000')
    args=parser.parse_args()
    phrases=['can you check when my shipment will arrive',
             'I wanted to ask you about the', 'yes thank you that is all',
             'I need to change the delivery date to Friday']
    artifact=load_artifact()
    for _ in range(40): probability(phrases[0],artifact)
    direct=[]
    for i in range(args.n):
        start=time.perf_counter_ns()
        probability(phrases[i%len(phrases)],artifact)
        direct.append((time.perf_counter_ns()-start)/1e6)
    output={'machine':platform.platform(),'python':platform.python_version(),
            'in_process_single_sequential':summary(direct)}
    with httpx.Client(base_url=args.url,timeout=10.0) as client:
        for _ in range(20):
            r=client.post('/predict',json={'text':phrases[0], 'silence_ms':450});r.raise_for_status()
        http=[]
        start_batch=time.perf_counter()
        for i in range(args.n):
            start=time.perf_counter_ns()
            r=client.post('/predict',json={'text':phrases[i%len(phrases)],'silence_ms':450})
            r.raise_for_status()
            http.append((time.perf_counter_ns()-start)/1e6)
        batch_seconds=time.perf_counter()-start_batch
        output['http_sequential']=summary(http)
        output['http_sequential']['wallclock_rps']=round(args.n/batch_seconds,2)
    (ROOT/'artifacts'/'benchmark.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output,indent=2))


if __name__=='__main__':main()
