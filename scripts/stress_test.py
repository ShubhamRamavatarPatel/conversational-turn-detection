"""Concurrent HTTP load test: sends request batches while limiting in-flight concurrency."""
import argparse
import asyncio
import json
import time
import numpy as np
import httpx
from pathlib import Path
from turn_detector.model import ROOT


async def main():
    arg=argparse.ArgumentParser()
    arg.add_argument('--url',default='http://127.0.0.1:8000')
    arg.add_argument('--n',type=int,default=300)
    arg.add_argument('--concurrency',type=int,default=10)
    args=arg.parse_args()
    sem=asyncio.Semaphore(args.concurrency)
    times=[];failed=0
    async with httpx.AsyncClient(base_url=args.url,timeout=15) as client:
        async def request(i):
            nonlocal failed
            async with sem:
                t=time.perf_counter()
                try:
                    resp=await client.post('/predict',json={'text':
                        'I want to change the delivery to next Tuesday' if i%2 else
                        'I want to change the delivery to', 'silence_ms':400})
                    resp.raise_for_status()
                except Exception: failed+=1
                finally:times.append((time.perf_counter()-t)*1000)
        start=time.perf_counter()
        await asyncio.gather(*(request(i) for i in range(args.n)))
        elapsed=time.perf_counter()-start
    result={'n':args.n,'concurrency':args.concurrency,'errors':failed,
            'wallclock_seconds':round(elapsed,3),'throughput_rps':round(args.n/elapsed,2),
            'p50_ms':round(float(np.percentile(times,50)),2),
            'p95_ms':round(float(np.percentile(times,95)),2),
            'p99_ms':round(float(np.percentile(times,99)),2)}
    (ROOT/'artifacts'/'stress_test.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':asyncio.run(main())
