"""A simple CLI client demonstrating partial-to-complete transcript updates."""
import argparse
import httpx


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--url',default='http://127.0.0.1:8000')
    args=p.parse_args()
    stream=[('could you', 180,False),
            ('could you tell me when', 400,False),
            ('can you tell me when the delivery arrives', 180,False),
            ('can you tell me when the delivery arrives', 750,False)]
    with httpx.Client(base_url=args.url,timeout=5) as c:
        for phrase,pause,active in stream:
            r=c.post('/predict',json={'text':phrase,'silence_ms':pause,'speech_active':active})
            r.raise_for_status()
            data=r.json()
            print(f"{phrase!r:56} pause={pause:<4} p_complete={data['p_complete']:.3f} "
                  f"end={data['end_of_turn']} reason={data['reason']}")


if __name__=='__main__':main()
