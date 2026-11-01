"""Produce a submission-ready, non-exaggerated case study PDF from local reports."""
import json
from pathlib import Path
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.colors import HexColor, white
from reportlab.lib.units import inch
from turn_detector.model import ROOT

NAVY=HexColor('#0F172A'); BLUE=HexColor('#1D4ED8'); GRAY=HexColor('#475569'); LIGHT=HexColor('#F1F5F9'); LINE=HexColor('#CBD5E1')


def pdf():
    evals=json.loads((ROOT/'artifacts/evaluation.json').read_text())
    bench=json.loads((ROOT/'artifacts/benchmark.json').read_text())
    stress=json.loads((ROOT/'artifacts/stress_test.json').read_text())
    out=ROOT/'docs'/'End_of_Turn_Case_Study.pdf'
    doc=SimpleDocTemplate(str(out),pagesize=(612,792),rightMargin=50,leftMargin=50,topMargin=35,bottomMargin=34,title='Conversational: End-of-Turn Detection Case Study',author='Project contributors')
    ss=getSampleStyleSheet()
    ss.add(ParagraphStyle(name='TitleCustom',fontName='Helvetica-Bold',fontSize=18.5,leading=21.5,textColor=NAVY,spaceAfter=5))
    ss.add(ParagraphStyle(name='SubtitleCustom',fontName='Helvetica',fontSize=9.6,leading=12.5,textColor=GRAY,spaceAfter=10))
    ss.add(ParagraphStyle(name='HCustom',fontName='Helvetica-Bold',fontSize=10.4,leading=13,textColor=NAVY,spaceBefore=7,spaceAfter=4,keepWithNext=True))
    ss.add(ParagraphStyle(name='BodyCustom',fontName='Helvetica',fontSize=8.3,leading=11.5,textColor=NAVY,spaceAfter=4))
    ss.add(ParagraphStyle(name='SmallCustom',fontName='Helvetica',fontSize=7.25,leading=9.5,textColor=GRAY,spaceAfter=4))
    ss.add(ParagraphStyle(name='Metric',fontName='Helvetica-Bold',fontSize=12,leading=16,textColor=BLUE,alignment=TA_CENTER))
    ss.add(ParagraphStyle(name='MetricLabel',fontName='Helvetica',fontSize=7,leading=11,textColor=GRAY,alignment=TA_CENTER))
    def p(text,style='BodyCustom'):return Paragraph(text,ss[style])
    story=[p('End-of-Turn Detection', 'TitleCustom'),p('Conversational | AI/ML Engineering Case Study | English text baseline + silence-aware serving','SubtitleCustom')]
    story += [p('<b>Goal.</b> Help voice agents start speaking promptly when the caller is finished, without interrupting someone who has only paused. Designed as a transparent, fast CPU baseline: <b>not production-validated</b>.')]
    story.append(p('Architecture and rationale','HCustom'))
    arch=Table([[p('<b>Streaming audio</b><br/>VAD + ASR','SmallCustom'),p('&rarr;','BodyCustom'),p('<b>Completion model</b><br/>TF-IDF + lexical cues','SmallCustom'),p('&rarr;','BodyCustom'),p('<b>Pause policy</b><br/>WAIT / TURN END','SmallCustom'),p('&rarr;','BodyCustom'),p('<b>LLM + TTS</b>','SmallCustom')]],colWidths=[92,21,133,21,110,21,114])
    arch.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),LIGHT),('BOX',(0,0),(-1,-1),.5,LINE),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
    story.append(arch)
    story.append(Spacer(1,7))
    story.append(p('Text scoring uses word 1-2grams, character 2-5grams and nine lightweight structural features with class-weighted logistic regression. It accepts partial or final ASR text; VAD activity always blocks end-of-turn. Pause gates are 160/300/650/1300 ms depending on linguistic confidence. These pauses are separate from model-request latency.'))
    story.append(p('Data and evaluation','HCustom'))
    story.append(p('Original hand-authored support/logistics/scheduling examples, deliberately incomplete phrases and weak negative prefixes. Data is split by original utterance ID, not by generated snippet. A separate 36-item challenge set tests unfamiliar short/ambiguous expressions. Validation selected threshold <b>0.63</b>, weighting premature endpoints 1.5x.'))
    names=[('Validation','validation'),('Held-out test','heldout_test'),('Hard challenge','hand_authored_unseen_challenge')]
    tbl=[[p('<b>Split</b>','SmallCustom'),p('<b>N</b>','SmallCustom'),p('<b>Accuracy</b>','SmallCustom'),p('<b>Recall</b>','SmallCustom'),p('<b>False-early</b>','SmallCustom')]]
    for display,key in names:
        r=evals[key]
        tbl.append([p(display,'SmallCustom'),str(r['n']),f"{r['accuracy']:.1%}",f"{r['recall_complete']:.1%}",f"{r['false_early_rate']:.1%}"])
    t=Table(tbl,colWidths=[164,50,99,99,100],hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),LIGHT),('LINEBELOW',(0,0),(-1,0),.6,LINE),('LINEBELOW',(0,1),(-1,-1),.3,LINE),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5),('FONTSIZE',(1,1),(-1,-1),9),('TEXTCOLOR',(1,1),(-1,-1),NAVY),('VALIGN',(0,0),(-1,-1),'MIDDLE')]))
    story.append(t)
    story.append(Spacer(1,7))
    story.append(p('<b>Candid result:</b> 91.5% accuracy on held-out hand-authored text is <b>not</b> evidence of true speech turn accuracy. Hard-challenge recall fell to 38.9%, exposing missed completed short turns. Label noise, ASR errors, intonation and call context remain unaddressed.','SmallCustom'))
    story.append(p('Serving and local performance','HCustom'))
    story.append(p('FastAPI <b>POST /predict</b> accepts text, elapsed VAD silence and speaking status; <b>GET /healthz</b> checks model load. The submission includes Dockerfile, trained joblib weights, client example, group-leakage/test assertions, and automated benchmark/stress scripts.'))
    b=bench['http_sequential']; s=stress
    perf=Table([[p(f"{b['p95_ms']:.2f} ms",'Metric'),p(f"{s['p95_ms']:.2f} ms",'Metric'),p('6/6','Metric')],
                [p('Sequential API p95 (350 local POSTs)','MetricLabel'),p('Concurrent p95 (250 POSTs, c=10)','MetricLabel'),p('Local automated tests passed','MetricLabel')]],colWidths=[170,170,172])
    perf.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),LIGHT),('BOX',(0,0),(-1,-1),.5,LINE),('TOPPADDING',(0,0),(-1,0),6),('BOTTOMPADDING',(0,1),(-1,1),5)]))
    story.append(perf)
    story.append(p('Measured on local Linux and loopback HTTP (one Uvicorn worker). These p95 values are under the requested &lt;100 ms <b>model/API processing</b> target; they exclude speech collection, ASR, VAD silence waiting, LLM and TTS. Dockerfile was not run with a Docker daemon in this environment.','SmallCustom'))
    story.append(p('Monitoring and continuous improvement','HCustom'))
    story.append(p('<b>User outcome:</b> agent barge-in/interruption, time from true completion to reply, repeated caller prompting, excessive 1300 ms fallbacks, abandonments and human-rated friction. <b>ML/data:</b> text drift, ASR revisions, score distribution, carefully consented/redacted audio review, stratification by context/noise/accent/language where permitted. <b>Platform:</b> API p95/p99, rate peaks, errors, CPU/memory and inference-to-LLM traces.'))
    story.append(p('At about 1M calls/month, size capacity for bursty *partial transcript scoring*, not average call counts. Label real, privacy-safe end boundaries; evaluate by conversation/customer/time; shadow then A/B test a new model/pause policy; monitor and retain rollback.'))
    story.append(p('Next iteration','HCustom'))
    story.append(p('Collect genuine turn-end labels and audio/prosody; compare text-only, VAD-only, and multimodal text+audio+ASR-stability baselines. Add conversation-state features, robust background-noise/no-speech tests, threshold calibration and formal interruption-versus-latency A/B metrics.'))
    story.append(p('Reproduce: <b>pip install -r requirements.txt</b> | <b>python -m scripts.train</b> | <b>python -m pytest -q</b> | <b>uvicorn turn_detector.api:app --port 8000</b>. See README for benchmark, stress, Docker and video walkthrough commands.','SmallCustom'))
    def footer(canvas,doc):
        canvas.setStrokeColor(LINE);canvas.line(50,42,562,42)
        canvas.setFont('Helvetica',7.5);canvas.setFillColor(GRAY)
        canvas.drawString(50,30,'Conversational | End-of-Turn Detector | Prototype case study')
        canvas.drawRightString(562,30,f'{doc.page}')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    print(out)


if __name__=='__main__':pdf()
