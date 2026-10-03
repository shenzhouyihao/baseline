import json
import os
import time
from pathlib import Path

from openai import OpenAI
from utils.get_result import get_result

ROOT = Path('/home/lzm/work/MemCast')
PROMPT = ROOT / 'Memory/predictions/ETTh/prompt/ETTh1/prompt_OT_ETTh1_96_720_547_cosine.txt'
OUT = ROOT / 'results/memcast_720_model_probe.jsonl'
MODELS = [
    'claude-fable-5',
    'claude-fable-5-1',
    'claude-haiku-4-5',
    'claude-haiku-4-5-20251001',
    'claude-opus-4-5',
    'claude-opus-4-5-20251101',
    'claude-opus-4-6',
    'claude-opus-4-7',
    'claude-opus-4-8',
    'claude-opus-5',
    'claude-opus-5-5',
    'claude-sonnet-4-5',
    'claude-sonnet-4-5-20250929',
    'claude-sonnet-4-6',
    'claude-sonnet-5',
    'codex-auto-review',
    'gpt-5.3-codex-spark',
    'gpt-5.5',
    'gpt-5.6-sol',
    'gpt-5.6-terra',
    'gpt-6-astra',
    'gpt-6-sol',
]

prompt = PROMPT.read_text()
client = OpenAI(
    base_url=os.environ['OPENAI_BASE_URL'],
    api_key=os.environ['OPENAI_API_KEY'],
    timeout=125,
    max_retries=0,
)

for model in MODELS:
    started = time.monotonic()
    row = {'model': model, 'prompt_chars': len(prompt)}
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': prompt},
            ],
            temperature=0.6,
            top_p=0.7,
            max_tokens=16384,
        )
        answer = response.choices[0].message.content or ''
        prediction = get_result(answer)
        row.update({
            'status': 'ok',
            'elapsed_seconds': round(time.monotonic() - started, 3),
            'response_chars': len(answer),
            'parsed_points': len(prediction),
            'complete_720': len(prediction) >= 720,
        })
    except Exception as exc:
        row.update({
            'status': 'error',
            'elapsed_seconds': round(time.monotonic() - started, 3),
            'error': str(exc)[:2000],
        })
    print(json.dumps(row), flush=True)
    with OUT.open('a') as f:
        f.write(json.dumps(row) + '\n')

