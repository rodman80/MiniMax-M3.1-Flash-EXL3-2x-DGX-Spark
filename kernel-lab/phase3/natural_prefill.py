#!/usr/bin/env python3
"""Phase 3: long prefill on NATURAL text, not the bench's random-word filler.

PERFORMANCE.md already flags this: "Prefill tuning used synthetic prompts. In a 16K chunk
of those prompts one expert receives about half the tokens. The tuning was not repeated on
natural-text routing." The same caveat applies to A3: the sparse-attention slice planner
scales with how *diverse* the indexer's block selections are, and a wall of random words
maximizes that diversity. A natural document is the case the server actually serves.

Builds a prompt of real prose from this repository's own markdown (README, PERFORMANCE,
docs/RECIPE_DECISIONS), shuffled by a seed and salted with a nonce so prefix caching
cannot apply, and measures TTFT and prompt-token throughput. Same text for every cell.
"""
import argparse
import glob
import json
import os
import random
import sys
import time
import urllib.request
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))


def natural_text(seed: int, target_tokens: int, ratio: float = 1.3) -> str:
    paras = []
    for f in ["README.md", "PERFORMANCE.md", "docs/RECIPE_DECISIONS.md", "docs/FIRST-RUN.md",
              "bench/README.md", "patches/README.md", "AGENTS.md", "NOTICE"]:
        p = os.path.join(ROOT, f)
        if not os.path.exists(p):
            continue
        for chunk in open(p).read().split("\n\n"):
            chunk = chunk.strip()
            if len(chunk) > 120:
                paras.append(chunk)
    rng = random.Random(seed)
    rng.shuffle(paras)
    need = max(1, int(target_tokens / ratio))
    out, total = [], 0
    i = 0
    while total < need:
        p = paras[(i * 7919) % len(paras)]        # a fixed but non-sequential walk
        out.append(p)
        total += len(p.split())
        i += 1
    return "Nonce " + uuid.uuid4().hex[:10] + ".\n\n" + "\n\n".join(out)


def count_tokens(url_base: str, text: str, model: str, api_key: str):
    """Ask the server to tokenize, so the prompt is sized against the real tokenizer."""
    url = url_base.rstrip("/") + "/tokenize"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(
        url, json.dumps({"model": model, "prompt": text, "max_tokens": 200000}).encode(),
        headers)
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.loads(r.read()).get("count")
    except Exception:
        return None


def stream_ttft(url, body, api_key, timeout=3600):
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    body = dict(body)
    body["stream"] = True
    body.setdefault("stream_options", {"include_usage": True})
    req = urllib.request.Request(url, json.dumps(body).encode(), headers)
    t0 = time.time()
    ttft, usage = None, None
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                ev = json.loads(payload)
            except ValueError:
                continue
            if ev.get("usage"):
                usage = ev["usage"]
            if ttft is None:
                for ch in ev.get("choices") or []:
                    d = ch.get("delta") or {}
                    if d.get("content") or d.get("reasoning_content") or d.get("reasoning"):
                        ttft = time.time() - t0
    return ttft, usage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="minimax-m3.1-flash")
    ap.add_argument("--api-key", default=os.environ.get("API_KEY", ""))
    ap.add_argument("--tokens", type=int, default=131072)
    ap.add_argument("--max-prompt-tokens", type=int, default=295000,
                    help="leave headroom under max_model_len for the chat template")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20261102)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    url = a.base_url.rstrip("/") + "/v1/chat/completions"
    rows = []
    for rep in range(a.reps):
        # a fresh document per rep: vLLM's automatic prefix caching would otherwise make
        # rep 2 measure the cache, not the prefill. Same seed sequence per cell, so every
        # cell sees the same documents.
        text = natural_text(a.seed + rep, a.tokens)
        n = count_tokens(a.base_url, text, a.model, a.api_key)
        # markdown and fenced code tokenize denser than prose: trim until the server
        # agrees the prompt fits, leaving headroom for the chat template and the one
        # generated token.
        while n and n > a.max_prompt_tokens:
            a.tokens = int(a.tokens * 0.92)
            text = natural_text(a.seed + rep, a.tokens)
            n = count_tokens(a.base_url, text, a.model, a.api_key)
        body = {"model": a.model, "messages": [{"role": "user", "content": text}],
                "max_tokens": 1, "temperature": 0}
        try:
            ttft, usage = stream_ttft(url, body, a.api_key)
            pt = (usage or {}).get("prompt_tokens")
            rows.append({"rep": rep, "prompt_tokens": pt, "precount_tokens": n,
                         "ttft_s": round(ttft, 2) if ttft else None,
                         "tok_s": round(pt / ttft, 1) if pt and ttft else None})
            print(f"  rep {rep}: {pt} tokens  TTFT {rows[-1]['ttft_s']}s  {rows[-1]['tok_s']} tok/s",
                  flush=True)
        except Exception as exc:
            rows.append({"rep": rep, "error": str(exc)})
            print(f"  rep {rep}: ERROR {exc}", flush=True)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "results.json"), "w") as fh:
        json.dump({"step": "natural_prefill", "target_tokens": a.tokens, "seed": a.seed,
                   "rows": rows}, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
