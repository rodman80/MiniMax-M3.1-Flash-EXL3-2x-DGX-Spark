#!/usr/bin/env python3
"""Phase 3 §5.4: mixed load. Decode streams running while a long prefill arrives.

The served lane chunks prefill at 16K tokens, so a 128K prompt is eight ~10 s chunk steps
that share the device with whatever decode streams are already running. A prefill win is
only useful if it does not make those streams stutter; a decode win is only useful if it
survives a prefill arriving.

Per decode stream it records inter-token latency in the window BEFORE the long request
lands and in the window AFTER, plus the long request's TTFT. Same prompts and parameters
for every cell; only the switches change.

Stdlib only, like the repo's own bench tools; it writes nothing to the server.
"""
import argparse
import json
import os
import random
import statistics
import sys
import threading
import time
import urllib.request
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "bench"))
from bench import rng_words  # the repo's own filler generator, so prompts match the benches

DEC_PROMPT = ("Write a long, detailed essay on how the invention of the printing press "
              "changed European science, religion and politics over the following three "
              "centuries.")


def stream_tokens(url, body, api_key, stop_at=None, timeout=1800):
    """Stream one chat request, returning per-delta timestamps and usage."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    body = dict(body)
    body["stream"] = True
    body.setdefault("stream_options", {"include_usage": True})
    req = urllib.request.Request(url, json.dumps(body).encode(), headers)
    stamps, usage, ttft = [], None, None
    started = time.time()
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
            for ch in ev.get("choices") or []:
                d = ch.get("delta") or {}
                if (d.get("content") or d.get("reasoning_content") or d.get("reasoning")):
                    now = time.time()
                    if ttft is None:
                        ttft = now - started
                    stamps.append(now)
                    if stop_at and now > stop_at:
                        return stamps, usage, ttft, started
    return stamps, usage, ttft, started


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="minimax-m3.1-flash")
    ap.add_argument("--api-key", default=os.environ.get("API_KEY", ""))
    ap.add_argument("--streams", type=int, default=4)
    ap.add_argument("--seconds", type=float, default=75)
    ap.add_argument("--prefill-words", type=int, default=98304,
                    help="words of filler; ~1.3 words/token on MiniMax, so 98304 ~= 128K tokens")
    ap.add_argument("--prefill-delay", type=float, default=12.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    url = a.base_url.rstrip("/") + "/v1/chat/completions"
    os.makedirs(a.out, exist_ok=True)
    t_start = time.time()
    split = t_start + a.prefill_delay + 1.0
    stop_at = t_start + a.seconds

    res = [{} for _ in range(a.streams)]

    def worker(i):
        try:
            stamps, usage, ttft, started = stream_tokens(
                url, {"model": a.model, "messages": [{"role": "user", "content": DEC_PROMPT}],
                      "max_tokens": 512, "temperature": 0}, a.api_key, stop_at)
            res[i].update(stamps=stamps, usage=usage, ttft=ttft, started=started)
        except Exception as exc:
            res[i]["error"] = str(exc)

    th = [threading.Thread(target=worker, args=(i,)) for i in range(a.streams)]
    for t in th:
        t.start()
    time.sleep(a.prefill_delay)
    nonce = uuid.uuid4().hex[:10]
    text = f"Nonce {nonce}. " + rng_words(random.Random(nonce), a.prefill_words) + "\n\nSay OK."
    pre = {}
    try:
        _, usage, ttft, started = stream_tokens(
            url, {"model": a.model, "messages": [{"role": "user", "content": text}],
                  "max_tokens": 1, "temperature": 0}, a.api_key)
        pre = {"ttft_s": round(ttft, 2) if ttft else None,
               "prompt_tokens": (usage or {}).get("prompt_tokens"), "started": started}
    except Exception as exc:
        pre = {"error": str(exc)}
    for t in th:
        t.join()

    def itl(seq):
        gaps = [seq[j + 1] - seq[j] for j in range(len(seq) - 1)]
        if not gaps:
            return {"n": 0, "median_ms": None, "p90_ms": None}
        return {"n": len(gaps), "median_ms": round(statistics.median(gaps) * 1000, 1),
                "p90_ms": round(sorted(gaps)[min(len(gaps) - 1, int(0.9 * len(gaps)))] * 1000, 1)}

    rows = []
    for i, r in enumerate(res):
        st = r.get("stamps") or []
        rows.append({"stream": i, "tokens": len(st), "ttft_s": round(r.get("ttft") or 0, 2),
                     "completion_tokens": (r.get("usage") or {}).get("completion_tokens"),
                     "before": itl([t for t in st if t < split]),
                     "after": itl([t for t in st if t >= split]),
                     "error": r.get("error")})
    before_med = [r["before"]["median_ms"] for r in rows if r["before"]["median_ms"]]
    after_med = [r["after"]["median_ms"] for r in rows if r["after"]["median_ms"]]
    out = {
        "streams": a.streams, "seconds": a.seconds, "prefill_words": a.prefill_words,
        "prefill": pre,
        "decode_tokens_total": sum(r["tokens"] for r in rows),
        "decode_agg_tok_s": round(sum(r["tokens"] for r in rows) / a.seconds, 2),
        "itl_before_median_ms": round(statistics.median(before_med), 1) if before_med else None,
        "itl_after_median_ms": round(statistics.median(after_med), 1) if after_med else None,
        "rows": rows,
    }
    with open(os.path.join(a.out, "results.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"long prefill: TTFT {pre.get('ttft_s')}s tokens {pre.get('prompt_tokens')}")
    print(f"decode: {out['decode_tokens_total']} tok over {a.seconds}s = "
          f"{out['decode_agg_tok_s']} tok/s  ITL before {out['itl_before_median_ms']} ms "
          f"after {out['itl_after_median_ms']} ms")
    for r in rows:
        print(f"  stream {r['stream']}: {r['tokens']} tok  before {r['before']}  after {r['after']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
