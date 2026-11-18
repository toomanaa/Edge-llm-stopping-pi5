#!/usr/bin/env python3
"""E1 generation runner. Talks to a local llama.cpp server, generates every
prompt to natural EOS, and logs everything the labeler needs.

Prereq (separate terminals):
  1) ./llama-server -m models/llama-3.2-1b-instruct-q4_k_m.gguf -c 2048 --port 8080
  2) python3 power_logger.py --out out/run_power.csv --hz 20
Then:
  python3 run_e1_generation.py --data data/commands.jsonl --out out/gen.jsonl

Output JSONL per prompt:
  id, prompt, text, tokens [{tok, logprob, entropy_topk, t}], t_start, t_end
Timestamps align with the power CSV, so joules per request (and per sentence)
are computed afterwards by integrating watts between t_start/t_end.
"""
import argparse, json, time, math, pathlib, urllib.request

SYSTEM = ("You are a smart home assistant. Respond to the user's request by "
          "outputting the tool call first on its own line, then any reply. "
          "Available tools: dim(room, percent), lights_on(room), "
          "lights_off(room), set_temp(celsius), lock(door), unlock(door), "
          "blinds_open(room), blinds_close(room), set_volume(level), "
          "music_play(), music_pause(), music_stop(), set_timer(minutes), "
          "fan(room, speed).")

def call_server(prompt, url, n_predict=200):
    body = json.dumps({
        "prompt": prompt, "n_predict": n_predict, "temperature": 0.0,
        "n_probs": 5, "stream": False, "cache_prompt": False,
    }).encode()
    req = urllib.request.Request(url + "/completion", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())

def entropy_from_probs(probs):
    """Top-k entropy in nats from llama.cpp's completion_probabilities entry."""
    ps = [p.get("prob", 0.0) for p in probs if p.get("prob", 0.0) > 0]
    return -sum(p * math.log(p) for p in ps) if ps else 0.0

def chat_prompt(utterance):
    # Llama-3.2 instruct chat template, minimal form
    return ("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{SYSTEM}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{utterance}<|eot_id|><|start_header_id|>assistant"
            "<|end_header_id|>\n\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    prompts = [json.loads(l) for l in open(args.data)]
    if args.limit:
        prompts = prompts[: args.limit]
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    done = set()
    outp = pathlib.Path(args.out)
    if outp.exists():  # resumable
        done = {json.loads(l)["id"] for l in open(outp)}
        print(f"resuming: {len(done)} already done")

    with open(outp, "a") as f:
        for i, ex in enumerate(prompts):
            if ex["id"] in done:
                continue
            t_start = time.time()
            resp = call_server(chat_prompt(ex["utterance"]), args.url)
            t_end = time.time()
            toks = []
            for tp in resp.get("completion_probabilities", []):
                toks.append({
                    "tok": tp.get("content", ""),
                    "logprob": next((c["prob"] for c in tp.get("probs", [])
                                     if c.get("tok_str") == tp.get("content")), None),
                    "entropy_topk": entropy_from_probs(tp.get("probs", [])),
                })
            rec = {"id": ex["id"], "utterance": ex["utterance"],
                   "ground_truth": ex["ground_truth"], "family": ex["family"],
                   "text": resp.get("content", ""),
                   "tokens": toks, "t_start": t_start, "t_end": t_end,
                   "n_tokens": resp.get("tokens_predicted")}
            f.write(json.dumps(rec) + "\n"); f.flush()
            if i % 10 == 0:
                dur = t_end - t_start
                print(f"[{i+1}/{len(prompts)}] {ex['id']} "
                      f"{rec['n_tokens']} tok in {dur:.1f}s")
    print("done")

if __name__ == "__main__":
    main()
