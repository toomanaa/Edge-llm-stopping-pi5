#!/usr/bin/env python3
"""E1-QA generation runner. Deliberately NO few-shot format anchoring, so the
model answers naturally -- this is where verbosity (and waste) should show up,
unlike E0's command-parsing task.

Prereq (as with E0):
  1) ./llama.cpp/build/bin/llama-server -m models/<model>.gguf -c 2048 --port 8080
  2) python3 power_logger.py --out out/qa_power.csv --hz 20
Then:
  python3 run_qa_generation.py --data data/qa.jsonl --out out/qa_gen.jsonl
"""
import argparse, json, time, math, pathlib, urllib.request

# Deliberately open: a normal assistant system prompt, NOT anchored to a
# rigid output format. This is the fair test -- if the model over-answers
# here, that is the genuine phenomenon the controller targets.
SYSTEM = ("You are a helpful assistant. Answer the user's question clearly "
          "and correctly.")

def call_server(prompt, url, n_predict=200):
    body = json.dumps({"prompt": prompt, "n_predict": n_predict,
                       "temperature": 0.0, "n_probs": 5, "stream": False,
                       "cache_prompt": False}).encode()
    req = urllib.request.Request(url + "/completion", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())

def entropy_from_probs(probs):
    ps = [p.get("prob", 0.0) for p in probs if p.get("prob", 0.0) > 0]
    return -sum(p * math.log(p) for p in ps) if ps else 0.0

def chat_prompt(question):
    return ("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            + SYSTEM + "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            + question + "<|eot_id|><|start_header_id|>assistant"
            "<|end_header_id|>\n\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    prompts = [json.loads(l) for l in open(args.data)]
    if args.limit:
        prompts = prompts[:args.limit]
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    done = set()
    outp = pathlib.Path(args.out)
    if outp.exists():
        done = {json.loads(l)["id"] for l in open(outp)}
        print(f"resuming: {len(done)} already done")
    with open(outp, "a") as f:
        for i, ex in enumerate(prompts):
            if ex["id"] in done:
                continue
            t_start = time.time()
            resp = call_server(chat_prompt(ex["question"]), args.url)
            t_end = time.time()
            toks = []
            for tp in resp.get("completion_probabilities", []):
                toks.append({"tok": tp.get("content", ""),
                             "entropy_topk": entropy_from_probs(tp.get("probs", []))})
            rec = {"id": ex["id"], "question": ex["question"],
                   "answers": ex["answers"], "text": resp.get("content", ""),
                   "tokens": toks, "t_start": t_start, "t_end": t_end,
                   "n_tokens": resp.get("tokens_predicted")}
            f.write(json.dumps(rec) + "\n"); f.flush()
            if i % 10 == 0:
                print(f"[{i+1}/{len(prompts)}] {ex['id']} "
                      f"{rec['n_tokens']} tok in {t_end-t_start:.1f}s")
    print("done")

if __name__ == "__main__":
    main()
