#!/usr/bin/env python3
"""Compare the hidden states the single-pass tool read during generation with
what the llama-server /embeddings endpoint returns for the same text (the old
pipeline). Run with a llama-server (started with --embeddings) on port 8080.

  python3 verify_hidden.py --dump dump.jsonl --data qa_v2_pilot.jsonl --probe probe_weights.npz
"""
import argparse, json, sys, urllib.request
import numpy as np
sys.path.insert(0, "controller")
from probe_inference import Probe

SYSTEM = "You are a helpful assistant. Answer the user's question clearly and correctly."
def chat_prompt(q):
    return ("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n" + SYSTEM +
            "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n" + q +
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")
def server_embedding(text, url):
    body = json.dumps({"content": text, "pooling": "none", "cache_prompt": True}).encode()
    req = urllib.request.Request(url + "/embeddings", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return np.array(json.loads(r.read())[0]["embedding"][-1], dtype=np.float64)

ap = argparse.ArgumentParser()
ap.add_argument("--dump", required=True); ap.add_argument("--data", required=True)
ap.add_argument("--probe", default="probe_weights.npz"); ap.add_argument("--url", default="http://127.0.0.1:8080")
ap.add_argument("--n", type=int, default=60)
a = ap.parse_args()
q = {}
for l in open(a.data):
    j = json.loads(l); q[j["id"]] = j["question"]
probe = Probe(a.probe)
rows = [json.loads(l) for l in open(a.dump)][: a.n]
cos, dp, al = [], [], []
for r in rows:
    h = np.array(r["emb"], dtype=np.float64)
    s = server_embedding(chat_prompt(q[r["id"]]) + r["prefix"], a.url)
    cos.append(float(h @ s / (np.linalg.norm(h) * np.linalg.norm(s))))
    dp.append(abs(probe.predict_proba(h) - probe.predict_proba(s)))
    al.append(r["aligned"])
cos, dp, al = np.array(cos), np.array(dp), np.array(al)
print(f"checked {len(rows)} boundaries ({al.sum()} read directly, {(~al).sum()} via KV rollback)")
print(f"cosine similarity: min {cos.min():.5f}  median {np.median(cos):.5f}")
print(f"|probe(single-pass) - probe(server)|: max {dp.max():.5f}  median {np.median(dp):.5f}")
print("PASS" if cos.min() > 0.99 and dp.max() < 0.05 else "CHECK: states differ more than expected")
