#!/usr/bin/env python3
"""Export probe_weights.npz to a plain text file the C++ tool can read.
Usage: python3 export_probe_txt.py probe_weights.npz probe.txt"""
import sys, numpy as np
d = np.load(sys.argv[1])
assert str(d["activation"]) == "relu" and str(d["out_activation"]) == "logistic" and int(d["n_layers"]) == 2
mean, scale, W0, b0, W1, b1 = (d[k].astype(np.float64) for k in ["scaler_mean", "scaler_scale", "W0", "b0", "W1", "b1"])
with open(sys.argv[2], "w") as f:
    f.write(f"{W0.shape[0]} {W0.shape[1]}\n")
    for a in (mean, scale, W0.reshape(-1), b0, W1.reshape(-1), b1):
        f.write(" ".join(repr(float(x)) for x in a) + "\n")
print("wrote", sys.argv[2], "input dim", W0.shape[0], "hidden", W0.shape[1])
