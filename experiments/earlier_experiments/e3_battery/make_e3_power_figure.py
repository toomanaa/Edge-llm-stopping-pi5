"""Redraws fig_e3_power_trace.pdf with corrected segment times and means.
The plotted points were recovered from the published figure (e3_power_trace_points.csv);
segment times and means come from the three power-meter logs (check_e3_output.txt)."""
import csv, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 10, "font.family": "serif",
                     "axes.spines.top": False, "axes.spines.right": False})
segs = {}
for r in csv.DictReader(open("e3_power_trace_points.csv")):
    segs.setdefault(r["segment"], ([], []))
    segs[r["segment"]][0].append(float(r["elapsed_h"]))
    segs[r["segment"]][1].append(float(r["power_W"]))
info = {"1": ("Segment 1 (07:12-08:39), mean 7.29 W", "#8fa8c4"),
        "2": ("Segment 2 (10:54-12:53), mean 7.30 W", "#c46b4f"),
        "3": ("Segment 3 (13:45-15:02), mean 7.47 W", "#6ba368")}
fig, ax = plt.subplots(figsize=(7, 3.6))
for k, (h, w) in segs.items():
    ax.plot(h, w, color=info[k][1], linewidth=0.8, label=info[k][0])
ax.set_xlim(-0.39, 8.22); ax.set_ylim(0, 12)
ax.set_xlabel("Elapsed time (hours)"); ax.set_ylabel("Power draw (W)")
ax.set_title("Measured power draw across the battery drain\n"
             "(three logging segments; gaps are logger stops, not power loss)")
ax.legend(fontsize=8, loc="upper right", frameon=True)
fig.tight_layout()
fig.savefig("fig_e3_power_trace.pdf")
fig.savefig("fig_e3_power_trace.png", dpi=150)
