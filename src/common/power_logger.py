#!/usr/bin/env python3
"""Power logger for Raspberry Pi 5 using the on-board PMIC. No hardware needed.

Polls `vcgencmd pmic_read_adc`, which reports voltage (V) and current (A) for
each supply rail. Total board power = sum over rails of V*A. Writes CSV:
    timestamp_s, total_watts
Run in a second terminal (or background) during generation:
    python3 power_logger.py --out run1_power.csv --hz 20
Stop with Ctrl-C. Joules for any interval = integral of watts dt, computed
later by label/analysis scripts using the timestamps in the generation log.

Notes:
- Poll rate ~20 Hz is reliable; the vcgencmd call itself takes ~10-20 ms.
- The PMIC covers the board rails (SoC, DRAM, IO), not USB peripherals.
  Keep the run headless (no monitor) and identical across policies so that
  differences are attributable to decoding. Validate later with an inline
  USB-C meter; report method in the paper either way.
"""
import argparse, subprocess, time, sys, re

PAIR = re.compile(r"(\w+)_[VA] (?:volt|current)\((\d+)\)=([\d.]+)")

def read_power():
    out = subprocess.run(["vcgencmd", "pmic_read_adc"],
                         capture_output=True, text=True, timeout=1).stdout
    volts, amps = {}, {}
    for line in out.splitlines():
        line = line.strip()
        m = re.match(r"\s*(\w+)_V volt\(\d+\)=([\d.]+)V", line)
        if m:
            volts[m.group(1)] = float(m.group(2)); continue
        m = re.match(r"\s*(\w+)_A current\(\d+\)=([\d.]+)A", line)
        if m:
            amps[m.group(1)] = float(m.group(2))
    return sum(volts[r] * amps[r] for r in volts if r in amps)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--hz", type=float, default=20.0)
    args = ap.parse_args()
    period = 1.0 / args.hz
    with open(args.out, "w") as f:
        f.write("t,watts\n")
        print(f"logging to {args.out} at {args.hz} Hz — Ctrl-C to stop")
        try:
            while True:
                t0 = time.time()
                try:
                    w = read_power()
                    f.write(f"{t0:.4f},{w:.4f}\n")
                except Exception as e:
                    print(f"read error: {e}", file=sys.stderr)
                dt = time.time() - t0
                if dt < period:
                    time.sleep(period - dt)
        except KeyboardInterrupt:
            print("\nstopped")

if __name__ == "__main__":
    main()
