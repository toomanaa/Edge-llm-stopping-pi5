# Environment

- Board: Raspberry Pi 5 (power read with `vcgencmd pmic_read_adc`, logged at 20 Hz by `src/common/power_logger.py`).
- llama.cpp pinned to commit `89e0aa6fd362617d9073e0dafc18e41241521572`
  (download the tarball of that commit; build with cmake, then compile the controller):

```
g++ -O2 -std=c++17 src/controller/sp_controller.cpp -o sp_controller \
  -I llama.cpp/include -I llama.cpp/ggml/include -I llama.cpp/vendor \
  -L llama.cpp/build/bin -lllama -lggml -lggml-base -lggml-cpu -Wl,-rpath,$HOME/e1/llama.cpp/build/bin
```

- Model: `llama-3.2-1b-instruct-q4_k_m.gguf`. SHA-256: TO FILL (run `sha256sum` on the Pi).
- Python: numpy, scipy, scikit-learn, matplotlib (analysis); `python3-sklearn` on the Pi for probe training.
- `setup_pi.sh`: earlier setup script, kept for reference.
