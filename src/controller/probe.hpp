// Tiny C++ copy of the trained probe (StandardScaler -> 64 ReLU -> sigmoid).
#pragma once
#include <cmath>
#include <fstream>
#include <string>
#include <vector>
struct Probe {
    int d = 0, h = 0;
    std::vector<double> mean, scale, W0, b0, W1;
    double b1 = 0;
    bool load(const std::string & path) {
        std::ifstream f(path);
        if (!f) return false;
        f >> d >> h;
        if (d <= 0 || h <= 0) return false;
        mean.resize(d); scale.resize(d); W0.resize((size_t) d * h); b0.resize(h); W1.resize(h);
        for (auto & x : mean) f >> x;
        for (auto & x : scale) f >> x;
        for (auto & x : W0) f >> x;
        for (auto & x : b0) f >> x;
        for (auto & x : W1) f >> x;
        f >> b1;
        return (bool) f;
    }
    double predict(const float * x) const {
        std::vector<double> z(b0);
        for (int i = 0; i < d; ++i) {
            const double xi = ((double) x[i] - mean[i]) / scale[i];
            const double * w = &W0[(size_t) i * h];
            for (int j = 0; j < h; ++j) z[j] += xi * w[j];
        }
        double o = b1;
        for (int j = 0; j < h; ++j) o += (z[j] > 0 ? z[j] : 0.0) * W1[j];
        return 1.0 / (1.0 + std::exp(-o));
    }
};
