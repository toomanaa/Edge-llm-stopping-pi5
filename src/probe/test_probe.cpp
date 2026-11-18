#include "probe.hpp"
#include <cstdio>
#include <random>
int main(int argc, char ** argv) {
    Probe p; if (!p.load(argv[1])) { puts("load failed"); return 1; }
    std::mt19937 g(0); std::normal_distribution<float> n(0.f, 1.f);
    std::vector<float> x(p.d);
    for (int k = 0; k < 5; ++k) { for (auto & v : x) v = n(g); printf("%.9f\n", p.predict(x.data())); }
    // also dump the inputs for python comparison
    FILE * f = fopen("test_inputs.txt", "w"); g.seed(0);
    for (int k = 0; k < 5; ++k) { for (auto & v : x) v = n(g); for (float v : x) fprintf(f, "%.9g ", v); fprintf(f, "\n"); }
    fclose(f); return 0;
}
