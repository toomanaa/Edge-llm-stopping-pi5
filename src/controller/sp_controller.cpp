// Single-pass stopping controller.
//
// The old controller asked a llama.cpp server for the hidden state with a
// separate /embeddings request at every probe check; that endpoint re-reads the
// whole prompt + text so far each time (no prompt cache), which is where most
// of the controller's extra energy went. Here generation and the probe share one
// process and one KV cache: the hidden state of every generated token is kept
// from the decode step that already computed it, so a probe check costs about
// one 2048x64 matrix product.
//
// Exactness: the old pipeline re-tokenised (prompt + text up to the boundary)
// and used the hidden state of the LAST token of that tokenisation. We do the
// same. If the re-tokenised sequence is a prefix of the tokens we generated, the
// stored hidden state is exactly that token's. If it differs (e.g. the model
// emitted ".\n\n" as one token but the prefix ends at "."), we roll the KV cache
// back to the first differing position, decode the few re-tokenised tokens, read
// the state, and re-decode the original tokens. That costs a handful of tokens.
//
// All three conditions (no stopping, controller, token cap) run through the
// same loop, one after another for each question.
#include "llama.h"
#include "probe.hpp"
#include <nlohmann/json.hpp>
#include <sys/time.h>
#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <vector>
using json = nlohmann::json;

static double now_s() { timeval tv; gettimeofday(&tv, nullptr); return tv.tv_sec + tv.tv_usec * 1e-6; }

static const char * SYSTEM = "You are a helpful assistant. Answer the user's question clearly and correctly.";
static std::string chat_prompt(const std::string & q) {
    return std::string("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n") + SYSTEM +
           "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n" + q +
           "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n";
}

static std::vector<llama_token> tok(const llama_vocab * v, const std::string & s) {
    int n = -llama_tokenize(v, s.c_str(), (int) s.size(), nullptr, 0, true, true);
    std::vector<llama_token> t(std::max(n, 1));
    int r = llama_tokenize(v, s.c_str(), (int) s.size(), t.data(), (int) t.size(), true, true);
    t.resize(r < 0 ? 0 : r);
    return t;
}
static std::string piece(const llama_vocab * v, llama_token t) {
    char buf[256];
    int n = llama_token_to_piece(v, t, buf, sizeof buf, 0, false);
    if (n < 0) { std::string s(-n, 0); n = llama_token_to_piece(v, t, s.data(), (int) s.size(), 0, false); return s; }
    return std::string(buf, n);
}
// decode toks[from, to) in chunks; positions are tracked by the KV cache
static bool decode_range(llama_context * ctx, std::vector<llama_token> & toks, size_t from, size_t to) {
    while (from < to) {
        size_t n = std::min<size_t>(256, to - from);
        if (llama_decode(ctx, llama_batch_get_one(toks.data() + from, (int32_t) n)) != 0) return false;
        from += n;
    }
    return true;
}
// same rule as the Python BOUNDARY regex (?<!\d\.)(?<=[.!?])\s+ : whitespace right after . ! ?
// but not after "<digit>." (a list marker such as "1.")
static bool is_boundary_at(const std::string & t, size_t j) {
    if (j == 0 || j >= t.size()) return false;
    if (!isspace((unsigned char) t[j])) return false;
    char p = t[j - 1];
    if (p != '.' && p != '!' && p != '?') return false;
    if (p == '.' && j >= 2 && isdigit((unsigned char) t[j - 2])) return false;
    return true;
}

struct Cfg {
    std::string mode;     // full | controller | cap | check_single | check_reencode
                          // check_single / check_reencode: run the probe checks but never stop (overhead measurement)
    llama_context * ctx2 = nullptr;   // second context, used only by check_reencode
    double threshold = 0.99;
    int every_n = 2;
    int cap = 200;        // cap mode: stop at first sentence boundary with >= cap tokens
    int max_tokens = 800;
    bool dump = false;    // dump hidden states at every boundary (verification)
};
struct Run {
    std::string text; bool stopped = false; int n_tokens = 0, n_checks = 0, n_rollbacks = 0;
    double t0 = 0, t1 = 0, overhead_s = 0; std::vector<double> ps; std::string end_reason;
    std::vector<double> check_s; std::vector<int> check_tok;   // time and token count of every probe check
};

static Run generate(llama_model * model, llama_context * ctx, const Probe & probe, const Cfg & c,
                    const std::string & question, json * dump_out) {
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_embd = llama_model_n_embd_out(model);
    llama_memory_t mem = llama_get_memory(ctx);
    Run r;
    r.t0 = now_s();
    llama_memory_clear(mem, true);
    const std::string prompt = chat_prompt(question);
    std::vector<llama_token> S = tok(vocab, prompt);
    const size_t n_prompt = S.size();
    if (!decode_range(ctx, S, 0, S.size())) { r.end_reason = "prompt_decode_failed"; r.t1 = now_s(); return r; }
    llama_sampler * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
    std::vector<std::vector<float>> emb;   // hidden state of each generated token
    std::string text;
    size_t scan = 1;
    int since_check = 0;
    r.end_reason = "max_tokens";
    for (int g = 0; g < c.max_tokens; ++g) {
        llama_token t = llama_sampler_sample(smpl, ctx, -1);
        if (llama_vocab_is_eog(vocab, t)) { r.end_reason = "natural_end"; break; }
        S.push_back(t);
        text += piece(vocab, t);
        if (llama_decode(ctx, llama_batch_get_one(&S.back(), 1)) != 0) { r.end_reason = "decode_failed"; break; }
        const float * e = llama_get_embeddings_ith(ctx, -1);
        emb.emplace_back(e ? std::vector<float>(e, e + n_embd) : std::vector<float>(n_embd, 0.f));
        r.n_tokens = g + 1;
        bool stop = false;
        for (size_t j = scan; j < text.size() && !stop; ++j) {
            if (!is_boundary_at(text, j)) continue;
            if (c.mode == "cap") {
                if (r.n_tokens >= c.cap) { text.resize(j); r.stopped = true; r.end_reason = "cap"; stop = true; }
                continue;
            }
            if (c.mode != "controller" && c.mode != "check_single" && c.mode != "check_reencode" && !c.dump) continue;
            if (!c.dump) {
                if (++since_check < c.every_n) continue;
                since_check = 0;
            }
            // ---- probe check at this boundary ----
            const double c0 = now_s();
            r.n_checks++;
            std::string prefix = text.substr(0, j);
            std::vector<llama_token> P = tok(vocab, prompt + prefix);
            size_t L = 0;
            while (L < P.size() && L < S.size() && P[L] == S[L]) ++L;
            std::vector<float> h;
            bool aligned = (L == P.size() && P.size() > n_prompt && P.size() - n_prompt <= emb.size());
            if (c.mode == "check_reencode" && c.ctx2) {
                // server-style check: read the whole prompt + text so far again in a separate context with an empty
                // cache, as the /embeddings request did; the generation context is left untouched
                llama_memory_clear(llama_get_memory(c.ctx2), true);
                bool ok2 = decode_range(c.ctx2, P, 0, P.size());
                const float * e3 = ok2 ? llama_get_embeddings_ith(c.ctx2, -1) : nullptr;
                h = e3 ? std::vector<float>(e3, e3 + n_embd) : std::vector<float>();
            } else if (aligned) {
                h = emb[P.size() - 1 - n_prompt];
            } else {
                r.n_rollbacks++;
                llama_memory_seq_rm(mem, 0, (llama_pos) L, -1);
                bool ok = decode_range(ctx, P, L, P.size());
                const float * e2 = ok ? llama_get_embeddings_ith(ctx, -1) : nullptr;
                h = e2 ? std::vector<float>(e2, e2 + n_embd) : std::vector<float>();
                llama_memory_seq_rm(mem, 0, (llama_pos) L, -1);
                decode_range(ctx, S, L, S.size());   // restore the generated tokens
            }
            double p = h.empty() ? 0.0 : probe.predict(h.data());   // fail open: no state -> never stop
            r.ps.push_back(p);
            r.overhead_s += now_s() - c0;
            r.check_s.push_back(now_s() - c0); r.check_tok.push_back((int) P.size());
            if (dump_out) dump_out->push_back({{"prefix", prefix}, {"aligned", aligned}, {"p_safe", p}, {"emb", h}});
            if (!c.dump && c.mode == "controller" && p >= c.threshold) { text.resize(j); r.stopped = true; r.end_reason = "probe_stop"; stop = true; }
        }
        scan = text.size() > 0 ? text.size() : 1;
        if (stop) break;
    }
    llama_sampler_free(smpl);
    r.text = text;
    r.t1 = now_s();
    return r;
}

static std::vector<std::string> split(const std::string & s, char d) {
    std::vector<std::string> o; std::stringstream ss(s); std::string x;
    while (std::getline(ss, x, d)) if (!x.empty()) o.push_back(x);
    return o;
}

int main(int argc, char ** argv) {
    std::string model_path, data, out, probe_path = "probe.txt", modes = "controller", ids_file, dump_path;
    Cfg c; int limit = 0, threads = 4, ctxn = 2048;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto nx = [&]() { return std::string(i + 1 < argc ? argv[++i] : ""); };
        if (a == "--model") model_path = nx();
        else if (a == "--data") data = nx();
        else if (a == "--out") out = nx();
        else if (a == "--probe") probe_path = nx();
        else if (a == "--modes") modes = nx();
        else if (a == "--threshold") c.threshold = atof(nx().c_str());
        else if (a == "--check-every-n") c.every_n = atoi(nx().c_str());
        else if (a == "--cap-tokens") c.cap = atoi(nx().c_str());
        else if (a == "--max-tokens") c.max_tokens = atoi(nx().c_str());
        else if (a == "--limit") limit = atoi(nx().c_str());
        else if (a == "--threads") threads = atoi(nx().c_str());
        else if (a == "--ids") ids_file = nx();
        else if (a == "--dump-hidden") { c.dump = true; dump_path = nx(); }
        else { fprintf(stderr, "unknown arg %s\n", a.c_str()); return 2; }
    }
    if (model_path.empty() || data.empty() || out.empty()) {
        fprintf(stderr, "usage: sp_controller --model M.gguf --data qa.jsonl --out out.jsonl --probe probe.txt "
                        "[--modes full,controller,cap] [--threshold 0.99] [--check-every-n 2] [--cap-tokens 200] "
                        "[--limit N] [--ids ids.txt] [--dump-hidden dump.jsonl]\n");
        return 2;
    }
    Probe probe;
    if (!probe.load(probe_path)) { fprintf(stderr, "cannot read probe %s\n", probe_path.c_str()); return 1; }
    llama_log_set([](ggml_log_level l, const char * t, void *) { if (l >= GGML_LOG_LEVEL_ERROR) fputs(t, stderr); }, nullptr);
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) { fprintf(stderr, "model load failed\n"); return 1; }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = ctxn; cp.n_batch = 512; cp.n_ubatch = 512;
    cp.n_threads = threads; cp.n_threads_batch = threads;
    cp.embeddings = true; cp.pooling_type = LLAMA_POOLING_TYPE_NONE;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "context failed\n"); return 1; }
    llama_context * ctx2 = nullptr;
    if (modes.find("check_reencode") != std::string::npos) {
        ctx2 = llama_init_from_model(model, cp);
        if (!ctx2) { fprintf(stderr, "second context failed\n"); return 1; }
    }
    if (probe.d != llama_model_n_embd_out(model)) { fprintf(stderr, "probe dim %d != model %d\n", probe.d, llama_model_n_embd_out(model)); return 1; }

    std::set<std::string> keep;
    if (!ids_file.empty()) { std::ifstream f(ids_file); std::string s; while (f >> s) keep.insert(s); }
    std::set<std::string> done;
    { std::ifstream f(out); std::string line; while (std::getline(f, line)) { try { auto j = json::parse(line); done.insert(j["id"].get<std::string>() + "|" + j["mode"].get<std::string>()); } catch (...) {} } }
    std::ifstream fin(data);
    std::ofstream fout(out, std::ios::app);
    std::ofstream fdump; if (c.dump) fdump.open(dump_path, std::ios::app);
    std::string line; int n = 0;
    while (std::getline(fin, line)) {
        if (line.empty()) continue;
        json q = json::parse(line);
        std::string id = q.value("id", "q" + std::to_string(n));
        if (!keep.empty() && !keep.count(id)) continue;
        if (limit && n >= limit) break;
        ++n;
        for (const std::string & m : split(modes, ',')) {
            if (done.count(id + "|" + m)) continue;
            Cfg cc = c; cc.mode = m; cc.ctx2 = ctx2;
            json dumpj = json::array();
            Run r = generate(model, ctx, probe, cc, q["question"].get<std::string>(), c.dump ? &dumpj : nullptr);
            json o = {{"id", id}, {"mode", m}, {"question", q["question"]}, {"text", r.text},
                      {"stopped_early_by_probe", r.stopped && r.end_reason == "probe_stop"}, {"end_reason", r.end_reason},
                      {"boundary_checks", r.n_checks}, {"n_rollbacks", r.n_rollbacks}, {"n_tokens", r.n_tokens},
                      {"n_words_approx", (int) split(r.text, ' ').size()},
                      {"t_start", r.t0}, {"t_end", r.t1}, {"wall_seconds", r.t1 - r.t0},
                      {"check_overhead_seconds", r.overhead_s}, {"p_safe", r.ps},
                      {"check_seconds", r.check_s}, {"check_tokens", r.check_tok}};
            fout << o.dump(-1, ' ', false, json::error_handler_t::replace) << "\n"; fout.flush();
            if (c.dump) { for (auto & d : dumpj) { d["id"] = id; fdump << d.dump(-1, ' ', false, json::error_handler_t::replace) << "\n"; } fdump.flush(); }
            fprintf(stderr, "[%d] %s %-10s tokens=%d checks=%d %s %.1fs\n", n, id.c_str(), m.c_str(), r.n_tokens, r.n_checks, r.end_reason.c_str(), r.t1 - r.t0);
        }
    }
    if (ctx2) llama_free(ctx2);
    llama_free(ctx); llama_model_free(model); llama_backend_free();
    return 0;
}
