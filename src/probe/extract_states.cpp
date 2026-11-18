// Extract, from the deployed 4-bit model on the Pi, the hidden state the live
// controller would see at every sentence boundary of every stored answer.
//
// For each answer: tokenise (prompt + full text), run ONE forward pass that keeps
// the hidden state of every position, then for every sentence boundary use the
// state of the last token of tokenise(prompt + text-up-to-boundary), exactly as
// the old /embeddings pipeline did. When that tokenisation is not a prefix of the
// full one (e.g. "." versus ".\n\n"), the KV cache is rolled back to the first
// differing position and the few differing tokens are decoded.
// Boundaries are handled from the last to the first so rollbacks stay short.
//
// Output: states.f32 (float32, n_embd per boundary, in index order) and
// index.jsonl (one line per boundary: id, boundary, off, aligned, n_tok_prefix,
// n_tok_total).
#include "llama.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cctype>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>
using json = nlohmann::json;

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
static bool is_boundary_at(const std::string & t, size_t j) {   // same rule as the live controller
    if (j == 0 || j >= t.size() || !isspace((unsigned char) t[j])) return false;
    char p = t[j - 1];
    if (p != '.' && p != '!' && p != '?') return false;
    if (p == '.' && j >= 2 && isdigit((unsigned char) t[j - 2])) return false;
    return true;
}
// decode toks[from,to) in chunks of <=256; if `keep` is non-null, store every position's state
static bool decode_range(llama_context * ctx, std::vector<llama_token> & toks, size_t from, size_t to,
                         int n_embd, std::vector<float> * keep) {
    while (from < to) {
        size_t n = std::min<size_t>(256, to - from);
        if (llama_decode(ctx, llama_batch_get_one(toks.data() + from, (int32_t) n)) != 0) return false;
        if (keep) {
            for (size_t i = 0; i < n; ++i) {
                const float * e = llama_get_embeddings_ith(ctx, (int32_t) i);
                if (!e) return false;
                std::copy(e, e + n_embd, keep->begin() + (from + i) * n_embd);
            }
        }
        from += n;
    }
    return true;
}

int main(int argc, char ** argv) {
    std::string model_path, data, outbin = "states.f32", outidx = "index.jsonl";
    int threads = 4, limit = 0, skip = 0;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto nx = [&]() { return std::string(i + 1 < argc ? argv[++i] : ""); };
        if (a == "--model") model_path = nx();
        else if (a == "--data") data = nx();
        else if (a == "--outbin") outbin = nx();
        else if (a == "--outidx") outidx = nx();
        else if (a == "--threads") threads = atoi(nx().c_str());
        else if (a == "--limit") limit = atoi(nx().c_str());
        else if (a == "--skip") skip = atoi(nx().c_str());
    }
    if (model_path.empty() || data.empty()) { fprintf(stderr, "usage: extract_states --model M --data gen_slim.jsonl [--outbin states.f32 --outidx index.jsonl --limit N --skip N]\n"); return 2; }
    llama_log_set([](ggml_log_level l, const char * t, void *) { if (l >= GGML_LOG_LEVEL_ERROR) fputs(t, stderr); }, nullptr);
    llama_backend_init();
    llama_model * model = llama_model_load_from_file(model_path.c_str(), llama_model_default_params());
    if (!model) { fprintf(stderr, "model load failed\n"); return 1; }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 2048; cp.n_batch = 512; cp.n_ubatch = 512; cp.n_threads = threads; cp.n_threads_batch = threads;
    cp.embeddings = true; cp.pooling_type = LLAMA_POOLING_TYPE_NONE;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "context failed\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_embd = llama_model_n_embd_out(model);
    llama_memory_t mem = llama_get_memory(ctx);

    std::ifstream fin(data);
    std::ofstream fbin(outbin, std::ios::binary | std::ios::app), fidx(outidx, std::ios::app);
    std::string line; int n = 0, done = 0;
    while (std::getline(fin, line)) {
        if (line.empty()) continue;
        if (n++ < skip) continue;
        if (limit && done >= limit) break;
        json r = json::parse(line);
        const std::string id = r["id"], text = r["text"].get<std::string>(), prompt = chat_prompt(r["question"].get<std::string>());
        std::vector<llama_token> S = tok(vocab, prompt + text);
        const size_t n_prompt = tok(vocab, prompt).size();
        if (S.size() > 1900) { fprintf(stderr, "skip %s: %zu tokens\n", id.c_str(), S.size()); continue; }
        std::vector<size_t> B;
        for (size_t j = 1; j < text.size(); ++j) if (is_boundary_at(text, j)) B.push_back(j);
        ++done;
        if (B.empty()) { fprintf(stderr, "[%d] %s no boundaries\n", n, id.c_str()); continue; }
        llama_memory_clear(mem, true);
        std::vector<float> all(S.size() * (size_t) n_embd);
        if (!decode_range(ctx, S, 0, S.size(), n_embd, &all)) { fprintf(stderr, "decode failed %s\n", id.c_str()); continue; }
        size_t valid = S.size();                  // leading positions of the KV cache that equal S
        std::vector<std::vector<float>> H(B.size());
        std::vector<char> aligned(B.size(), 0);
        std::vector<size_t> ntok(B.size(), 0);
        int n_roll = 0;
        for (size_t k = B.size(); k-- > 0;) {
            std::string prefix = text.substr(0, B[k]);
            std::vector<llama_token> P = tok(vocab, prompt + prefix);
            size_t L = 0;
            while (L < P.size() && L < S.size() && P[L] == S[L]) ++L;
            ntok[k] = P.size() > n_prompt ? P.size() - n_prompt : 0;
            if (L == P.size() && P.size() > 0) {
                aligned[k] = 1;
                H[k].assign(all.begin() + (P.size() - 1) * n_embd, all.begin() + P.size() * n_embd);
            } else {
                ++n_roll;
                if (L > valid) {                       // restore the cache up to L from the full sequence
                    llama_memory_seq_rm(mem, 0, (llama_pos) valid, -1);
                    decode_range(ctx, S, valid, L, n_embd, nullptr);
                    valid = L;
                }
                llama_memory_seq_rm(mem, 0, (llama_pos) L, -1);
                std::vector<float> last(n_embd, 0.f);
                bool ok = decode_range(ctx, P, L, P.size(), n_embd, nullptr);
                const float * e = ok ? llama_get_embeddings_ith(ctx, -1) : nullptr;
                if (e) last.assign(e, e + n_embd);
                H[k] = last;
                valid = L;
            }
        }
        for (size_t k = 0; k < B.size(); ++k) {
            fbin.write((const char *) H[k].data(), sizeof(float) * n_embd);
            json o = {{"id", id}, {"boundary", (int) k}, {"off", (int) B[k]}, {"aligned", (bool) aligned[k]},
                      {"n_tok_prefix", (int) ntok[k]}, {"n_tok_total", (int) (S.size() - n_prompt)}};
            fidx << o.dump() << "\n";
        }
        fbin.flush(); fidx.flush();
        fprintf(stderr, "[%d] %s boundaries=%zu rollbacks=%d tokens=%zu\n", n, id.c_str(), B.size(), n_roll, S.size() - n_prompt);
    }
    llama_free(ctx); llama_model_free(model); llama_backend_free();
    return 0;
}
