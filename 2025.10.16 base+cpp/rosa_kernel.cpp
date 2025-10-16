#include <torch/extension.h>
#include <vector>
#include <cstdint>
#include <limits>

#ifdef _OPENMP
#include <omp.h>
#endif

using i32 = int32_t;
using i64 = int64_t;

static inline i64 off3(i64 b, i64 t, i64 m, i64 T, i64 M) {
    return ((b * T) + t) * M + m;
}

static inline i64 off4_bmsk(i64 b, i64 m, i64 s, i64 k, i64 M, i64 S, i64 K) {
    return (((b * M + m) * S) + s) * K + k;
}

static inline i64 off3_bms(i64 b, i64 m, i64 s, i64 M, i64 S) {
    return ((b * M + m) * S) + s;
}

static inline i64 off3_bmt(i64 b, i64 m, i64 t, i64 M, i64 T) {
    return ((b * M + m) * T) + t;
}

static inline i64 off2_bm(i64 b, i64 m, i64 M) {
    return b * M + m;
}

static inline i32 sam_new_state(
    i32* next_arr, i32* link, i32* length, i32* e,
    i32& size, const i32 K, const i32 L, const i64 row_strideK)
{
    i32 s = size;
    size += 1;
    length[s] = L;
    link[s]   = -1;
    e[s]      = -1;
    i32* row = next_arr + (i64)s * row_strideK;
    for (i32 j = 0; j < K; ++j) row[j] = -1;
    return s;
}

static inline void sam_extend(
    i32* next_arr, i32* link, i32* length, i32* e,
    i32& last_state, i32& size, const i32 K, const i32 x, const i32 pos,
    const i64 row_strideK)
{
    const i32 last = last_state;
    const i32 cur = sam_new_state(next_arr, link, length, e, size, K, length[last] + 1, row_strideK);
    i32 p = last;

    while (p != -1) {
        i32* prow = next_arr + (i64)p * row_strideK;
        if (prow[x] == -1) {
            prow[x] = cur;
            p = link[p];
        } else {
            break;
        }
    }

    if (p == -1) {
        link[cur] = 0;
    } else {
        i32* prow = next_arr + (i64)p * row_strideK;
        const i32 q = prow[x];
        if (length[p] + 1 == length[q]) {
            link[cur] = q;
        } else {
            const i32 clone = sam_new_state(next_arr, link, length, e, size, K, length[p] + 1, row_strideK);
            i32* crow = next_arr + (i64)clone * row_strideK;
            i32* qrow = next_arr + (i64)q     * row_strideK;
            for (i32 j = 0; j < K; ++j) crow[j] = qrow[j];
            link[clone] = link[q];
            e[clone]    = e[q];
            while (p != -1) {
                i32* prow2 = next_arr + (i64)p * row_strideK;
                if (prow2[x] == q) {
                    prow2[x] = clone;
                    p = link[p];
                } else break;
            }
            link[q]   = clone;
            link[cur] = clone;
        }
    }

    i32 v = cur;
    while (v != -1 && e[v] != pos) {
        e[v] = pos;
        v = link[v];
    }
    last_state = cur;
}

static inline i32 sam_match_next(const i32* next_arr, const i32* link,
                                 i32 last_state, const i32 x, const i32 row_strideK)
{
    i32 p = last_state;
    while (p != -1) {
        const i32* prow = next_arr + (i64)p * row_strideK;
        if (prow[x] != -1) return prow[x];
        p = link[p];
    }
    return -1;
}

std::vector<at::Tensor> rosa_batch_btm_with_ws(at::Tensor z_btm, i64 K_in) {
    TORCH_CHECK(z_btm.device().is_cpu(), "z_btm must be on CPU");
    TORCH_CHECK(z_btm.dtype() == at::kInt, "z_btm must be int32");
    TORCH_CHECK(z_btm.dim() == 3, "z_btm must be [B,T,M]");

    const i64 B = z_btm.size(0);
    const i64 T = z_btm.size(1);
    const i64 M = z_btm.size(2);
    const i32 K = static_cast<i32>(K_in);
    const i64 S = 2 * T + 5;

    auto opts_i32 = at::TensorOptions().dtype(at::kInt).device(at::kCPU).pinned_memory(true);

    at::Tensor y_btm         = at::empty({B, T, M}, opts_i32);
    at::Tensor last_btm      = at::empty({B, T, M}, opts_i32);
    at::Tensor next_bmsk     = at::empty({B, M, S, K}, opts_i32);
    at::Tensor link_bms      = at::empty({B, M, S}, opts_i32);
    at::Tensor e_bms         = at::empty({B, M, S}, opts_i32);
    at::Tensor c_bmt         = at::empty({B, M, T}, opts_i32);
    at::Tensor c_len_bm      = at::empty({B, M},    opts_i32);

    const i32* z_ptr   = z_btm.data_ptr<i32>();
    i32* y_ptr         = y_btm.data_ptr<i32>();
    i32* last_ptr      = last_btm.data_ptr<i32>();
    i32* next_ptr      = next_bmsk.data_ptr<i32>();
    i32* link_ptr      = link_bms.data_ptr<i32>();
    i32* e_ptr         = e_bms.data_ptr<i32>();
    i32* c_ptr         = c_bmt.data_ptr<i32>();
    i32* clen_ptr      = c_len_bm.data_ptr<i32>();

    const i64 BM = B * M;

    #pragma omp parallel for schedule(static)
    for (i64 idx = 0; idx < BM; ++idx) {
        const i64 b = idx / M;
        const i64 m = idx % M;

        i32* next_bm = next_ptr + off4_bmsk(b, m, 0, 0, M, S, K);
        i32* link_bm = link_ptr + off3_bms(b, m, 0, M, S);
        i32* e_bm    = e_ptr    + off3_bms(b, m, 0, M, S);

        std::vector<i32> length(S, 0);

        const i64 row_strideK = K;
        for (i32 j = 0; j < K; ++j) next_bm[j] = -1;
        link_bm[0]   = -1;
        length[0]    = 0;
        e_bm[0]      = -1;
        i32 size     = 1;
        i32 last     = 0;

        i32* c_bm = c_ptr + off3_bmt(b, m, 0, M, T);
        i32  c_len = 0;
        i32  last_sym = std::numeric_limits<i32>::min();

        for (i64 t = 0; t < T; ++t) {
            const i32 x = z_ptr[ off3(b, t, m, T, M) ];
            last_ptr[ off3(b, t, m, T, M) ] = last;

            const i32 q = sam_match_next(next_bm, link_bm, last, x, (i32)row_strideK);
            i32 yv = -1;
            if (q != -1) {
                const i32 rpos = e_bm[q];
                const i32 nxt  = rpos + 1;
                if (rpos >= 0 && nxt < c_len) {
                    yv = c_bm[nxt];
                }
            }
            y_ptr[ off3(b, t, m, T, M) ] = yv;

            if (t == 0 || x != last_sym) {
                c_bm[c_len] = x;
                sam_extend(next_bm, link_bm, length.data(), e_bm,
                           last, size, K, x, c_len, (i32)row_strideK);
                ++c_len;
                last_sym = x;
            }
        }

        clen_ptr[ off2_bm(b, m, M) ] = c_len;
        for (i64 t = c_len; t < T; ++t) c_bm[t] = 0;
    }

    return {y_btm, last_btm, next_bmsk, link_bms, e_bms, c_bmt, c_len_bm};
}

std::vector<at::Tensor> rosa_lcg_prefix(
    at::Tensor z_btm,
    at::Tensor cand_bmtk,
    c10::optional<at::Tensor> pos_mask_bt,
    int64_t K_in)
{
    TORCH_CHECK(z_btm.device().is_cpu(),    "z_btm must be on CPU");
    TORCH_CHECK(cand_bmtk.device().is_cpu(),"cand_bmtk must be on CPU");
    TORCH_CHECK(z_btm.dtype() == at::kInt,  "z_btm must be int32");
    TORCH_CHECK(cand_bmtk.dtype() == at::kInt, "cand_bmtk must be int32");
    TORCH_CHECK(z_btm.dim()==3 && cand_bmtk.dim()==4, "bad dims");

    const int64_t B = z_btm.size(0);
    const int64_t T = z_btm.size(1);
    const int64_t M = z_btm.size(2);
    const int64_t topk = cand_bmtk.size(3);
    const int32_t K = static_cast<int32_t>(K_in);
    const int64_t S = 2 * T + 5;

    at::Tensor pos_mask_cpu;
    const bool have_mask = pos_mask_bt.has_value();
    if (have_mask) {
        pos_mask_cpu = pos_mask_bt.value();
        TORCH_CHECK(pos_mask_cpu.device().is_cpu(), "pos_mask_bt must be on CPU");
        TORCH_CHECK(pos_mask_cpu.dtype() == at::kBool, "pos_mask_bt must be bool");
        TORCH_CHECK(pos_mask_cpu.sizes()==at::IntArrayRef({B,T}), "pos_mask_bt shape [B,T]");
    }

    auto opts_i32p = at::TensorOptions().dtype(at::kInt).device(at::kCPU).pinned_memory(true);
    auto opts_i16p = at::TensorOptions().dtype(at::kShort).device(at::kCPU).pinned_memory(true);

    at::Tensor y_btm     = at::empty({B, T, M},               opts_i32p);
    at::Tensor ycf_bmtk  = at::empty({B, M, T, topk},         opts_i16p);

    const int32_t* z_ptr    = z_btm.data_ptr<int32_t>();
    const int32_t* cand_ptr = cand_bmtk.data_ptr<int32_t>();
    const bool*    pmask    = have_mask ? pos_mask_cpu.data_ptr<bool>() : nullptr;

    int32_t* y_ptr   = y_btm.data_ptr<int32_t>();
    int16_t* ycf_ptr = ycf_bmtk.data_ptr<int16_t>();

    auto off3_loc = [&](int64_t b,int64_t t,int64_t m){ return ((b*T)+t)*M + m; };
    auto off4_loc = [&](int64_t b,int64_t m,int64_t t,int64_t j){ return (((b*M)+m)*T + t)*topk + j; };

    const int64_t BM = B * M;

    #pragma omp parallel for schedule(static)
    for (int64_t idx = 0; idx < BM; ++idx) {
        const int64_t b = idx / M;
        const int64_t m = idx % M;

        std::vector<int32_t> next_arr(S * K, -1);
        std::vector<int32_t> link(S, -1);
        std::vector<int32_t> length(S, 0);
        std::vector<int32_t> e(S, -1);
        std::vector<int32_t> c(T, 0);

        auto row = [&](int32_t s) -> int32_t* { return &next_arr[(int64_t)s * K]; };

        for (int32_t j=0;j<K;++j) row(0)[j] = -1;
        link[0] = -1; length[0]=0; e[0]=-1;
        int32_t size = 1;
        int32_t last = 0;
        int32_t c_len = 0;
        int32_t last_sym = std::numeric_limits<int32_t>::min();

        auto q_match = [&](int32_t last_state, int32_t x)->int32_t {
            int32_t p = last_state;
            while (p != -1) {
                int32_t q = row(p)[x];
                if (q != -1) return q;
                p = link[p];
            }
            return -1;
        };

        auto do_extend = [&](int32_t x, int32_t pos){
            int32_t cur = size++;
            for (int32_t j=0;j<K;++j) row(cur)[j] = -1;
            length[cur] = length[last] + 1;
            link[cur] = -1; e[cur] = -1;

            int32_t p = last;
            while (p != -1 && row(p)[x] == -1) {
                row(p)[x] = cur;
                p = link[p];
            }
            if (p == -1) {
                link[cur] = 0;
            } else {
                int32_t q = row(p)[x];
                if (length[p] + 1 == length[q]) {
                    link[cur] = q;
                } else {
                    int32_t clone = size++;
                    for (int32_t j=0;j<K;++j) row(clone)[j] = row(q)[j];
                    length[clone] = length[p] + 1;
                    link[clone] = link[q];
                    e[clone] = e[q];
                    while (p != -1 && row(p)[x] == q) {
                        row(p)[x] = clone;
                        p = link[p];
                    }
                    link[q] = clone;
                    link[cur] = clone;
                }
            }
            int32_t v = cur;
            while (v != -1 && e[v] != pos) {
                e[v] = pos;
                v = link[v];
            }
            last = cur;
        };

        for (int64_t t = 0; t < T; ++t) {
            const int32_t x = z_ptr[ off3_loc(b,t,m) ];

            const bool do_pos = (pmask==nullptr) ? true : pmask[b*T + t];
            if (!do_pos) {
                for (int64_t j=0;j<topk;++j) {
                    ycf_ptr[ off4_loc(b,m,t,j) ] = (int16_t)-2;
                }
            } else {
                for (int64_t j=0;j<topk;++j) {
                    const int32_t k = cand_ptr[ off4_loc(b,m,t,j) ];
                    const int32_t q = q_match(last, k);
                    int16_t outv = -1;
                    if (q != -1) {
                        const int32_t rpos = e[q];
                        const int32_t nxt  = rpos + 1;
                        if (rpos >= 0 && nxt < c_len) outv = (int16_t)c[nxt];
                    }
                    ycf_ptr[ off4_loc(b,m,t,j) ] = outv;
                }
            }

            {
                const int32_t q = q_match(last, x);
                int32_t yv = -1;
                if (q != -1) {
                    const int32_t rpos = e[q];
                    const int32_t nxt  = rpos + 1;
                    if (rpos >= 0 && nxt < c_len) yv = c[nxt];
                }
                y_ptr[ off3_loc(b,t,m) ] = yv;
            }

            if (t==0 || x != last_sym) {
                c[c_len] = x;
                do_extend(x, c_len);
                ++c_len;
                last_sym = x;
            }
        }
    }

    return { y_btm, ycf_bmtk };
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rosa_batch_btm_with_ws", &rosa_batch_btm_with_ws,
          "ROSA batch processing with workspace",
          pybind11::arg("z_btm"), pybind11::arg("K"));

    m.def("rosa_lcg_prefix", &rosa_lcg_prefix,
          "ROSA prefix processing",
          pybind11::arg("z_btm"),
          pybind11::arg("cand_bmtk"),
          pybind11::arg("pos_mask_bt") = c10::nullopt,
          pybind11::arg("K"));
}