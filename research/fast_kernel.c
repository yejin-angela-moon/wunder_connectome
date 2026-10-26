/* One row of the GRU ensemble in multi_solution.py with half-precision weights.

Compiled by the callback at load time (gcc -O3 -march=native). Weight matrices
are stored in panels of 32 rows (the last panel holds the remaining 8, 16 or 24): for each panel, column by column, the 32 row
values (float16). Activations, hidden states and output heads stay float32.
Member layout (meta, 8 int64 per member):
    kind (0 plain, 1 split encoder, 2 second layer, 3 plain on a column subset, 4 group
    MLP encoder: n hidden units, then its group count in the head field and the x index
    of its codes in the state field; it runs before the members that read the codes), n,
    input columns, W_ih offset, W_hh offset, head offset, hidden-state offset, then
    the encoder offset (split), the first source's hidden-state offset (second
    layer: it reads that many consecutive state values, i.e. [h, 1] of one or more
    adjacent members), or the offset of its column indices into x (kind 3)
The input row x is [clipped raw (112), 1, event features] and is filled by the
caller. A hidden state holds n values and a constant 1 (bias column).
*/
#include <stdint.h>

#if defined(__AVX2__) && defined(__FMA__) && defined(__F16C__)
#include <immintrin.h>
#elif !defined(__aarch64__)
#error "needs AVX2, FMA and F16C, or AArch64"
#endif

#define PANEL 32

typedef struct {
    int64_t members;
    float *x;
    const int64_t *meta;
    const uint16_t *weights;
    const float *heads;
    float *state;
    float *scratch;                       /* gi (3n), gh (3n), encoder (96), encoded (137) */
    float *out;                           /* (2) */
    const int64_t *columns;               /* column indices for kind-3 members */
} Context;

/* y[rows] = W v for a paneled float16 W of shape (rows, cols), rows a multiple of 8: full panels of PANEL rows,
   then one panel of the remaining 8, 16 or 24 rows, each stored column by column */
static void gemv(const uint16_t *restrict w, const float *restrict v, int64_t rows, int64_t cols,
                 float *restrict y)
{
    int64_t p = 0;
    for (; p + PANEL <= rows; p += PANEL) {
#if defined(__AVX2__)
        __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
        __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
        for (int64_t j = 0; j < cols; j++, w += PANEL) {
            __m256 s = _mm256_broadcast_ss(v + j);
            a0 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(w + 0))), s, a0);
            a1 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(w + 8))), s, a1);
            a2 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(w + 16))), s, a2);
            a3 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(w + 24))), s, a3);
        }
        _mm256_storeu_ps(y + p, a0);
        _mm256_storeu_ps(y + p + 8, a1);
        _mm256_storeu_ps(y + p + 16, a2);
        _mm256_storeu_ps(y + p + 24, a3);
#else
        const __fp16 *h = (const __fp16 *)w;
        float a[PANEL] = {0};
        for (int64_t j = 0; j < cols; j++, h += PANEL)
            for (int r = 0; r < PANEL; r++)
                a[r] += (float)h[r] * v[j];
        for (int r = 0; r < PANEL; r++)
            y[p + r] = a[r];
        w += PANEL * cols;
#endif
    }
    int64_t r = rows - p;
    if (r) {
#if defined(__AVX2__)
        __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps(), a2 = _mm256_setzero_ps();
        for (int64_t j = 0; j < cols; j++, w += r) {
            __m256 s = _mm256_broadcast_ss(v + j);
            a0 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)w)), s, a0);
            if (r > 8)
                a1 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(w + 8))), s, a1);
            if (r > 16)
                a2 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(w + 16))), s, a2);
        }
        _mm256_storeu_ps(y + p, a0);
        if (r > 8)
            _mm256_storeu_ps(y + p + 8, a1);
        if (r > 16)
            _mm256_storeu_ps(y + p + 16, a2);
#else
        const __fp16 *h = (const __fp16 *)w;
        float a[PANEL] = {0};
        for (int64_t j = 0; j < cols; j++, h += r)
            for (int64_t k = 0; k < r; k++)
                a[k] += (float)h[k] * v[j];
        for (int64_t k = 0; k < r; k++)
            y[p + k] = a[k];
#endif
    }
}

/* exp by Cody-Waite reduction and the Cephes polynomial; written to vectorize */
static inline float fast_exp(float x)
{
    x = x < -87.0f ? -87.0f : x;
    x = x > 88.0f ? 88.0f : x;
    float k = __builtin_floorf(x * 1.44269504088896341f + 0.5f);
    float r = x - k * 0.693359375f + k * 2.12194440e-4f;
    float p = 1.9875691500e-4f;
    p = p * r + 1.3981999507e-3f;
    p = p * r + 8.3334519073e-3f;
    p = p * r + 4.1665795894e-2f;
    p = p * r + 1.6666665459e-1f;
    p = p * r + 5.0000001201e-1f;
    p = p * r * r + r + 1.0f;
    union { int32_t i; float f; } scale = {((int32_t)k + 127) << 23};
    return p * scale.f;
}

static void gru(const uint16_t *restrict w_hh, const float *restrict gi, float *restrict gh,
                float *restrict h, int64_t n)
{
    gemv(w_hh, h, 3 * n, n + 1, gh);
    for (int64_t i = 0; i < 2 * n; i++)                                   /* reset and update gates */
        gh[i] = 1.0f / (1.0f + fast_exp(-(gi[i] + gh[i])));
    for (int64_t i = 0; i < n; i++) {
        float c = 2.0f / (1.0f + fast_exp(-2.0f * (gi[2 * n + i] + gh[i] * gh[2 * n + i]))) - 1.0f;
        h[i] = (h[i] - c) * gh[n + i] + c;
    }
}

/* One GRU update on fixed data (sum of the new state, 18.15214); the caller runs it in a child process
   before trusting this build on the host CPU */
double selftest(void)
{
    static uint16_t w[96 * 33];
    static float gi[96], gh[96], h[33];
    for (int i = 0; i < 96 * 33; i++)
        w[i] = 0x2e66;                                                   /* float16 0.1 */
    for (int i = 0; i < 96; i++)
        gi[i] = 0.01f * i - 0.5f;
    for (int i = 0; i < 32; i++)
        h[i] = 0.5f;
    h[32] = 1.0f;
    gru(w, gi, gh, h, 32);
    double s = 0.0;
    for (int i = 0; i < 32; i++)
        s += h[i];
    return s;
}

void step(Context *c, int need)
{
    float o0 = 0.0f, o1 = 0.0f;
    for (int64_t m = 0; m < c->members; m++) {
        const int64_t *d = c->meta + 8 * m;
        int64_t n = d[1];
        if (d[0] == 4) {                          /* group MLP encoder: writes 16 codes per group into x */
            const int64_t *columns = c->columns + d[7];
            float *g = c->scratch, *hid = g + d[2];
            for (int64_t k = 0; k < d[5]; k++) {
                for (int64_t i = 0; i < d[2]; i++)
                    g[i] = c->x[columns[d[2] * k + i]];
                gemv(c->weights + d[3] + k * n * d[2], g, n, d[2], hid);
                for (int64_t i = 0; i < n; i++)
                    hid[i] = 2.0f / (1.0f + fast_exp(-2.0f * hid[i])) - 1.0f;
                hid[n] = 1.0f;
                gemv(c->weights + d[4] + k * 16 * (n + 1), hid, 16, n + 1, c->x + d[6] + 16 * k);
            }
            continue;
        }
        float *gi = c->scratch, *gh = gi + 3 * n, *h = c->state + d[6];
        const float *input = c->x;
        if (d[0] == 1) {
            float *z = gh + 3 * n, *e = z + 96;
            gemv(c->weights + d[7], c->x, 72, 113, z);
            for (int i = 0; i < 72; i++)
                z[i] = 2.0f / (1.0f + fast_exp(-2.0f * z[i])) - 1.0f;
            for (int i = 0; i < 32; i++) {
                e[i] = z[i];
                e[32 + i] = z[32 + i];
                e[64 + i] = z[i] - z[32 + i];
                e[96 + i] = z[i] * z[32 + i];
            }
            for (int i = 0; i < 8; i++)
                e[128 + i] = z[64 + i];
            e[136] = 1.0f;
            input = e;
        } else if (d[0] == 2) {
            input = c->state + d[7];
        } else if (d[0] == 3) {
            float *g = gh + 3 * n;
            const int64_t *columns = c->columns + d[7];
            for (int64_t i = 0; i < d[2]; i++)
                g[i] = c->x[columns[i]];
            input = g;
        }
        gemv(c->weights + d[3], input, 3 * n, d[2], gi);
        gru(c->weights + d[4], gi, gh, h, n);
        if (need) {
            const float *head = c->heads + d[5];
            for (int64_t i = 0; i <= n; i++) {
                o0 += head[i] * h[i];
                o1 += head[n + 1 + i] * h[i];
            }
        }
    }
    c->out[0] = o0;
    c->out[1] = o1;
}
