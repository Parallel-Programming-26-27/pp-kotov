#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#define SEED 20261021ULL
#define MIN_RUN_TIME 0.1
#define ZERO_EPS 1e-6
#define TOL_W 1e-9
#define TOL_LOSS 1e-12
#define CACHE_LINE_DOUBLES 8

typedef enum { SERIAL, OMP, SIMD, OMP_SIMD, VARIANT_COUNT } Variant;

static const char *VARIANT_NAMES[] = {"serial", "omp", "simd", "omp_simd"};

typedef struct {
    int m, n, iters, threads, runs;
    double lambda, lr;
    int selected[VARIANT_COUNT];
} Params;

typedef struct {
    int m, n;
    double *X;
    double *y;
} Dataset;

typedef struct {
    double *w;
    double b;
} Model;

typedef struct {
    double *grad;
    double *local;
    size_t stride;
} Workspace;

static double next_uniform(unsigned long long *s) {
    *s ^= *s >> 12;
    *s ^= *s << 25;
    *s ^= *s >> 27;
    return (double)((*s * 2685821657736338717ULL) >> 11) / 9007199254740992.0;
}

static void generate(Dataset *d, double *w_true) {
    unsigned long long s = SEED;
    int n = d->n;
    for (int j = 0; j < n; ++j) {
        double magnitude = 0.05 + 1.95 * next_uniform(&s);
        double sign = next_uniform(&s) < 0.5 ? -1.0 : 1.0;
        w_true[j] = (j % 2) ? 0.0 : sign * magnitude;
    }
    double b_true = 0.5;
    for (int i = 0; i < d->m; ++i) {
        double *x = d->X + (size_t)i * n;
        double target = b_true;
        for (int j = 0; j < n; ++j) {
            x[j] = 2.0 * next_uniform(&s) - 1.0;
            target += x[j] * w_true[j];
        }
        d->y[i] = target + 0.1 * (2.0 * next_uniform(&s) - 1.0);
    }
}

/* Невязка r_i = <x_i, w> + b - y_i для одного объекта: добавляет r_i * x_i в g
   и r_i в g[n]. Без прагм порядок сложений в редукции по IEEE менять нельзя,
   поэтому GCC векторизует скалярное произведение только как in-order редукцию:
   загрузки и умножения векторные, а сложения идут по одному. Обновление g
   компилятор векторизует сам. */
static inline void accumulate_row(const double *restrict x, double y, const double *restrict w,
                                  double b, double *restrict g, int n) {
    double pred = 0.0;
    for (int j = 0; j < n; ++j)
        pred += x[j] * w[j];
    double r = pred + b - y;
    for (int j = 0; j < n; ++j)
        g[j] += r * x[j];
    g[n] += r;
}

/* То же самое, но с явным SIMD. reduction(+:pred) разрешает компилятору держать
   несколько частичных сумм в векторных регистрах и сложить их в конце, а второй
   цикл помечен как SIMD, потому что его итерации независимы. */
static inline void accumulate_row_simd(const double *restrict x, double y, const double *restrict w,
                                       double b, double *restrict g, int n) {
    double pred = 0.0;
    #pragma omp simd reduction(+:pred)
    for (int j = 0; j < n; ++j)
        pred += x[j] * w[j];
    double r = pred + b - y;
    #pragma omp simd
    for (int j = 0; j < n; ++j)
        g[j] += r * x[j];
    g[n] += r;
}

__attribute__((noinline)) static void gradient_serial(const Dataset *d, const Model *mdl, Workspace *ws) {
    memset(ws->grad, 0, (d->n + 1) * sizeof(double));
    for (int i = 0; i < d->m; ++i)
        accumulate_row(d->X + (size_t)i * d->n, d->y[i], mdl->w, mdl->b, ws->grad, d->n);
}

__attribute__((noinline)) static void gradient_simd(const Dataset *d, const Model *mdl, Workspace *ws) {
    memset(ws->grad, 0, (d->n + 1) * sizeof(double));
    for (int i = 0; i < d->m; ++i)
        accumulate_row_simd(d->X + (size_t)i * d->n, d->y[i], mdl->w, mdl->b, ws->grad, d->n);
}

/* Объекты статически делятся между потоками. Каждый поток копит свою часть
   градиента в отдельном буфере, поэтому внутри цикла нет ни гонок, ни
   синхронизации. Буферы складываются после него — тоже параллельно, по признакам. */
static void gradient_omp(const Dataset *d, const Model *mdl, Workspace *ws, int threads, int use_simd) {
    int n = d->n;
    #pragma omp parallel num_threads(threads)
    {
        int nt = omp_get_num_threads();
        double *g = ws->local + (size_t)omp_get_thread_num() * ws->stride;
        memset(g, 0, (n + 1) * sizeof(double));

        if (use_simd) {
            #pragma omp for schedule(static)
            for (int i = 0; i < d->m; ++i)
                accumulate_row_simd(d->X + (size_t)i * n, d->y[i], mdl->w, mdl->b, g, n);
        } else {
            #pragma omp for schedule(static)
            for (int i = 0; i < d->m; ++i)
                accumulate_row(d->X + (size_t)i * n, d->y[i], mdl->w, mdl->b, g, n);
        }

        #pragma omp for schedule(static)
        for (int j = 0; j <= n; ++j) {
            double sum = 0.0;
            for (int t = 0; t < nt; ++t)
                sum += ws->local[(size_t)t * ws->stride + j];
            ws->grad[j] = sum;
        }
    }
}

static double soft_threshold(double v, double t) {
    return v > t ? v - t : (v < -t ? v + t : 0.0);
}

static void update(Model *mdl, const double *grad, int m, int n, double lr, double lambda) {
    for (int j = 0; j < n; ++j)
        mdl->w[j] = soft_threshold(mdl->w[j] - lr * grad[j] / m, lr * lambda);
    mdl->b -= lr * grad[n] / m;
}

static void train(const Dataset *d, Model *mdl, Workspace *ws, const Params *p, Variant v) {
    memset(mdl->w, 0, d->n * sizeof(double));
    mdl->b = 0.0;
    for (int it = 0; it < p->iters; ++it) {
        switch (v) {
        case SERIAL:   gradient_serial(d, mdl, ws); break;
        case SIMD:     gradient_simd(d, mdl, ws); break;
        case OMP:      gradient_omp(d, mdl, ws, p->threads, 0); break;
        case OMP_SIMD: gradient_omp(d, mdl, ws, p->threads, 1); break;
        default: break;
        }
        update(mdl, ws->grad, d->m, d->n, p->lr, p->lambda);
    }
}

static double loss(const Dataset *d, const Model *mdl, double lambda) {
    double sq = 0.0, l1 = 0.0;
    for (int i = 0; i < d->m; ++i) {
        const double *x = d->X + (size_t)i * d->n;
        double r = mdl->b - d->y[i];
        for (int j = 0; j < d->n; ++j)
            r += x[j] * mdl->w[j];
        sq += r * r;
    }
    for (int j = 0; j < d->n; ++j)
        l1 += fabs(mdl->w[j]);
    return sq / (2.0 * d->m) + lambda * l1;
}

static int count_zeros(const Model *mdl, int n) {
    int zeros = 0;
    for (int j = 0; j < n; ++j)
        zeros += fabs(mdl->w[j]) < ZERO_EPS;
    return zeros;
}

static double time_reps(const Dataset *d, Model *mdl, Workspace *ws, const Params *p, Variant v, int reps) {
    double t0 = omp_get_wtime();
    for (int r = 0; r < reps; ++r)
        train(d, mdl, ws, p, v);
    return (omp_get_wtime() - t0) / reps;
}

static int calibrate_reps(const Dataset *d, Model *mdl, Workspace *ws, const Params *p, Variant v) {
    int reps = 1;
    while (time_reps(d, mdl, ws, p, v, reps) * reps < MIN_RUN_TIME)
        reps *= 2;
    return reps;
}

static int cmp_double(const void *x, const void *y) {
    double diff = *(const double *)x - *(const double *)y;
    return (diff > 0) - (diff < 0);
}

static double measure(const Dataset *d, Model *mdl, Workspace *ws, const Params *p, Variant v) {
    double *t = malloc(p->runs * sizeof(double));
    int reps = calibrate_reps(d, mdl, ws, p, v);
    for (int r = 0; r < p->runs; ++r)
        t[r] = time_reps(d, mdl, ws, p, v, reps);
    qsort(t, p->runs, sizeof(double), cmp_double);
    double median = t[p->runs / 2];
    free(t);
    return median;
}

static int parse_args(int argc, char *argv[], Params *p) {
    *p = (Params){100000, 32, 100, omp_get_max_threads(), 5, 1e-3, 0.5, {1, 1, 1, 1}};
    for (int k = 1; k < argc; ++k) {
        char key[32];
        char value[64];
        if (sscanf(argv[k], "%31[^=]=%63s", key, value) != 2)
            return 0;
        if (strcmp(key, "m") == 0) p->m = atoi(value);
        else if (strcmp(key, "n") == 0) p->n = atoi(value);
        else if (strcmp(key, "iters") == 0) p->iters = atoi(value);
        else if (strcmp(key, "threads") == 0) p->threads = atoi(value);
        else if (strcmp(key, "runs") == 0) p->runs = atoi(value);
        else if (strcmp(key, "lambda") == 0) p->lambda = atof(value);
        else if (strcmp(key, "lr") == 0) p->lr = atof(value);
        else if (strcmp(key, "variant") == 0) {
            int found = strcmp(value, "all") == 0;
            for (int v = 0; v < VARIANT_COUNT; ++v)
                if (strcmp(value, VARIANT_NAMES[v]) == 0)
                    found = 1;
            if (!found)
                return 0;
            for (int v = 0; v < VARIANT_COUNT; ++v)
                p->selected[v] = strcmp(value, "all") == 0 || strcmp(value, VARIANT_NAMES[v]) == 0;
        } else
            return 0;
    }
    return p->m > 0 && p->n > 0 && p->iters > 0 && p->threads > 0 && p->runs > 0 && p->lr > 0 && p->lambda >= 0;
}

int main(int argc, char *argv[]) {
    Params p;
    if (!parse_args(argc, argv, &p)) {
        fprintf(stderr, "Usage: %s [m=] [n=] [iters=] [lambda=] [lr=] [threads=] [runs=] "
                        "[variant=all|serial|omp|simd|omp_simd]\n", argv[0]);
        return 1;
    }

    Dataset d = {p.m, p.n, malloc((size_t)p.m * p.n * sizeof(double)), malloc(p.m * sizeof(double))};
    double *w_true = malloc(p.n * sizeof(double));
    Model mdl = {malloc(p.n * sizeof(double)), 0.0};
    Model ref = {malloc(p.n * sizeof(double)), 0.0};
    /* Буферы потоков разнесены минимум на одну кэш-линию, чтобы не было
       ложного разделения (false sharing) */
    size_t stride = ((size_t)p.n + 1 + CACHE_LINE_DOUBLES - 1) / CACHE_LINE_DOUBLES * CACHE_LINE_DOUBLES
                    + CACHE_LINE_DOUBLES;
    Workspace ws = {malloc((p.n + 1) * sizeof(double)), malloc(stride * p.threads * sizeof(double)), stride};
    if (!d.X || !d.y || !w_true || !mdl.w || !ref.w || !ws.grad || !ws.local) {
        fprintf(stderr, "not enough memory\n");
        return 1;
    }
    generate(&d, w_true);

    train(&d, &ref, &ws, &p, SERIAL);
    double ref_loss = loss(&d, &ref, p.lambda);

    printf("m=%d n=%d iters=%d lambda=%g lr=%g threads=%d runs=%d\n",
           p.m, p.n, p.iters, p.lambda, p.lr, p.threads, p.runs);
    double t_serial = 0.0;
    int failed = 0;
    for (int v = 0; v < VARIANT_COUNT; ++v) {
        if (!p.selected[v])
            continue;
        double time = measure(&d, &mdl, &ws, &p, (Variant)v);
        double final_loss = loss(&d, &mdl, p.lambda);
        double dw = 0.0;
        for (int j = 0; j < p.n; ++j)
            dw = fmax(dw, fabs(mdl.w[j] - ref.w[j]));
        int ok = dw <= TOL_W && fabs(final_loss - ref_loss) <= TOL_LOSS * fmax(1.0, ref_loss);
        failed |= !ok;
        if (v == SERIAL)
            t_serial = time;

        int threads = (v == OMP || v == OMP_SIMD) ? p.threads : 1;
        printf("variant=%-8s threads=%d time=%.6f s loss=%.10f zeros=%d/%d dw=%.1e check=%s",
               VARIANT_NAMES[v], threads, time, final_loss, count_zeros(&mdl, p.n), p.n, dw, ok ? "OK" : "FAIL");
        if (t_serial > 0.0)
            printf(" speedup=%.2f", t_serial / time);
        printf("\n");
    }

    free(d.X);
    free(d.y);
    free(w_true);
    free(mdl.w);
    free(ref.w);
    free(ws.grad);
    free(ws.local);
    return failed ? 2 : 0;
}
