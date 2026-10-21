/*
 * Умножение матриц C = A * B (N x N, double) с использованием OpenMP.
 *
 * Сборка: gcc -O3 -fopenmp -o matmul matmul.c
 * Запуск: OMP_NUM_THREADS=<потоки> ./matmul N [block] [variants]
 * N = размер матрицы
 * block = размер блока для blocked_omp (по умолчанию 32)
 * variants = варианты через запятую: ijk_seq, ijk_omp, ikj_seq, ikj_omp, blocked_omp, либо all (по умолчанию)
 * Пример: OMP_NUM_THREADS=8 ./matmul 1024 64 ikj_seq,blocked_omp
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#define RUNS 5
#define MIN_RUN_TIME 0.1
#define TOLERANCE 1e-12
#define SEED 20261021u

static int min_int(int x, int y) {
    return x < y ? x : y;
}

void ijk_seq(const double *A, const double *B, double *C, int N) {
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            double sum = 0.0;
            for (int k = 0; k < N; ++k)
                sum += A[(size_t)i * N + k] * B[(size_t)k * N + j];
            C[(size_t)i * N + j] = sum;
        }
}

void ijk_omp(const double *A, const double *B, double *C, int N) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            double sum = 0.0;
            for (int k = 0; k < N; ++k)
                sum += A[(size_t)i * N + k] * B[(size_t)k * N + j];
            C[(size_t)i * N + j] = sum;
        }
}

static void ikj_row(const double *A, const double *B, double *C, int N, int i) {
    double *c = C + (size_t)i * N;
    memset(c, 0, (size_t)N * sizeof(double));
    for (int k = 0; k < N; ++k) {
        double aik = A[(size_t)i * N + k];
        const double *b = B + (size_t)k * N;
        for (int j = 0; j < N; ++j)
            c[j] += aik * b[j];
    }
}

void ikj_seq(const double *A, const double *B, double *C, int N) {
    for (int i = 0; i < N; ++i)
        ikj_row(A, B, C, N, i);
}

void ikj_omp(const double *A, const double *B, double *C, int N) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i)
        ikj_row(A, B, C, N, i);
}

static void compute_block(const double *A, const double *B, double *C, int N, int bs, int ii, int jj) {
    int i_end = min_int(ii + bs, N);
    int j_end = min_int(jj + bs, N);
    for (int i = ii; i < i_end; ++i)
        for (int j = jj; j < j_end; ++j)
            C[(size_t)i * N + j] = 0.0;
    for (int kk = 0; kk < N; kk += bs) {
        int k_end = min_int(kk + bs, N);
        for (int i = ii; i < i_end; ++i)
            for (int k = kk; k < k_end; ++k) {
                double aik = A[(size_t)i * N + k];
                for (int j = jj; j < j_end; ++j)
                    C[(size_t)i * N + j] += aik * B[(size_t)k * N + j];
            }
    }
}

void blocked_omp(const double *A, const double *B, double *C, int N, int bs) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (int ii = 0; ii < N; ii += bs)
        for (int jj = 0; jj < N; jj += bs)
            compute_block(A, B, C, N, bs, ii, jj);
}

typedef enum { IJK_SEQ, IJK_OMP, IKJ_SEQ, IKJ_OMP, BLOCKED_OMP, VARIANT_COUNT } Variant;

static const char *VARIANT_NAMES[] = {"ijk_seq", "ijk_omp", "ikj_seq", "ikj_omp", "blocked_omp"};

typedef struct {
    Variant variant;
    int N, bs;
    const double *A, *B;
    double *C;
} Task;

static void run_task(const Task *task) {
    switch (task->variant) {
    case IJK_SEQ:     ijk_seq(task->A, task->B, task->C, task->N); break;
    case IJK_OMP:     ijk_omp(task->A, task->B, task->C, task->N); break;
    case IKJ_SEQ:     ikj_seq(task->A, task->B, task->C, task->N); break;
    case IKJ_OMP:     ikj_omp(task->A, task->B, task->C, task->N); break;
    case BLOCKED_OMP: blocked_omp(task->A, task->B, task->C, task->N, task->bs); break;
    default: break;
    }
}

static int is_parallel(Variant v) {
    return v == IJK_OMP || v == IKJ_OMP || v == BLOCKED_OMP;
}

static double time_reps(const Task *task, int reps) {
    double t0 = omp_get_wtime();
    for (int r = 0; r < reps; ++r)
        run_task(task);
    return (omp_get_wtime() - t0) / reps;
}

static int calibrate_reps(const Task *task) {
    int reps = 1;
    while (time_reps(task, reps) * reps < MIN_RUN_TIME)
        reps *= 2;
    return reps;
}

static int cmp_double(const void *x, const void *y) {
    double d = *(const double *)x - *(const double *)y;
    return (d > 0) - (d < 0);
}

static double max_rel_error(const double *C, const double *ref, size_t count) {
    double err = 0.0;
    for (size_t k = 0; k < count; ++k) {
        double e = fabs(C[k] - ref[k]) / fmax(1.0, fabs(ref[k]));
        if (!(e <= err))
            err = isnan(e) ? INFINITY : e;
    }
    return err;
}

static void fill_matrix(double *M, size_t count, unsigned *state) {
    for (size_t k = 0; k < count; ++k) {
        *state = *state * 1664525u + 1013904223u;
        M[k] = (double)(*state >> 8) / 8388608.0 - 1.0;
    }
}

static int parse_variants(const char *list, int selected[VARIANT_COUNT]) {
    if (strcmp(list, "all") == 0) {
        for (int v = 0; v < VARIANT_COUNT; ++v)
            selected[v] = 1;
        return 1;
    }
    char buf[256];
    snprintf(buf, sizeof buf, "%s", list);
    for (char *name = strtok(buf, ","); name; name = strtok(NULL, ",")) {
        int found = 0;
        for (int v = 0; v < VARIANT_COUNT; ++v)
            if (strcmp(name, VARIANT_NAMES[v]) == 0)
                selected[v] = found = 1;
        if (!found)
            return 0;
    }
    return 1;
}

int main(int argc, char *argv[]) {
    int N = argc > 1 ? atoi(argv[1]) : 512;
    int bs = argc > 2 ? atoi(argv[2]) : 32;
    int selected[VARIANT_COUNT] = {0};
    if (N <= 0 || bs <= 0 || !parse_variants(argc > 3 ? argv[3] : "all", selected)) {
        fprintf(stderr, "Usage: OMP_NUM_THREADS=<threads> %s N [block] [variants|all]\n", argv[0]);
        return 1;
    }

    size_t count = (size_t)N * N;
    double *A = malloc(count * sizeof(double));
    double *B = malloc(count * sizeof(double));
    double *C = malloc(count * sizeof(double));
    double *ref = malloc(count * sizeof(double));
    if (!A || !B || !C || !ref) {
        fprintf(stderr, "not enough memory for N=%d\n", N);
        return 1;
    }
    unsigned state = SEED;
    fill_matrix(A, count, &state);
    fill_matrix(B, count, &state);
    ijk_seq(A, B, ref, N);

    int threads = omp_get_max_threads();
    double flops = 2.0 * N * (double)N * N;
    int failed = 0;
    printf("N=%d block=%d threads=%d runs=%d (median)\n", N, bs, threads, RUNS);

    for (int v = 0; v < VARIANT_COUNT; ++v) {
        if (!selected[v])
            continue;
        Task task = {(Variant)v, N, bs, A, B, C};
        int reps = calibrate_reps(&task);
        double t[RUNS];
        double err = 0.0;
        for (int r = 0; r < RUNS; ++r) {
            memset(C, 0, count * sizeof(double));
            t[r] = time_reps(&task, reps);
            double e = max_rel_error(C, ref, count);
            if (e > err)
                err = e;
        }
        qsort(t, RUNS, sizeof(double), cmp_double);
        double time = t[RUNS / 2];
        int ok = err <= TOLERANCE;
        failed |= !ok;
        printf("%-12s N=%d block=%d threads=%d time=%.6f s GFLOPS=%.2f max_rel_err=%.1e check=%s\n",
               VARIANT_NAMES[v], N, bs, is_parallel((Variant)v) ? threads : 1,
               time, flops / time / 1e9, err, ok ? "OK" : "FAIL");
    }

    free(A);
    free(B);
    free(C);
    free(ref);
    return failed ? 2 : 0;
}
