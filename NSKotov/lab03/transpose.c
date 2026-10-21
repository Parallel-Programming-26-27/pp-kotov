#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

typedef double elem_t;

#define RUNS 5
#define MIN_RUN_TIME 0.1

static int min_int(int x, int y) {
    return x < y ? x : y;
}

void transpose_rows(const elem_t *A, elem_t *B, int N) {
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            B[(size_t)j * N + i] = A[(size_t)i * N + j];
}

void transpose_cols(const elem_t *A, elem_t *B, int N) {
    for (int j = 0; j < N; ++j)
        for (int i = 0; i < N; ++i)
            B[(size_t)j * N + i] = A[(size_t)i * N + j];
}

static void transpose_block(const elem_t *A, elem_t *B, int N, int ii, int jj, int b) {
    int i_end = min_int(ii + b, N);
    int j_end = min_int(jj + b, N);
    for (int i = ii; i < i_end; ++i)
        for (int j = jj; j < j_end; ++j)
            B[(size_t)j * N + i] = A[(size_t)i * N + j];
}

void transpose_blocked(const elem_t *A, elem_t *B, int N, int b) {
    for (int ii = 0; ii < N; ii += b)
        for (int jj = 0; jj < N; jj += b)
            transpose_block(A, B, N, ii, jj, b);
}

void transpose_rows_omp(const elem_t *A, elem_t *B, int N, int threads) {
    #pragma omp parallel for schedule(runtime) num_threads(threads)
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            B[(size_t)j * N + i] = A[(size_t)i * N + j];
}

void transpose_blocked_omp(const elem_t *A, elem_t *B, int N, int b, int threads) {
    #pragma omp parallel for schedule(static) num_threads(threads)
    for (int ii = 0; ii < N; ii += b)
        for (int jj = 0; jj < N; jj += b)
            transpose_block(A, B, N, ii, jj, b);
}

static void swap_blocks(elem_t *A, int N, int ii, int jj, int b) {
    int i_end = min_int(ii + b, N);
    int j_end = min_int(jj + b, N);
    for (int i = ii; i < i_end; ++i)
        for (int j = (ii == jj ? i + 1 : jj); j < j_end; ++j) {
            elem_t t = A[(size_t)i * N + j];
            A[(size_t)i * N + j] = A[(size_t)j * N + i];
            A[(size_t)j * N + i] = t;
        }
}

void transpose_inplace(elem_t *A, int N, int b) {
    for (int ii = 0; ii < N; ii += b)
        for (int jj = ii; jj < N; jj += b)
            swap_blocks(A, N, ii, jj, b);
}

void transpose_inplace_omp(elem_t *A, int N, int b, int threads) {
    #pragma omp parallel for schedule(dynamic, 1) num_threads(threads)
    for (int ii = 0; ii < N; ii += b)
        for (int jj = ii; jj < N; jj += b)
            swap_blocks(A, N, ii, jj, b);
}

typedef enum { ROWS, COLS, BLOCKED, ROWS_OMP, BLOCKED_OMP, INPLACE, INPLACE_OMP, UNKNOWN } Version;

static const char *VERSION_NAMES[] = {
    "rows", "cols", "blocked", "rows_omp", "blocked_omp", "inplace", "inplace_omp"
};

static Version parse_version(const char *name) {
    for (int v = ROWS; v < UNKNOWN; ++v)
        if (strcmp(name, VERSION_NAMES[v]) == 0)
            return (Version)v;
    return UNKNOWN;
}

static int is_inplace(Version v) {
    return v == INPLACE || v == INPLACE_OMP;
}

static int is_parallel(Version v) {
    return v == ROWS_OMP || v == BLOCKED_OMP || v == INPLACE_OMP;
}

typedef struct {
    Version version;
    int N, b, threads;
    const elem_t *A;
    elem_t *B;
} Task;

static void run_task(const Task *task) {
    switch (task->version) {
    case ROWS:        transpose_rows(task->A, task->B, task->N); break;
    case COLS:        transpose_cols(task->A, task->B, task->N); break;
    case BLOCKED:     transpose_blocked(task->A, task->B, task->N, task->b); break;
    case ROWS_OMP:    transpose_rows_omp(task->A, task->B, task->N, task->threads); break;
    case BLOCKED_OMP: transpose_blocked_omp(task->A, task->B, task->N, task->b, task->threads); break;
    case INPLACE:     transpose_inplace(task->B, task->N, task->b); break;
    case INPLACE_OMP: transpose_inplace_omp(task->B, task->N, task->b, task->threads); break;
    default: break;
    }
}

static void prepare(const Task *task) {
    size_t bytes = (size_t)task->N * task->N * sizeof(elem_t);
    if (is_inplace(task->version))
        memcpy(task->B, task->A, bytes);
    else
        memset(task->B, 0, bytes);
}

static double time_reps(const Task *task, int reps) {
    double t0 = omp_get_wtime();
    for (int k = 0; k < reps; ++k)
        run_task(task);
    return (omp_get_wtime() - t0) / reps;
}

static int calibrate_reps(const Task *task) {
    int reps = 1;
    prepare(task);
    while (time_reps(task, reps) * reps < MIN_RUN_TIME)
        reps *= 2;
    if (is_inplace(task->version) && reps % 2 == 0)
        ++reps;
    return reps;
}

static long long count_mismatches(const elem_t *X, const elem_t *Y, int N) {
    long long bad = 0;
    for (size_t k = 0; k < (size_t)N * N; ++k)
        bad += X[k] != Y[k];
    return bad;
}

static int cmp_double(const void *x, const void *y) {
    double d = *(const double *)x - *(const double *)y;
    return (d > 0) - (d < 0);
}

static void usage(const char *prog) {
    printf("Usage: %s N b threads version [chunk]\n"
           "  version: rows | cols | blocked | rows_omp | blocked_omp | inplace | inplace_omp\n"
           "  chunk:   rows_omp only, 0 = schedule(static) (default), k = schedule(static, k)\n",
           prog);
}

int main(int argc, char *argv[]) {
    if (argc < 5) {
        usage(argv[0]);
        return 1;
    }
    int N = atoi(argv[1]);
    int b = atoi(argv[2]);
    int threads = atoi(argv[3]);
    Version version = parse_version(argv[4]);
    int chunk = argc > 5 ? atoi(argv[5]) : 0;
    if (N <= 0 || b <= 0 || threads <= 0 || version == UNKNOWN) {
        usage(argv[0]);
        return 1;
    }
    omp_set_schedule(omp_sched_static, chunk);

    size_t count = (size_t)N * N;
    elem_t *A = malloc(count * sizeof(elem_t));
    elem_t *B = malloc(count * sizeof(elem_t));
    elem_t *ref = malloc(count * sizeof(elem_t));
    if (!A || !B || !ref) {
        fprintf(stderr, "not enough memory for N=%d\n", N);
        return 1;
    }
    for (size_t k = 0; k < count; ++k)
        A[k] = (elem_t)k;
    transpose_rows(A, ref, N);

    Task task = {version, N, b, threads, A, B};
    int reps = calibrate_reps(&task);

    double t[RUNS];
    long long mismatches = 0;
    for (int r = 0; r < RUNS; ++r) {
        prepare(&task);
        t[r] = time_reps(&task, reps);
        mismatches += count_mismatches(B, ref, N);
    }
    qsort(t, RUNS, sizeof(double), cmp_double);
    double time = t[RUNS / 2];
    double bw = 2.0 * (double)count * sizeof(elem_t) / time / 1e9;

    char schedule[32] = "";
    if (version == ROWS_OMP)
        snprintf(schedule, sizeof schedule, chunk > 0 ? " schedule=static,%d" : " schedule=static", chunk);
    printf("N=%d b=%d threads=%d version=%s%s time=%.6f s BW=%.2f GB/s check=%s\n",
           N, b, is_parallel(version) ? threads : 1, VERSION_NAMES[version], schedule,
           time, bw, mismatches == 0 ? "OK" : "FAIL");

    free(A);
    free(B);
    free(ref);
    return mismatches == 0 ? 0 : 2;
}
