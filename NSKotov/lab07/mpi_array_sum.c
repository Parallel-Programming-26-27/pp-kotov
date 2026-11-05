#include <mpi.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define RUNS 10
#define AGG_REPS 20000

enum { PHASE_SCATTER, PHASE_LOCAL, PHASE_COLLECT, PHASE_TOTAL, PHASE_COUNT };

typedef struct {
    long long gather;
    long long reduce;
    long long allreduce;
} Sums;

typedef struct {
    int use_scatterv;
    int *counts;
    int *displs;
    int *data;
    int *local;
    long long *gathered;
} Layout;

long long local_sum(const int *a, int count) {
    long long s = 0;
    for (int i = 0; i < count; ++i)
        s += a[i];
    return s;
}

static void build_layout(Layout *l, long long n, int size, int rank) {
    l->counts = malloc(size * sizeof(int));
    l->displs = malloc(size * sizeof(int));
    int base = (int)(n / size), extra = (int)(n % size), offset = 0;
    for (int r = 0; r < size; ++r) {
        l->counts[r] = base + (r < extra);
        l->displs[r] = offset;
        offset += l->counts[r];
    }
    l->local = malloc((l->counts[rank] > 0 ? l->counts[rank] : 1) * sizeof(int));
    l->data = NULL;
    l->gathered = NULL;
    if (rank == 0) {
        l->data = malloc(n * sizeof(int));
        for (long long i = 0; i < n; ++i)
            l->data[i] = (int)(i + 1);
        l->gathered = malloc(size * sizeof(long long));
    }
}

static void free_layout(Layout *l) {
    free(l->counts);
    free(l->displs);
    free(l->local);
    free(l->data);
    free(l->gathered);
}

static long long gather_and_add(long long local, long long *gathered, int rank, int size) {
    MPI_Gather(&local, 1, MPI_LONG_LONG_INT, gathered, 1, MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);
    long long total = 0;
    if (rank == 0)
        for (int r = 0; r < size; ++r)
            total += gathered[r];
    return total;
}

static long long array_sum(const Layout *l, int rank, int size, Sums *s, double *phase) {
    int count = l->counts[rank];
    double t0 = MPI_Wtime();
    if (l->use_scatterv)
        MPI_Scatterv(l->data, l->counts, l->displs, MPI_INT, l->local, count, MPI_INT, 0, MPI_COMM_WORLD);
    else
        MPI_Scatter(l->data, count, MPI_INT, l->local, count, MPI_INT, 0, MPI_COMM_WORLD);
    double t1 = MPI_Wtime();

    long long local = local_sum(l->local, count);
    double t2 = MPI_Wtime();

    s->gather = gather_and_add(local, l->gathered, rank, size);
    MPI_Reduce(&local, &s->reduce, 1, MPI_LONG_LONG_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Allreduce(&local, &s->allreduce, 1, MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD);
    double t3 = MPI_Wtime();

    phase[PHASE_SCATTER] = t1 - t0;
    phase[PHASE_LOCAL] = t2 - t1;
    phase[PHASE_COLLECT] = t3 - t2;
    phase[PHASE_TOTAL] = t3 - t0;
    return local;
}

static void measure_pipeline(const Layout *l, int rank, int size, double *mean) {
    Sums s;
    double phase[PHASE_COUNT], phase_max[PHASE_COUNT];
    memset(mean, 0, PHASE_COUNT * sizeof(double));
    array_sum(l, rank, size, &s, phase);
    for (int run = 0; run < RUNS; ++run) {
        MPI_Barrier(MPI_COMM_WORLD);
        array_sum(l, rank, size, &s, phase);
        MPI_Reduce(phase, phase_max, PHASE_COUNT, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        for (int k = 0; k < PHASE_COUNT; ++k)
            mean[k] += phase_max[k] / RUNS;
    }
}

static double time_aggregation(int use_reduce, long long local, long long *gathered, int rank, int size,
                               long long *result) {
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    for (int r = 0; r < AGG_REPS; ++r) {
        if (use_reduce)
            MPI_Reduce(&local, result, 1, MPI_LONG_LONG_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        else
            *result = gather_and_add(local, gathered, rank, size);
    }
    double t = (MPI_Wtime() - t0) / AGG_REPS, t_max;
    MPI_Reduce(&t, &t_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    return t_max;
}

/* Доп. задание: общий результат нужен каждому процессу. По сумме S из MPI_Allreduce
   каждый процесс сам находит среднее S/n и свою долю суммы и без лишних обменов
   считает вклад своих элементов в дисперсию; итог собирает MPI_Reduce. */
static void use_total_everywhere(const Layout *l, long long n, long long total, long long expected,
                                 long long local, int rank) {
    int count = l->counts[rank];
    double mean = (double)total / (double)n;
    double local_sq = 0.0;
    for (int i = 0; i < count; ++i) {
        double d = l->local[i] - mean;
        local_sq += d * d;
    }
    int ok = total == expected, all_ok;
    double sq;
    MPI_Reduce(&ok, &all_ok, 1, MPI_INT, MPI_LAND, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_sq, &sq, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        double var = sq / (double)n;
        double var_expected = ((double)n * (double)n - 1.0) / 12.0;
        printf("allreduce use: all ranks got S=%lld: %s; rank 0 share=%.4f%%; mean=%.1f; variance=%.6e "
               "(expected (n^2-1)/12=%.6e, rel.err=%.1e)\n",
               expected, all_ok ? "yes" : "no", 100.0 * (double)local / (double)total, mean, var, var_expected,
               (var - var_expected) / var_expected);
    }
}

int main(int argc, char **argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    long long params[2] = {0, 0};
    if (rank == 0) {
        params[0] = argc > 1 ? atoll(argv[1]) : 0;
        params[1] = argc > 2 && strcmp(argv[2], "scatterv") == 0;
    }
    MPI_Bcast(params, 2, MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);
    long long n = params[0];
    Layout l = {.use_scatterv = (int)params[1]};

    const char *error = NULL;
    if (n <= 0 || n > INT_MAX)
        error = "n must be in [1, INT_MAX]";
    else if (!l.use_scatterv && n % size != 0)
        error = "n must be divisible by the number of processes (or pass 'scatterv')";
    if (error) {
        if (rank == 0)
            fprintf(stderr, "error: %s; got n=%lld, p=%d\nusage: %s N [scatterv]\n", error, n, size, argv[0]);
        MPI_Finalize();
        return 1;
    }

    build_layout(&l, n, size, rank);
    long long expected = n * (n + 1) / 2;
    Sums s;
    double phase[PHASE_COUNT];
    long long local = array_sum(&l, rank, size, &s, phase);

    if (rank == 0) {
        printf("p=%d n=%lld distribution=%s counts=%d..%d\n", size, n, l.use_scatterv ? "Scatterv" : "Scatter",
               l.counts[size - 1], l.counts[0]);
        printf("expected=%lld gather=%lld reduce=%lld allreduce(rank 0)=%lld correct=%s\n", expected, s.gather,
               s.reduce, s.allreduce,
               s.gather == expected && s.reduce == expected && s.allreduce == expected ? "yes" : "no");
    }
    use_total_everywhere(&l, n, s.allreduce, expected, local, rank);

    double mean[PHASE_COUNT];
    measure_pipeline(&l, rank, size, mean);
    if (rank == 0)
        printf("time (mean of %d, max over ranks): total=%.3e s scatter=%.3e local=%.3e collectives=%.3e\n", RUNS,
               mean[PHASE_TOTAL], mean[PHASE_SCATTER], mean[PHASE_LOCAL], mean[PHASE_COLLECT]);

    long long agg_gather, agg_reduce;
    double t_gather = time_aggregation(0, local, l.gathered, rank, size, &agg_gather);
    double t_reduce = time_aggregation(1, local, l.gathered, rank, size, &agg_reduce);
    if (rank == 0)
        printf("aggregation (%d reps): gather+sum=%.3e s (%s) reduce=%.3e s (%s)\n", AGG_REPS, t_gather,
               agg_gather == expected ? "ok" : "wrong", t_reduce, agg_reduce == expected ? "ok" : "wrong");

    free_layout(&l);
    MPI_Finalize();
    return 0;
}
