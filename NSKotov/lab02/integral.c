#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#define RUNS 5
#define SEQUENTIAL 0
#define START_N 16
#define MIN_RUN_TIME 0.1

double f(double x) {
    return (x * x - 1.0) * pow(10.0, -2.0 * x);
}

double exact_integral(void) {
    double c = 2.0 * M_LN10;
    return 2.0 / (c * c * c) - 1.0 / c - exp(-c) * (2.0 / (c * c) + 2.0 / (c * c * c));
}

double integrate_seq(double a, double b, long long n) {
    double h = (b - a) / (double)n;
    double sum = 0.0;
    for (long long i = 0; i < n; ++i)
        sum += f(a + ((double)i + 0.5) * h);
    return h * sum;
}

double integrate_omp(double a, double b, long long n, int threads) {
    double h = (b - a) / (double)n;
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum) num_threads(threads) schedule(static)
    for (long long i = 0; i < n; ++i)
        sum += f(a + ((double)i + 0.5) * h);
    return h * sum;
}

static double integrate(double a, double b, long long n, int threads) {
    return threads == SEQUENTIAL ? integrate_seq(a, b, n) : integrate_omp(a, b, n, threads);
}

long long integrate_eps(double a, double b, double eps, int threads, double *result) {
    long long n = START_N;
    double prev = integrate(a, b, n, threads);
    double cur;
    for (;;) {
        n *= 2;
        cur = integrate(a, b, n, threads);
        if (fabs(cur - prev) < eps)
            break;
        prev = cur;
    }
    *result = cur;
    return n;
}

typedef struct {
    double a, b, eps;
    long long n;
    int threads;
    double value;
} Task;

static void run_task(Task *task) {
    if (task->eps > 0.0)
        task->n = integrate_eps(task->a, task->b, task->eps, task->threads, &task->value);
    else
        task->value = integrate(task->a, task->b, task->n, task->threads);
}

static double time_reps(Task *task, int reps) {
    double t0 = omp_get_wtime();
    for (int k = 0; k < reps; ++k)
        run_task(task);
    return (omp_get_wtime() - t0) / reps;
}

static int calibrate_reps(Task *task) {
    int reps = 1;
    while (time_reps(task, reps) * reps < MIN_RUN_TIME)
        reps *= 2;
    return reps;
}

static int cmp_double(const void *x, const void *y) {
    double d = *(const double *)x - *(const double *)y;
    return (d > 0) - (d < 0);
}

static double measure(Task *task) {
    double t[RUNS];
    int reps = calibrate_reps(task);
    for (int r = 0; r < RUNS; ++r)
        t[r] = time_reps(task, reps);
    qsort(t, RUNS, sizeof(double), cmp_double);
    return t[RUNS / 2];
}

static void experiment1(void) {
    int tlist[] = {1, 2, 4, 8, 16};
    int nt = sizeof(tlist) / sizeof(tlist[0]);
    Task task = {0.0, 1.0, 0.0, 100000000LL, SEQUENTIAL, 0.0};

    printf("=== Experiment 1: n = 1e8, [0; 1] ===\n");
    double t_seq = measure(&task);
    printf("Sequential: I=%.15f  time=%.4f s  threads=1\n\n", task.value, t_seq);

    printf("Threads\tTime(s)\tSpeedup\tI\n");
    for (int k = 0; k < nt; ++k) {
        task.threads = tlist[k];
        double tp = measure(&task);
        printf("%d\t%.4f\t%.2f\t%.15f\n", tlist[k], tp, t_seq / tp, task.value);
    }
    printf("\n");
}

static void experiment2(void) {
    int threads = 8;
    double exact = exact_integral();

    printf("=== Experiment 2: threads = %d, exact I = %.15f ===\n", threads, exact);
    printf("n\t\tI (OpenMP)\t\tError\t\tSeq(s)\t\tPar(s)\t\tSpeedup\n");
    for (long long n = 10000LL; n <= 100000000LL; n *= 10) {
        Task seq = {0.0, 1.0, 0.0, n, SEQUENTIAL, 0.0};
        Task par = {0.0, 1.0, 0.0, n, threads, 0.0};
        double t_seq = measure(&seq);
        double t_par = measure(&par);
        printf("%-10lld\t%.15f\t%.3e\t%.6f\t%.6f\t%.2f\n",
               n, par.value, fabs(par.value - exact), t_seq, t_par, t_seq / t_par);
    }
    printf("\n");
}

static void experiment3(double eps, int threads) {
    Task seq = {0.0, 1.0, eps, 0, SEQUENTIAL, 0.0};
    Task par = {0.0, 1.0, eps, 0, threads, 0.0};
    double exact = exact_integral();

    printf("=== Experiment 3: eps = %.0e, Runge criterion |I_2n - I_n| < eps ===\n", eps);
    double t_seq = measure(&seq);
    double t_par = measure(&par);
    printf("Sequential: n=%lld  I=%.15f  error=%.3e  time=%.6f s  threads=1\n",
           seq.n, seq.value, fabs(seq.value - exact), t_seq);
    printf("OpenMP:     n=%lld  I=%.15f  error=%.3e  time=%.6f s  threads=%d\n\n",
           par.n, par.value, fabs(par.value - exact), t_par, threads);
}

static void usage(const char *prog) {
    printf("Usage:\n"
           "  %s                          run experiments 1-3\n"
           "  %s run a b n threads        sequential and OpenMP run\n"
           "  %s eps a b eps threads      run with given accuracy (Runge rule)\n",
           prog, prog, prog);
}

int main(int argc, char *argv[]) {
    if (argc == 1) {
        experiment1();
        experiment2();
        experiment3(1e-8, 8);
        return 0;
    }
    if (argc == 6 && (strcmp(argv[1], "run") == 0 || strcmp(argv[1], "eps") == 0)) {
        int eps_mode = strcmp(argv[1], "eps") == 0;
        double a = atof(argv[2]), b = atof(argv[3]);
        double eps = eps_mode ? atof(argv[4]) : 0.0;
        long long n = eps_mode ? 0 : atoll(argv[4]);
        int threads = atoi(argv[5]);
        Task seq = {a, b, eps, n, SEQUENTIAL, 0.0};
        Task par = {a, b, eps, n, threads, 0.0};

        double t_seq = measure(&seq);
        printf("Sequential: n=%lld  I=%.15f  time=%.6f s  threads=1\n", seq.n, seq.value, t_seq);
        double t_par = measure(&par);
        printf("OpenMP:     n=%lld  I=%.15f  time=%.6f s  threads=%d\n", par.n, par.value, t_par, threads);
        printf("Speedup = %.2f\n", t_seq / t_par);
        return 0;
    }
    usage(argv[0]);
    return 1;
}
