#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <string.h>

double pi_sequential(long long n) {
    double sum = 0.0;
    for (long long i = 0; i < n; ++i) {
        double term = 1.0 / (2.0 * (double)i + 1.0);
        sum += (i & 1) ? -term : term;
    }
    return 4.0 * sum;
}

double pi_parallel(long long n, int threads) {
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum) num_threads(threads) schedule(static)
    for (long long i = 0; i < n; ++i) {
        double term = 1.0 / (2.0 * (double)i + 1.0);
        sum += (i & 1) ? -term : term;
    }
    return 4.0 * sum;
}

static void print_run(const char *label, double pi, double t, int threads) {
    printf("%-11s pi=%.15f  error=%.3e  time=%.4f s  threads=%d\n",
           label, pi, fabs(pi - M_PI), t, threads);
}

static void experiment1(void) {
    long long n = 1000000000LL;
    int tlist[] = {1, 2, 4, 8, 16};
    int nt = sizeof(tlist) / sizeof(tlist[0]);

    printf("=== Experiment 1: n = 1e9 ===\n");

    double t0 = omp_get_wtime();
    double pi1 = pi_sequential(n);
    double t_seq = omp_get_wtime() - t0;
    print_run("Sequential", pi1, t_seq, 1);

    printf("\nThreads\tTime(s)\tSpeedup\n");
    for (int k = 0; k < nt; ++k) {
        int th = tlist[k];
        t0 = omp_get_wtime();
        double pi2 = pi_parallel(n, th);
        double tp = omp_get_wtime() - t0;
        printf("%d\t%.4f\t%.2f\n", th, tp, t_seq / tp);
    }
    printf("\n");
}

static void experiment2(void) {
    long long nlist[] = {10000000LL, 100000000LL, 1000000000LL, 10000000000LL};
    int nn = sizeof(nlist) / sizeof(nlist[0]);
    int threads = 8;

    printf("=== Experiment 2: threads = %d ===\n", threads);
    printf("n\t\tSeq(s)\tPar(s)\tSpeedup\n");

    for (int k = 0; k < nn; ++k) {
        long long n = nlist[k];

        double t0 = omp_get_wtime();
        double pi1 = pi_sequential(n);
        double t_seq = omp_get_wtime() - t0;

        t0 = omp_get_wtime();
        double pi2 = pi_parallel(n, threads);
        double t_par = omp_get_wtime() - t0;

        printf("%-13lld\t%.4f\t%.4f\t%.2f\n",
               n, t_seq, t_par, t_seq / t_par);
    }
    printf("\n");
}

int main(int argc, char *argv[]) {

    experiment1();

    experiment2();

    long long n = 1000000000LL;
    int threads = omp_get_max_threads();

    if (argc > 1) n = atoll(argv[1]);
    if (argc > 2) threads = atoi(argv[2]);

    printf("n = %lld, threads = %d\n\n", n, threads);

    double t0 = omp_get_wtime();
    double pi1 = pi_sequential(n);
    double t_seq = omp_get_wtime() - t0;
    print_run("Sequential", pi1, t_seq, 1);

    t0 = omp_get_wtime();
    double pi2 = pi_parallel(n, threads);
    double t_par = omp_get_wtime() - t0;
    print_run("Parallel", pi2, t_par, threads);

    printf("Speedup = %.2f\n", t_seq / t_par);
    return 0;
}