#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TAG_MESSAGE 1
#define TAG_REPLY 2
#define MSG_LEN 64
#define WARMUP 100
#define RUNS 5

static const int BENCH_SIZES[] = {1, 1024, 65536, 1048576};
static const int BENCH_REPS[] = {100000, 100000, 10000, 1000};

static void send_text(const char *text, int dest, int tag) {
    MPI_Send(text, (int)strlen(text) + 1, MPI_CHAR, dest, tag, MPI_COMM_WORLD);
}

void basic_exchange(int rank) {
    char msg[MSG_LEN];
    MPI_Status status;

    if (rank == 0) {
        snprintf(msg, MSG_LEN, "Hello from rank 0");
        printf("[rank 0] -> rank 1: \"%s\" (MPI_CHAR, count=%zu, tag=%d)\n", msg, strlen(msg) + 1, TAG_MESSAGE);
        send_text(msg, 1, TAG_MESSAGE);

        MPI_Recv(msg, MSG_LEN, MPI_CHAR, 1, TAG_REPLY, MPI_COMM_WORLD, &status);
        printf("[rank 0] <- rank %d: \"%s\" (tag=%d)\n", status.MPI_SOURCE, msg, status.MPI_TAG);
        printf("[rank 0] basic exchange finished\n");
    } else if (rank == 1) {
        MPI_Recv(msg, MSG_LEN, MPI_CHAR, 0, TAG_MESSAGE, MPI_COMM_WORLD, &status);
        printf("[rank 1] <- rank %d: \"%s\" (tag=%d)\n", status.MPI_SOURCE, msg, status.MPI_TAG);

        snprintf(msg, MSG_LEN, "Message received");
        printf("[rank 1] -> rank 0: \"%s\" (MPI_CHAR, count=%zu, tag=%d)\n", msg, strlen(msg) + 1, TAG_REPLY);
        send_text(msg, 0, TAG_REPLY);
    }
}

void exchange_with_all(int rank, int size) {
    char msg[MSG_LEN];
    char expected[MSG_LEN];
    MPI_Status status;

    if (rank == 0) {
        for (int dest = 1; dest < size; ++dest) {
            snprintf(msg, MSG_LEN, "Message for rank %d", dest);
            printf("[rank 0] -> rank %d: \"%s\" (tag=%d)\n", dest, msg, TAG_MESSAGE);
            send_text(msg, dest, TAG_MESSAGE);
        }

        int *replied = calloc(size, sizeof(int));
        int received = 0, correct = 0;
        for (int i = 1; i < size; ++i) {
            MPI_Recv(msg, MSG_LEN, MPI_CHAR, MPI_ANY_SOURCE, TAG_REPLY, MPI_COMM_WORLD, &status);
            int src = status.MPI_SOURCE;
            snprintf(expected, MSG_LEN, "Rank %d: got own message", src);
            int ok = src > 0 && src < size && !replied[src] && strcmp(msg, expected) == 0;
            replied[src] = 1;
            ++received;
            correct += ok;
            printf("[rank 0] <- rank %d: \"%s\" (tag=%d) %s\n", src, msg, status.MPI_TAG, ok ? "OK" : "WRONG");
        }
        printf("[rank 0] processes=%d sent=%d replies=%d correct=%d all_ok=%s\n",
               size, size - 1, received, correct, correct == size - 1 ? "yes" : "no");
        free(replied);
    } else {
        MPI_Recv(msg, MSG_LEN, MPI_CHAR, 0, TAG_MESSAGE, MPI_COMM_WORLD, &status);
        snprintf(expected, MSG_LEN, "Message for rank %d", rank);
        int own = strcmp(msg, expected) == 0;
        printf("[rank %d] <- rank %d: \"%s\" (tag=%d)\n", rank, status.MPI_SOURCE, msg, status.MPI_TAG);

        if (own)
            snprintf(msg, MSG_LEN, "Rank %d: got own message", rank);
        else
            snprintf(msg, MSG_LEN, "Rank %d: got foreign message", rank);
        printf("[rank %d] -> rank 0: \"%s\" (tag=%d)\n", rank, msg, TAG_REPLY);
        send_text(msg, 0, TAG_REPLY);
    }
}

static int check_count(const MPI_Status *status, int bytes) {
    int count;
    MPI_Get_count(status, MPI_CHAR, &count);
    return count == bytes;
}

static double ping_pong(int rank, char *buf, int bytes, int reps, int *ok) {
    MPI_Status status;
    double t0 = MPI_Wtime();
    for (int r = 0; r < reps; ++r) {
        if (rank == 0) {
            MPI_Send(buf, bytes, MPI_CHAR, 1, TAG_MESSAGE, MPI_COMM_WORLD);
            MPI_Recv(buf, bytes, MPI_CHAR, 1, TAG_REPLY, MPI_COMM_WORLD, &status);
        } else {
            MPI_Recv(buf, bytes, MPI_CHAR, 0, TAG_MESSAGE, MPI_COMM_WORLD, &status);
            MPI_Send(buf, bytes, MPI_CHAR, 0, TAG_REPLY, MPI_COMM_WORLD);
        }
        *ok &= check_count(&status, bytes);
    }
    return (MPI_Wtime() - t0) / reps;
}

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

void bench(int rank) {
    int count = sizeof(BENCH_SIZES) / sizeof(BENCH_SIZES[0]);
    int max_bytes = BENCH_SIZES[count - 1];
    char *buf = malloc(max_bytes);
    memset(buf, 'x', max_bytes);

    if (rank == 0)
        printf("bytes\treps\tround_trip_s\tround_trip_us\tbandwidth_MBps\tsize_ok\n");
    for (int k = 0; k < count; ++k) {
        int bytes = BENCH_SIZES[k], reps = BENCH_REPS[k], ok = 1;
        double times[RUNS];
        ping_pong(rank, buf, bytes, WARMUP, &ok);
        for (int run = 0; run < RUNS; ++run)
            times[run] = ping_pong(rank, buf, bytes, reps, &ok);
        qsort(times, RUNS, sizeof(double), cmp_double);
        double t = times[RUNS / 2];
        if (rank == 0)
            printf("%d\t%d\t%.3e\t%.2f\t%.1f\t%s\n", bytes, reps, t, t * 1e6, 2.0 * bytes / t / 1e6, ok ? "yes" : "no");
    }
    free(buf);
}

int main(int argc, char **argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    setvbuf(stdout, NULL, _IOLBF, 0);

    if (size < 2) {
        if (rank == 0)
            fprintf(stderr, "error: at least 2 processes required, got %d (run: mpiexec -n 2 %s)\n", size, argv[0]);
        MPI_Finalize();
        return 1;
    }

    const char *mode = argc > 1 ? argv[1] : "basic";
    if (strcmp(mode, "basic") == 0) {
        basic_exchange(rank);
    } else if (strcmp(mode, "all") == 0) {
        exchange_with_all(rank, size);
    } else if (strcmp(mode, "bench") == 0) {
        if (rank < 2)
            bench(rank);
    } else if (rank == 0) {
        fprintf(stderr, "usage: %s [basic|all|bench]\n", argv[0]);
    }

    MPI_Finalize();
    return 0;
}
