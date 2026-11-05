#include <mpi.h>
#include <omp.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DEFAULT_RUNS 5
#define INITIAL_LIST_CAP 1024

typedef enum { EXCHANGE_ALLREDUCE, EXCHANGE_ALLGATHERV } Exchange;

typedef struct {
    const char *path;
    int source;
    int directed;
    omp_sched_t schedule;
    int chunk;
    Exchange exchange;
    int runs;
    int stats;
} Params;

typedef struct {
    int n;
    long long m;
    int *offsets;
    int *edges;
} Graph;

typedef struct {
    _Alignas(64) int *data;
    int size;
    int cap;
} CandidateList;

typedef struct {
    int rank, nprocs, nthreads;
    int *frontier;
    int *next;
    int *local_candidate;
    int *global_candidate;
    CandidateList *lists;
    int *local_list;
    int *counts;
    int *displs;
    int *gathered;
    int gathered_cap;
} Workspace;

typedef struct {
    int levels;
    long long processed;
    long long edges_scanned;
    double comm_time;
    int *frontier_sizes;
} Stats;

static int read_number(FILE *f, long long *out) {
    int c;
    do c = getc_unlocked(f); while (c == ' ' || c == '\n' || c == '\r' || c == '\t');
    if (c < '0' || c > '9')
        return 0;
    long long v = 0;
    while (c >= '0' && c <= '9') {
        v = v * 10 + (c - '0');
        c = getc_unlocked(f);
    }
    *out = v;
    return 1;
}

static const char *load_graph(const char *path, int directed, Graph *g) {
    FILE *f = fopen(path, "r");
    if (!f)
        return "cannot open graph file";
    long long n, m;
    if (!read_number(f, &n) || !read_number(f, &m) || n <= 0 || n > INT_MAX - 1 || m < 0) {
        fclose(f);
        return "bad header, expected 'N M'";
    }
    long long stored = directed ? m : 2 * m;
    if (stored > INT_MAX) {
        fclose(f);
        return "too many edges for int CSR";
    }
    int *src = malloc((m > 0 ? m : 1) * sizeof(int));
    int *dst = malloc((m > 0 ? m : 1) * sizeof(int));
    g->n = (int)n;
    g->m = m;
    g->offsets = calloc(n + 1, sizeof(int));
    g->edges = malloc((stored > 0 ? stored : 1) * sizeof(int));
    for (long long i = 0; i < m; ++i) {
        long long u, v;
        if (!read_number(f, &u) || !read_number(f, &v) || u >= n || v >= n) {
            fclose(f);
            free(src);
            free(dst);
            return "bad edge line or vertex out of range";
        }
        src[i] = (int)u;
        dst[i] = (int)v;
        ++g->offsets[u + 1];
        if (!directed)
            ++g->offsets[v + 1];
    }
    fclose(f);
    for (int v = 0; v < g->n; ++v)
        g->offsets[v + 1] += g->offsets[v];
    int *pos = malloc(n * sizeof(int));
    memcpy(pos, g->offsets, n * sizeof(int));
    for (long long i = 0; i < m; ++i) {
        g->edges[pos[src[i]]++] = dst[i];
        if (!directed)
            g->edges[pos[dst[i]]++] = src[i];
    }
    free(pos);
    free(src);
    free(dst);
    return NULL;
}

int bfs_sequential(const Graph *g, int source, int *dist) {
    int *queue = malloc(g->n * sizeof(int));
    for (int v = 0; v < g->n; ++v)
        dist[v] = -1;
    int head = 0, tail = 0, levels = 0;
    dist[source] = 0;
    queue[tail++] = source;
    while (head < tail) {
        int v = queue[head++];
        if (dist[v] + 1 > levels)
            levels = dist[v] + 1;
        for (int j = g->offsets[v]; j < g->offsets[v + 1]; ++j) {
            int u = g->edges[j];
            if (dist[u] == -1) {
                dist[u] = dist[v] + 1;
                queue[tail++] = u;
            }
        }
    }
    free(queue);
    return levels;
}

static void push_candidate(CandidateList *list, int u) {
    if (list->size == list->cap) {
        list->cap *= 2;
        list->data = realloc(list->data, list->cap * sizeof(int));
    }
    list->data[list->size++] = u;
}

static void init_workspace(Workspace *w, int n) {
    MPI_Comm_rank(MPI_COMM_WORLD, &w->rank);
    MPI_Comm_size(MPI_COMM_WORLD, &w->nprocs);
    w->nthreads = omp_get_max_threads();
    w->frontier = malloc(n * sizeof(int));
    w->next = malloc(n * sizeof(int));
    w->local_candidate = calloc(n, sizeof(int));
    w->global_candidate = malloc(n * sizeof(int));
    w->lists = aligned_alloc(64, w->nthreads * sizeof(CandidateList));
    for (int t = 0; t < w->nthreads; ++t) {
        w->lists[t].cap = INITIAL_LIST_CAP;
        w->lists[t].size = 0;
        w->lists[t].data = malloc(INITIAL_LIST_CAP * sizeof(int));
    }
    w->local_list = malloc(n * sizeof(int));
    w->counts = malloc(w->nprocs * sizeof(int));
    w->displs = malloc(w->nprocs * sizeof(int));
    w->gathered_cap = n;
    w->gathered = malloc(n * sizeof(int));
}

static void free_workspace(Workspace *w) {
    for (int t = 0; t < w->nthreads; ++t)
        free(w->lists[t].data);
    free(w->lists);
    free(w->frontier);
    free(w->next);
    free(w->local_candidate);
    free(w->global_candidate);
    free(w->local_list);
    free(w->counts);
    free(w->displs);
    free(w->gathered);
}

/* Обмен плотным массивом признаков: local_candidate[u] = 1, если вершину u нашёл
   этот процесс; MPI_Allreduce(MPI_MAX) даёт всем одинаковый global_candidate. */
static int exchange_dense(const Graph *g, Workspace *w, int *dist, int level, double *comm_time) {
    int n = g->n;
    #pragma omp parallel
    {
        #pragma omp for schedule(static)
        for (int u = 0; u < n; ++u)
            w->local_candidate[u] = 0;
        const CandidateList *list = &w->lists[omp_get_thread_num()];
        for (int i = 0; i < list->size; ++i) {
            #pragma omp atomic write
            w->local_candidate[list->data[i]] = 1;
        }
    }
    double t0 = MPI_Wtime();
    MPI_Allreduce(w->local_candidate, w->global_candidate, n, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    *comm_time += MPI_Wtime() - t0;

    int next_size = 0;
    for (int u = 0; u < n; ++u)
        if (w->global_candidate[u] == 1 && dist[u] == -1) {
            dist[u] = level + 1;
            w->next[next_size++] = u;
        }
    return next_size;
}

/* Доп. задание: обмен только списками найденных вершин через MPI_Allgatherv. */
static int exchange_lists(Workspace *w, int *dist, int level, double *comm_time) {
    int count = 0;
    for (int t = 0; t < w->nthreads; ++t) {
        const CandidateList *list = &w->lists[t];
        for (int i = 0; i < list->size; ++i) {
            int u = list->data[i];
            if (!w->local_candidate[u]) {
                w->local_candidate[u] = 1;
                w->local_list[count++] = u;
            }
        }
    }
    for (int i = 0; i < count; ++i)
        w->local_candidate[w->local_list[i]] = 0;

    double t0 = MPI_Wtime();
    MPI_Allgather(&count, 1, MPI_INT, w->counts, 1, MPI_INT, MPI_COMM_WORLD);
    long long total = 0;
    for (int r = 0; r < w->nprocs; ++r) {
        w->displs[r] = (int)total;
        total += w->counts[r];
    }
    if (total > w->gathered_cap) {
        w->gathered_cap = (int)total;
        w->gathered = realloc(w->gathered, total * sizeof(int));
    }
    MPI_Allgatherv(w->local_list, count, MPI_INT, w->gathered, w->counts, w->displs, MPI_INT, MPI_COMM_WORLD);
    *comm_time += MPI_Wtime() - t0;

    int next_size = 0;
    for (long long i = 0; i < total; ++i) {
        int u = w->gathered[i];
        if (dist[u] == -1) {
            dist[u] = level + 1;
            w->next[next_size++] = u;
        }
    }
    return next_size;
}

/* Процесс rank берёт из текущего фронта индексы rank, rank + nprocs, ...;
   потоки пишут найденные вершины только в свои списки, dist в цикле лишь читается. */
int bfs_hybrid(const Graph *g, int source, int *dist, Workspace *w, Exchange exchange, Stats *st) {
    int n = g->n, rank = w->rank, nprocs = w->nprocs;
    const int *offsets = g->offsets, *edges = g->edges;
    #pragma omp parallel for schedule(static)
    for (int v = 0; v < n; ++v)
        dist[v] = -1;

    dist[source] = 0;
    w->frontier[0] = source;
    int frontier_size = 1, level = 0;
    st->processed = 0;
    st->edges_scanned = 0;
    st->comm_time = 0.0;

    while (frontier_size > 0) {
        if (st->frontier_sizes)
            st->frontier_sizes[level] = frontier_size;
        int local_frontier_size = rank < frontier_size ? (frontier_size - rank + nprocs - 1) / nprocs : 0;
        const int *frontier = w->frontier;
        long long scanned = 0;

        #pragma omp parallel reduction(+:scanned)
        {
            CandidateList *list = &w->lists[omp_get_thread_num()];
            list->size = 0;
            #pragma omp for schedule(runtime)
            for (int k = 0; k < local_frontier_size; ++k) {
                int v = frontier[rank + k * nprocs];
                for (int j = offsets[v]; j < offsets[v + 1]; ++j) {
                    int u = edges[j];
                    if (dist[u] == -1)
                        push_candidate(list, u);
                }
                scanned += offsets[v + 1] - offsets[v];
            }
        }
        st->processed += local_frontier_size;
        st->edges_scanned += scanned;

        if (exchange == EXCHANGE_ALLREDUCE)
            frontier_size = exchange_dense(g, w, dist, level, &st->comm_time);
        else
            frontier_size = exchange_lists(w, dist, level, &st->comm_time);

        int *tmp = w->frontier;
        w->frontier = w->next;
        w->next = tmp;
        ++level;
    }
    st->levels = level;
    return level;
}

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static double median(double *v, int count) {
    qsort(v, count, sizeof(double), cmp_double);
    return count % 2 ? v[count / 2] : 0.5 * (v[count / 2 - 1] + v[count / 2]);
}

static unsigned long long dist_checksum(const int *dist, int n) {
    unsigned long long h = 1469598103934665603ULL;
    for (int v = 0; v < n; ++v)
        h = (h ^ (unsigned long long)(dist[v] + 2)) * 1099511628211ULL;
    return h;
}

static int is_key(const char *arg, size_t len, const char *key) {
    return strlen(key) == len && strncmp(arg, key, len) == 0;
}

static const char *parse_params(int argc, char **argv, Params *p) {
    if (argc < 3)
        return "usage: bfs <graph.txt> <source> [directed=0|1] [schedule=static|dynamic|guided] [chunk=K] "
               "[exchange=allreduce|allgatherv] [runs=R] [stats=0|1]";
    p->path = argv[1];
    p->source = atoi(argv[2]);
    p->directed = 0;
    p->schedule = omp_sched_dynamic;
    p->chunk = 64;
    p->exchange = EXCHANGE_ALLREDUCE;
    p->runs = DEFAULT_RUNS;
    p->stats = 0;
    for (int i = 3; i < argc; ++i) {
        const char *eq = strchr(argv[i], '=');
        if (!eq)
            return "options must look like key=value";
        const char *val = eq + 1;
        size_t len = eq - argv[i];
        if (is_key(argv[i], len, "directed"))
            p->directed = atoi(val);
        else if (is_key(argv[i], len, "schedule"))
            p->schedule = strcmp(val, "static") == 0 ? omp_sched_static
                        : strcmp(val, "guided") == 0 ? omp_sched_guided : omp_sched_dynamic;
        else if (is_key(argv[i], len, "chunk"))
            p->chunk = atoi(val);
        else if (is_key(argv[i], len, "exchange"))
            p->exchange = strcmp(val, "allgatherv") == 0 ? EXCHANGE_ALLGATHERV : EXCHANGE_ALLREDUCE;
        else if (is_key(argv[i], len, "runs"))
            p->runs = atoi(val) > 0 ? atoi(val) : 1;
        else if (is_key(argv[i], len, "stats"))
            p->stats = atoi(val);
        else
            return "unknown option";
    }
    return NULL;
}

static const char *schedule_name(omp_sched_t s) {
    return s == omp_sched_static ? "static" : s == omp_sched_guided ? "guided" : "dynamic";
}

static void print_balance(const Stats *st, const Workspace *w) {
    long long mine[2] = {st->processed, st->edges_scanned};
    long long *all = w->rank == 0 ? malloc(2 * w->nprocs * sizeof(long long)) : NULL;
    MPI_Gather(mine, 2, MPI_LONG_LONG_INT, all, 2, MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);
    if (w->rank != 0)
        return;
    long long vmin = LLONG_MAX, vmax = 0, emin = LLONG_MAX, emax = 0, esum = 0;
    printf("per-rank processed vertices / scanned edges:");
    for (int r = 0; r < w->nprocs; ++r) {
        long long v = all[2 * r], e = all[2 * r + 1];
        printf(" %lld/%lld", v, e);
        vmin = v < vmin ? v : vmin;
        vmax = v > vmax ? v : vmax;
        emin = e < emin ? e : emin;
        emax = e > emax ? e : emax;
        esum += e;
    }
    printf("\nbalance: vertices min=%lld max=%lld, edges min=%lld max=%lld, edges max/avg=%.3f\n", vmin, vmax, emin,
           emax, esum > 0 ? (double)emax * w->nprocs / (double)esum : 1.0);
    free(all);
}

int main(int argc, char **argv) {
    int provided, rank, nprocs;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    Params p = {0};
    Graph g = {0};
    double t_prep = MPI_Wtime();
    const char *error = parse_params(argc, argv, &p);
    if (!error)
        error = load_graph(p.path, p.directed, &g);
    if (!error && (p.source < 0 || p.source >= g.n))
        error = "source vertex out of range";
    t_prep = MPI_Wtime() - t_prep;

    int ok = error == NULL, all_ok;
    MPI_Allreduce(&ok, &all_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (!all_ok) {
        if (rank == 0)
            fprintf(stderr, "error: %s\n", error ? error : "graph loading failed on another rank");
        MPI_Finalize();
        return 1;
    }
    omp_set_schedule(p.schedule, p.chunk);

    double t_prep_max;
    MPI_Reduce(&t_prep, &t_prep_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    int *dist_seq = NULL, levels_seq = 0;
    double t_seq = 0.0;
    double *times = malloc(p.runs * sizeof(double));
    if (rank == 0) {
        dist_seq = malloc(g.n * sizeof(int));
        bfs_sequential(&g, p.source, dist_seq);
        for (int r = 0; r < p.runs; ++r) {
            double t0 = MPI_Wtime();
            levels_seq = bfs_sequential(&g, p.source, dist_seq);
            times[r] = MPI_Wtime() - t0;
        }
        t_seq = median(times, p.runs);
    }

    Workspace w;
    init_workspace(&w, g.n);
    int *dist = malloc(g.n * sizeof(int));
    Stats st = {0};
    double *comm = malloc(p.runs * sizeof(double));
    bfs_hybrid(&g, p.source, dist, &w, p.exchange, &st);
    for (int r = 0; r < p.runs; ++r) {
        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();
        bfs_hybrid(&g, p.source, dist, &w, p.exchange, &st);
        double t_local = MPI_Wtime() - t0, t_max, comm_max;
        MPI_Reduce(&t_local, &t_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&st.comm_time, &comm_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        times[r] = t_max;
        comm[r] = comm_max;
    }

    unsigned long long h = dist_checksum(dist, g.n), h_min, h_max;
    MPI_Reduce(&h, &h_min, 1, MPI_UNSIGNED_LONG_LONG, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&h, &h_max, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (p.stats) {
        st.frontier_sizes = malloc((g.n + 1) * sizeof(int));
        bfs_hybrid(&g, p.source, dist, &w, p.exchange, &st);
    }

    if (rank == 0) {
        long long mismatches = 0, unreachable = 0;
        for (int v = 0; v < g.n; ++v) {
            mismatches += dist[v] != dist_seq[v];
            unreachable += dist_seq[v] == -1;
        }
        double t_par = median(times, p.runs), t_comm = median(comm, p.runs);
        double speedup = t_seq / t_par;
        printf("graph=%s n=%d m=%lld directed=%d csr_edges=%d avg_degree=%.2f source=%d prep_time=%.3f s\n", p.path,
               g.n, g.m, p.directed, g.offsets[g.n], (double)g.offsets[g.n] / g.n, p.source, t_prep_max);
        printf("procs=%d threads=%d total=%d schedule=%s,%d exchange=%s runs=%d\n", nprocs, w.nthreads,
               nprocs * w.nthreads, schedule_name(p.schedule), p.chunk,
               p.exchange == EXCHANGE_ALLREDUCE ? "allreduce" : "allgatherv", p.runs);
        printf("levels seq=%d par=%d unreachable=%lld mismatches=%lld ranks_consistent=%s correct=%s\n", levels_seq,
               st.levels, unreachable, mismatches, h_min == h_max ? "yes" : "no",
               mismatches == 0 && h_min == h_max ? "yes" : "no");
        printf("time seq=%.4f s par=%.4f s comm=%.4f s (%.1f%%) speedup=%.2f efficiency=%.3f\n", t_seq, t_par, t_comm,
               100.0 * t_comm / t_par, speedup, speedup / (nprocs * w.nthreads));
        if (p.stats) {
            int max_frontier = 0;
            printf("frontier sizes:");
            for (int l = 0; l < st.levels; ++l) {
                printf(" %d", st.frontier_sizes[l]);
                max_frontier = st.frontier_sizes[l] > max_frontier ? st.frontier_sizes[l] : max_frontier;
            }
            printf("\nmax frontier=%d levels=%d\n", max_frontier, st.levels);
        }
    }
    if (p.stats)
        print_balance(&st, &w);

    free(st.frontier_sizes);
    free(times);
    free(comm);
    free(dist);
    free(dist_seq);
    free_workspace(&w);
    free(g.offsets);
    free(g.edges);
    MPI_Finalize();
    return 0;
}
