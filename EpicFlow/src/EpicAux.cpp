#include "./EpicAux.hpp"

/*************************** NN-search on a graph ********************************/

/* auxiliary structures for nearest-neighbor search on a graph */
struct node_dist_t
{
    int node;
    float dis;
    node_dist_t() {}
    node_dist_t(int i, float f) : node(i), dis(f) {}
};

using current_t = node_dist_t;
using list_node_dist_t = std::vector<node_dist_t>;

template <typename T>
struct smallest_on_top
{
    bool operator()(const T& a, const T& b) const
    {
        return a.dis > b.dis;
    }
};

/* Find nearest neighbors in a weighted directed graph (The matrix must be symmetric !) */
int find_nn_graph_arr(const csr_matrix& graph, int seed, int nmax, cv::Mat& best, cv::Mat& dist)
{
    assert(nmax > 0);
    assert(graph.nr == graph.nc && (graph.indptr[graph.nr] % 2 == 0));

    // init done to INF
    std::vector<float> done(graph.nr, FLT_MAX);

    // explore nodes in order of increasing distances
    std::priority_queue<current_t, std::vector<current_t>, smallest_on_top<current_t>> stack;
    stack.emplace(seed, 0);
    done[seed] = 0; // mark as done

    int n = 0;
    while (stack.size())
    {
        current_t cur = stack.top();
        stack.pop();
        if (cur.dis > done[cur.node]) { continue; }

        // insert result
        best.ptr<float>(seed)[n] = cur.node;
        dist.ptr<float>(seed)[n] = cur.dis;
        n++;
        if (n >= nmax) { break; }

        // find nearest neighbors
        for (size_t i = graph.indptr[cur.node]; i < graph.indptr[cur.node + 1]; i++)
        {
            int neigh = graph.indices[i];
            float newd = cur.dis + graph.data[i];
            if (newd >= done[neigh])
                continue;
            stack.emplace(neigh, newd); // add only if it makes sense
            done[neigh] = newd;
        }
    }

    // in case we do not get enough results
    for (size_t i = 0; i < nmax; ++i) {
        best.ptr<float>(seed)[n] = FLT_MAX;
        dist.ptr<float>(seed)[n] = FLT_MAX;
    }

    return n;
}

/******************* DISTANCE TRANSFORM **************/

float arg_sweep(const cv::Mat& cost, cv::Mat& res, cv::Mat& labels, const int x, const int y)
{
    int i, j;
    const int tx = res.cols;
    const int ty = res.rows;
    float* A = (float*)res.data;
    float* L = (float*)labels.data;
    const float* Cost = (float*)cost.data;

    const int bx = x > 0 ? 0 : tx - 1;
    const int by = y > 0 ? 0 : ty - 1;
    const int ex = x > 0 ? tx : -1;
    const int ey = y > 0 ? ty : -1;

    float t0, t1, t2, C, max_diff = 0.0;
    int l0, l1, l2;
    for (j = by; j != ey; j += y)
        for (i = bx; i != ex; i += x)
        {
            // 处理边界像素
            if (j == by)
            {
                t1 = FLT_MAX;
                l1 = -1;
            }
            else
            {
                t1 = A[i + (j - y) * tx];
                l1 = L[i + (j - y) * tx];
            }

            // 处理边界像素
            if (i == bx)
            {
                t2 = FLT_MAX;
                l2 = -1;
            }
            else
            {
                t2 = A[i - x + j * tx];
                l2 = L[i - x + j * tx];
            }

            // 需注意的是，dmap中除种子点外的像素的灰度均为无穷大，种子点的灰度则和cost中相应位置的灰度值相同
            float dt12 = fabs(t1 - t2);
            C = Cost[i + j * tx];

            if (dt12 > C)
            {
                // 预给当前像素点传递信息的种子点具有有效信息
                // handle degenerate case
                if (t1 < t2)
                {
                    t0 = t1 + C;
                    l0 = l1;
                }
                else
                {
                    t0 = t2 + C;
                    l0 = l2;
                }
            }
            else
            {
                // 预给当前像素点传递信息的种子点不具有有效信息
                t0 = 0.5 * (t1 + t2 + sqrtf(2 * C * C - dt12 * dt12));
                l0 = (t1 < t2) ? l1 : l2;
            }

            // 迭代过程稳定后，信息传播导致的变化会逐渐减小
            if (t0 < A[i + j * tx])
            {
                max_diff = MY_MAX(max_diff, A[i + j * tx] - t0);
                A[i + j * tx] = t0;
                L[i + j * tx] = l0;
            }
        }
    return max_diff;
}

/* Compute distance map from a given seeds (in res) and a cost map.
  if labels!=NULL:  labels are propagated along with distance map
                    (for each pixel, we remember the closest seed)*/
float weighted_distance_transform(const cv::Mat& cost, const dt_params_t& dt_params,
    cv::Mat& res, cv::Mat& labels)
{
    assert(cost.size() == res.size());
    assert(cost.size() == labels.size());
    assert(dt_params.min_change >= 0.0f);

    const std::array<int, 4> x = { -1, 1, 1, -1 };
    const std::array<int, 4> y = { 1, 1, -1, -1 };
    int i = 0, end_iter = 4;
    float change = -1;
    while (++i <= end_iter)
    {
        change = arg_sweep(cost, res, labels, x[i % 4], y[i % 4]);
        if (change > dt_params.min_change)
        { // finish the turn
            end_iter = MY_MIN(dt_params.max_iter, i + 3);
        }
    }
    return change;
}

/****************** BUILD NEIGHBORHOOD GRAPH *************/

/* structure for the border between two regions in the assignment map */
struct border_t
{
    float accu; /* cost as the minimum over the border */
    int nb;     /* number of pixels */

    border_t() : accu(0), nb(0) {}

    void add(float v)
    {
        if (!nb || accu > v) { accu = v; }
        nb++;
    }

    float get_val() const { return accu; }
};

// Jia-Baos
// 2024-01-27 13:57
// a border between two int is represented as a long long
// 64-bit Linux下 long 为8字节，64-bit Windows下 long 为4字节
// 所以这里采用 long long 修正为8字节
// https://blog.csdn.net/wankcn/article/details/121209323
static inline long long key(long long i, long long j)
{
    // always i>j
    if (j > i) { MY_SWAP(i, j); }
    return i + (j << 32);
}

static inline int first(long long i) { return int(i); }
static inline int second(long long i) { return int(i >> 32); }

template <typename Ti>
struct Tint_float_t
{
    Ti i;
    float f;
    Tint_float_t() {}
    Tint_float_t(Ti i, float f) : i(i), f(f) {}
};

using int_float_t = Tint_float_t<int>;
using long_float_t = Tint_float_t<long long>;
static int cmp_long_float(const void* a, const void* b)
{
    long long diff = ((long_float_t*)a)->i - ((long_float_t*)b)->i;
    return (diff > 0) - (diff < 0);
}

/* Find neighboring labels and store their relation in a sparse matrix */
void ngh_labels_to_spmat(int ns, const cv::Mat& labels, const cv::Mat& dmap, csr_matrix& csr)
{
    assert(labels.size() == dmap.size());

    const int tx = labels.rows;
    const int ty = labels.cols;
    const float* lab = (float*)labels.data;
    const float* dis = (float*)dmap.data;

    using ngh_t = std::unordered_map<long long, border_t>;

    ngh_t ngh;
    for (int j = 1; j < ty; j++)
    {
        for (int i = 1; i < tx; i++)
        {
            int l0 = lab[i + j * tx];
            int l1 = lab[i - 1 + j * tx];
            int l2 = lab[i + (j - 1) * tx];
            if (l0 != l1)
            {
                long long k = key(l0, l1);
                float d = dis[i + j * tx] + dis[i - 1 + j * tx];
                ngh[k].add(d);
            }
            if (l0 != l2)
            {
                long long k = key(l0, l2);
                float d = dis[i + j * tx] + dis[i + (j - 1) * tx];
                ngh[k].add(d);
            }
        }
    }

    // convert result into a sparse graph
    std::vector<long_float_t> sorted;
    for (ngh_t::iterator it = ngh.begin(); it != ngh.end(); ++it)
    {
        const float cost = it->second.get_val();
        sorted.emplace_back(it->first, cost);
        long long symkey = (it->first >> 32) + (it->first << 32);
        sorted.emplace_back(symkey, cost); // add symmetric
    }

    // sort by (row,col) = (second,first)
    qsort(sorted.data(), sorted.size(), sizeof(long_float_t), cmp_long_float);

    csr.nr = ns;
    csr.nc = ns;
    csr.indptr.resize(ns + 1);
    csr.indices.resize(sorted.size());
    csr.data.resize(sorted.size());
    int n = 0, r = 0;
    csr.indptr[0] = 0;
    while (n < (signed)sorted.size())
    {
        int row = second(sorted[n].i);
        assert(r <= row);
        while (r < row) { csr.indptr[++r] = n; }
        csr.indices[n] = first(sorted[n].i);
        csr.data[n] = sorted[n].f;
        n++;
    }
    assert(r < ns);
    // finish row marker
    while (r < ns) { csr.indptr[++r] = n; }
}

/* Compute the neighboring matrix between seeds as well as the closest seed for each pixel */
void distance_transform_and_graph(const std::vector<cv::Vec4f>& seeds, const cv::Mat& cost,
    const dt_params_t& dt_params, cv::Mat& labels, cv::Mat& dmap, csr_matrix& ngh, int n_thread)
{
    const int ns = seeds.size();

    // compute distance transform for all seeds altogether
    for (size_t i = 0; i < ns; ++i) {
        const int x = seeds[i][0];
        const int y = seeds[i][1];

        labels.ptr<float>(y)[x] = i;
        dmap.ptr<float>(y)[x] = cost.ptr<float>(y)[x];
    }

    weighted_distance_transform(cost, dt_params, dmap, labels);

    // compute distances between neighboring seeds
    ngh_labels_to_spmat(ns, labels, dmap, ngh);
}

void dist_trf_nnfield_subset(cv::Mat& best, cv::Mat& dist, cv::Mat& labels,
    const std::vector<cv::Vec4f>& seeds,
    const cv::Mat& cost, dt_params_t& dt_params,
    const std::vector<cv::Vec4f>& pixels, const int n_thread)
{
    const int npix = pixels.size();
    assert(best.size() == dist.size());
    assert(best.rows == npix && dist.rows == npix);

    // compute distance transform and build graph
    cv::Mat dmap = cv::Mat::zeros(cost.size(), CV_32FC1);
    dmap.setTo(FLT_MAX);    // 源代码中使用0X7F
    csr_matrix ngh{};
    distance_transform_and_graph(seeds, cost, dt_params, labels, dmap, ngh, n_thread);

    // compute nearest neighbors using the graph
    const int nn = best.cols;
    const int ns = seeds.size();
    cv::Mat nnf = cv::Mat::zeros(ns, nn, CV_32FC1);
    cv::Mat dis = cv::Mat::zeros(ns, nn, CV_32FC1);
    for (int n = 0; n < ns; n++) {
        find_nn_graph_arr(ngh, n, nn, nnf, dis);
    }

    std::cout << "We are here....." << std::endl;
}