// WTiVo Fast Merge by Distance v7
// CPU implementation with OpenMP parallel candidate generation.
// Copyright (c) 2026 MostAadTech / WTiVo
// MIT License - see ../LICENSE

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(_WIN32)
#define WTIVO_EXPORT extern "C" __declspec(dllexport)
#else
#define WTIVO_EXPORT extern "C"
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

struct Vec3 { float x, y, z; };
struct Cell {
    std::int64_t x, y, z;
    bool operator==(const Cell& o) const noexcept { return x == o.x && y == o.y && z == o.z; }
};
struct CellHash {
    std::size_t operator()(const Cell& c) const noexcept {
        std::uint64_t h = 1469598103934665603ull;
        auto mix = [&h](std::uint64_t v) { h ^= v; h *= 1099511628211ull; };
        mix(static_cast<std::uint64_t>(c.x));
        mix(static_cast<std::uint64_t>(c.y));
        mix(static_cast<std::uint64_t>(c.z));
        return static_cast<std::size_t>(h);
    }
};
static inline Cell cell_of(const Vec3& p, double inv) noexcept {
    return {static_cast<std::int64_t>(std::floor(static_cast<double>(p.x) * inv)),
            static_cast<std::int64_t>(std::floor(static_cast<double>(p.y) * inv)),
            static_cast<std::int64_t>(std::floor(static_cast<double>(p.z) * inv))};
}
static inline float dist2(const Vec3& a, const Vec3& b) noexcept {
    const float dx = a.x - b.x, dy = a.y - b.y, dz = a.z - b.z;
    return dx * dx + dy * dy + dz * dz;
}

class DSU {
public:
    explicit DSU(std::size_t n) : parent(n), rank(n, 0) {
        for (std::size_t i = 0; i < n; ++i) parent[i] = static_cast<std::uint32_t>(i);
    }
    std::uint32_t find(std::uint32_t x) {
        while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; }
        return x;
    }
    void unite(std::uint32_t a, std::uint32_t b) {
        a = find(a); b = find(b);
        if (a == b) return;
        if (rank[a] < rank[b]) std::swap(a, b);
        parent[b] = a;
        if (rank[a] == rank[b]) ++rank[a];
    }
private:
    std::vector<std::uint32_t> parent;
    std::vector<std::uint8_t> rank;
};

static void build_buckets(
    const std::vector<Vec3>& v,
    double inv,
    std::unordered_map<Cell, std::vector<std::uint32_t>, CellHash>& buckets)
{
    // The hash table is built once. The expensive distance tests are parallel.
    buckets.reserve(static_cast<std::size_t>(v.size()) / 2 + 1);
#ifdef _OPENMP
    const int threads = omp_get_max_threads();
    std::vector<std::unordered_map<Cell, std::vector<std::uint32_t>, CellHash>> locals(static_cast<std::size_t>(threads));
#pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        auto& local = locals[static_cast<std::size_t>(tid)];
#pragma omp for schedule(static)
        for (std::int64_t i = 0; i < static_cast<std::int64_t>(v.size()); ++i) {
            local[cell_of(v[static_cast<std::size_t>(i)], inv)].push_back(static_cast<std::uint32_t>(i));
        }
    }
    for (auto& local : locals) {
        for (auto& kv : local) {
            auto& dst = buckets[kv.first];
            dst.insert(dst.end(), kv.second.begin(), kv.second.end());
        }
    }
    for (auto& kv : buckets) std::sort(kv.second.begin(), kv.second.end());
#else
    for (std::uint32_t i = 0; i < v.size(); ++i)
        buckets[cell_of(v[i], inv)].push_back(i);
#endif
}

WTIVO_EXPORT int wtivo_thread_count() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

WTIVO_EXPORT int wtivo_weld_f32_u32(
    const float* vertices_xyz, std::uint32_t vertex_count,
    const std::uint32_t* indices, std::uint32_t index_count,
    float distance, int centroid_merge, int remove_degenerate,
    float** out_vertices_xyz, std::uint32_t* out_vertex_count,
    std::uint32_t** out_indices, std::uint32_t* out_index_count,
    std::uint32_t** out_representatives, std::uint32_t** out_remap)
{
    if (!vertices_xyz || !indices || !out_vertices_xyz || !out_vertex_count ||
        !out_indices || !out_index_count || !out_representatives || !out_remap ||
        vertex_count == 0 || index_count == 0 || (index_count % 3) != 0 ||
        !std::isfinite(distance) || distance <= 0.0f) return 1;

    for (std::uint32_t i = 0; i < index_count; ++i)
        if (indices[i] >= vertex_count) return 2;

    std::vector<Vec3> v(vertex_count);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (std::int64_t i = 0; i < static_cast<std::int64_t>(vertex_count); ++i) {
        v[static_cast<std::size_t>(i)] = {vertices_xyz[3*i], vertices_xyz[3*i+1], vertices_xyz[3*i+2]};
    }

    const double inv = 1.0 / static_cast<double>(distance);
    const float threshold2 = distance * distance;
    std::unordered_map<Cell, std::vector<std::uint32_t>, CellHash> buckets;
    build_buckets(v, inv, buckets);

    // Generate candidate pairs in parallel. DSU union is deliberately kept
    // deterministic and serial after candidate generation.
#ifdef _OPENMP
    const int threads = omp_get_max_threads();
#else
    const int threads = 1;
#endif
    std::vector<std::vector<std::pair<std::uint32_t, std::uint32_t>>> local_pairs(static_cast<std::size_t>(threads));

#ifdef _OPENMP
#pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        auto& pairs = local_pairs[static_cast<std::size_t>(tid)];
#pragma omp for schedule(static)
        for (std::int64_t ii = 0; ii < static_cast<std::int64_t>(vertex_count); ++ii) {
            const std::uint32_t i = static_cast<std::uint32_t>(ii);
            const Cell c = cell_of(v[i], inv);
            for (int dz=-1; dz<=1; ++dz) for (int dy=-1; dy<=1; ++dy) for (int dx=-1; dx<=1; ++dx) {
                const Cell n{c.x+dx,c.y+dy,c.z+dz};
                const auto it = buckets.find(n);
                if (it == buckets.end()) continue;
                for (const std::uint32_t j : it->second) {
                    if (j >= i) continue;
                    if (dist2(v[i], v[j]) <= threshold2) pairs.emplace_back(j, i);
                }
            }
        }
    }
#else
    for (std::uint32_t i = 0; i < vertex_count; ++i) {
        const Cell c = cell_of(v[i], inv);
        for (int dz=-1; dz<=1; ++dz) for (int dy=-1; dy<=1; ++dy) for (int dx=-1; dx<=1; ++dx) {
            const Cell n{c.x+dx,c.y+dy,c.z+dz};
            const auto it = buckets.find(n);
            if (it == buckets.end()) continue;
            for (const std::uint32_t j : it->second)
                if (j < i && dist2(v[i], v[j]) <= threshold2) local_pairs[0].emplace_back(j, i);
        }
    }
#endif

    std::size_t pair_count = 0;
    for (const auto& p : local_pairs) pair_count += p.size();
    std::vector<std::pair<std::uint32_t, std::uint32_t>> pairs;
    pairs.reserve(pair_count);
    for (auto& p : local_pairs) {
        pairs.insert(pairs.end(), p.begin(), p.end());
        std::vector<std::pair<std::uint32_t, std::uint32_t>>().swap(p);
    }
    std::sort(pairs.begin(), pairs.end());

    DSU dsu(vertex_count);
    for (const auto& p : pairs) dsu.unite(p.first, p.second);

    std::unordered_map<std::uint32_t, std::uint32_t> root_to_output;
    root_to_output.reserve(vertex_count);
    std::vector<std::uint32_t> remap(vertex_count);
    std::vector<std::uint32_t> reps;
    std::vector<Vec3> out;
    out.reserve(vertex_count); reps.reserve(vertex_count);
    std::vector<double> sx, sy, sz;
    std::vector<std::uint32_t> counts;
    sx.reserve(vertex_count); sy.reserve(vertex_count); sz.reserve(vertex_count); counts.reserve(vertex_count);

    for (std::uint32_t i = 0; i < vertex_count; ++i) {
        const auto root = dsu.find(i);
        auto [it, inserted] = root_to_output.emplace(root, static_cast<std::uint32_t>(root_to_output.size()));
        const auto cid = it->second;
        if (inserted) {
            out.push_back(v[i]); reps.push_back(i);
            sx.push_back(0.0); sy.push_back(0.0); sz.push_back(0.0); counts.push_back(0);
        }
        remap[i] = cid;
        sx[cid] += v[i].x; sy[cid] += v[i].y; sz[cid] += v[i].z; counts[cid]++;
    }

    const std::size_t cc = out.size();
    std::vector<Vec3> centers(cc);
    std::vector<float> best(cc, std::numeric_limits<float>::infinity());
    for (std::size_t i=0;i<cc;++i)
        centers[i] = {static_cast<float>(sx[i]/counts[i]), static_cast<float>(sy[i]/counts[i]), static_cast<float>(sz[i]/counts[i])};
    for (std::uint32_t i=0;i<vertex_count;++i) {
        const auto cid = remap[i];
        const float d = dist2(v[i], centers[cid]);
        if (d < best[cid]) { best[cid]=d; reps[cid]=i; }
    }
    for (std::size_t i=0;i<cc;++i) if (centroid_merge) out[i]=centers[i]; else out[i]=v[reps[i]];

    std::vector<std::uint32_t> result;
    result.reserve(index_count);
    for (std::uint32_t i=0;i<index_count;i+=3) {
        const auto a=remap[indices[i]], b=remap[indices[i+1]], c=remap[indices[i+2]];
        if (remove_degenerate && (a==b || b==c || c==a)) continue;
        result.push_back(a); result.push_back(b); result.push_back(c);
    }

    const std::size_t vb=out.size()*3*sizeof(float), ib=result.size()*sizeof(std::uint32_t), rb=reps.size()*sizeof(std::uint32_t);
    auto* vm=static_cast<float*>(std::malloc(vb));
    auto* im=static_cast<std::uint32_t*>(std::malloc(ib));
    const std::size_t mb = remap.size() * sizeof(std::uint32_t);
    auto* rm=static_cast<std::uint32_t*>(std::malloc(rb));
    auto* mm=static_cast<std::uint32_t*>(std::malloc(mb));
    if (!vm || !im || !rm || !mm) { std::free(vm); std::free(im); std::free(rm); std::free(mm); return 3; }
    for (std::size_t i=0;i<out.size();++i) { vm[3*i]=out[i].x; vm[3*i+1]=out[i].y; vm[3*i+2]=out[i].z; }
    std::copy(result.begin(), result.end(), im);
    std::copy(reps.begin(), reps.end(), rm);
    std::copy(remap.begin(), remap.end(), mm);
    *out_vertices_xyz=vm; *out_vertex_count=static_cast<std::uint32_t>(out.size());
    *out_indices=im; *out_index_count=static_cast<std::uint32_t>(result.size());
    *out_representatives=rm; *out_remap=mm;
    return 0;
}

WTIVO_EXPORT void wtivo_free(void* ptr) { std::free(ptr); }
