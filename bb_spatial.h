#pragma once

#include <cstdint>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>

#ifdef _MSC_VER
#include <intrin.h>
#endif

constexpr uint32_t INVALID_NODE = 0xFFFFFFFFu;

namespace bb {

inline uint8_t quantize_abs(float v) noexcept {
    return static_cast<uint8_t>(std::max(0.0f, std::min(1.0f, v)) * 255.0f);
}

inline float dequantize_abs(uint8_t v) noexcept {
    return static_cast<float>(v) / 255.0f;
}

struct AcousticParams {
    uint16_t ir_primary;    
    uint16_t ir_secondary;  
    float    blend_t;       
    float    wet_level;     
    uint8_t  absorption[6]; 
    uint8_t  pad[2];        

    AcousticParams() 
    : ir_primary(0xFFFF), 
      ir_secondary(0xFFFF), 
      blend_t(0.f), 
      wet_level(0.0f),
      absorption{0,0,0,0,0,0},
      pad{0,0}
    {}

    bool operator==(const AcousticParams& o) const {
        return ir_primary == o.ir_primary && ir_secondary == o.ir_secondary &&
               blend_t == o.blend_t && wet_level == o.wet_level &&
               std::memcmp(absorption, o.absorption, 6) == 0; // had to do this to handle audio engine
    }
};

// setting up cpp internal memory
struct alignas(32) Leaf {
    uint32_t       morton;      
    AcousticParams params;      
    
    bool operator<(const Leaf& o) const { return morton < o.morton; }
};

// morton utility function, core logic
inline uint32_t spread_bits_10(uint32_t x) noexcept {
    x &= 0x000003FFu;
    x = (x | (x << 16)) & 0xFF0000FFu;
    x = (x | (x <<  8)) & 0x0F00F00Fu;
    x = (x | (x <<  4)) & 0xC30C30C3u;
    x = (x | (x <<  2)) & 0x49249249u;
    return x;
}

inline uint32_t compact_bits_10(uint32_t x) noexcept {
    x &= 0x49249249u;
    x = (x | (x >>  2)) & 0xC30C30C3u;
    x = (x | (x >>  4)) & 0x0F00F00Fu;
    x = (x | (x >>  8)) & 0xFF0000FFu;
    x = (x | (x >> 16)) & 0x000003FFu;
    return x;
}

inline uint32_t morton_encode(uint32_t x, uint32_t y, uint32_t z) noexcept {
    return spread_bits_10(x) | (spread_bits_10(y) << 1) | (spread_bits_10(z) << 2);
}

inline void morton_decode(uint32_t m, uint32_t& x, uint32_t& y, uint32_t& z) noexcept {
    x = compact_bits_10(m); y = compact_bits_10(m >> 1); z = compact_bits_10(m >> 2);
}

// quantizing world to match the intuition
inline uint32_t world2grid(float v, float world_size = 100.0f, uint32_t cells = 1024) noexcept {
    float norm = std::max(0.0f, std::min(1.0f - 1e-6f, v / world_size));
    return static_cast<uint32_t>(norm * static_cast<float>(cells));
}

// based on XOR divergence concept to find just the changes
inline int divergence_levels(uint32_t prev, uint32_t curr, uint32_t depth = 10) {
    uint32_t diff = prev ^ curr;
    if (diff == 0)return 0;
    unsigned long msb = 0;
#ifdef _MSC_VER
    _BitScanReverse(&msb, diff);
#else
    msb = 31 - __builtin_clz(diff); // assuming diff != 0
#endif
    return static_cast<int>(depth) - std::max(0, static_cast<int>(depth) - 1 - (static_cast<int>(msb) / 3));
}

// eytzinger layout in memory
namespace detail {
    void eytzinger_rec(const uint32_t* src, uint32_t* dst, size_t n, size_t node, size_t& k) {
        if (node > n) return;
        
        eytzinger_rec(src, dst, n, 2 * node, k);
        dst[node] = src[k++];
        eytzinger_rec(src, dst, n, 2 * node + 1, k);
    }
    void eytzinger_build(const uint32_t* src, uint32_t* dst, size_t n) {
        size_t k = 0; eytzinger_rec(src, dst, n, 1, k);
    }
}

// core structure making it easier to store room voxels
class LinearOctree {
public:
    explicit LinearOctree(uint32_t depth = 10) : depth_(depth), baked_(false) {}

    void insert(const Leaf& leaf) { leaves_.push_back(leaf); baked_ = false; }
    void insert(uint32_t m, const AcousticParams& p) { insert(Leaf{m, p}); }

    void bake() {
        std::sort(leaves_.begin(), leaves_.end());
        std::vector<Leaf> deduped;
        for (auto& lf : leaves_) {
            if (!deduped.empty() && deduped.back().morton == lf.morton) deduped.back() = lf;
            else deduped.push_back(lf);
        }
        leaves_ = std::move(deduped);
        size_t n = leaves_.size();
        keys_.resize(n);
        for (size_t i = 0; i < n; i++) keys_[i] = leaves_[i].morton;
        eyt_.assign(n + 1, INVALID_NODE);
        if (n > 0) detail::eytzinger_build(keys_.data(), eyt_.data(), n);
        baked_ = true;
    }

    const Leaf* find(uint32_t morton) const {
        if (!baked_) return nullptr;
        size_t n = leaves_.size(), i = 1;
        while (i <= n) {
            if (morton < eyt_[i]) i = 2*i;
            else if (morton > eyt_[i]) i = 2*i + 1;
            else {
                return &leaves_[i-1]; 
            }
        }
        return nullptr;
    }

    // revisitable, maybe some logical issue
    const Leaf* query(float x, float y, float z, float world_size = 100.0f) const {
        uint32_t gridy = 1u << depth_;
        return find(morton_encode(world2grid(x,world_size,gridy), world2grid(y,world_size,gridy), world2grid(z,world_size,gridy)));
    }

    size_t size() const{ return leaves_.size(); }
    uint32_t depth() const { return depth_; }
    const std::vector<Leaf>& leaves() const { return leaves_; }

private:
    // (probably unnecessary)
    // leaves and keys to pass to interface for obversability 
    uint32_t depth_; bool baked_;
    std::vector<Leaf> leaves_;
    std::vector<uint32_t> keys_; 
    std::vector<uint32_t> eyt_;
};

// already changed SIMD optims to FFT based because it was allowing 10s IR
// door/portal semantic abstraction features

void fill_box(LinearOctree& tree, float ox, float oy, float oz, float ex, float ey, float ez,
                     AcousticParams& params, float world_size = 100.0f) {
    uint32_t c = 1u << tree.depth();
    uint32_t x0 = world2grid(ox, world_size, c), x1 = world2grid(ox+ex, world_size, c);
    uint32_t y0 = world2grid(oy, world_size, c), y1 = world2grid(oy+ey, world_size, c);
    uint32_t z0 = world2grid(oz, world_size, c), z1 = world2grid(oz+ez, world_size, c);
    for (uint32_t gx = x0; gx < x1; ++gx)
        for (uint32_t gy = y0; gy < y1; ++gy)
            for (uint32_t gz = z0; gz < z1; ++gz)
                tree.insert(Leaf{morton_encode(gx,gy,gz), params});
}

void fill_portal(LinearOctree& tree, float ox, float oy, float oz, float ex, float ey, float ez,
                        uint16_t ir_a, uint16_t ir_b, int axis, float wet_a, float wet_b,
                        const float abs_a[6], const float abs_b[6], float world_size = 100.f) {
    uint32_t c = 1u << tree.depth();
    uint32_t x0 = world2grid(ox, world_size, c), x1 = world2grid(ox+ex, world_size, c);
    uint32_t y0 = world2grid(oy, world_size, c), y1 = world2grid(oy+ey, world_size, c);
    uint32_t z0 = world2grid(oz, world_size, c), z1 = world2grid(oz+ez, world_size, c);
    uint32_t span[3] = {x1-x0, y1-y0, z1-z0};

    for (uint32_t gx = x0; gx < x1; ++gx)
        for (uint32_t gy = y0; gy < y1; ++gy)
            for (uint32_t gz = z0; gz < z1; ++gz) {
                uint32_t local[3] = {gx-x0, gy-y0, gz-z0};
                float t = (span[axis] > 1) ? static_cast<float>(local[axis])/static_cast<float>(span[axis]-1) : 0.5f;
                AcousticParams p;
                p.ir_primary = ir_a; p.ir_secondary = ir_b;
                p.blend_t = t; p.wet_level = wet_a * (1.0f-t) + wet_b*t;
                for (int f = 0; f < 6; f++) p.absorption[f] = quantize_abs(abs_a[f]*(1.0f-t) + abs_b[f]*t);
                tree.insert(Leaf{morton_encode(gx,gy,gz), p});
            }
}
}