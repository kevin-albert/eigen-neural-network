#ifndef math_fns_h
#define math_fns_h

#include <cmath>
#include <functional>

inline float sigf(float x) {
    return x < -45 ? 0 : 
           x >  45 ? 1 : 
           1.0 / (1.0 + std::exp(-x));
}

inline float dtanhfi(float th) {
    return 1 - th*th;
}

inline float dsigfi(float sig) {
    return sig * (1.0 - sig);
}

template<int N, class V>
inline void mod_inplace(Eigen::VectorBlock<V> vec, std::function<float(float)> func) {
    for (int i = 0; i < N; ++i) {
        vec.data()[i] = func(vec.data()[i]);
    }
}

template<int N, class V>
inline void mod_inplace(V &vec, std::function<float(float)> func) {
    for (int i = 0; i < N; ++i) {
        vec.data()[i] = func(vec.data()[i]);
    }
}

#endif