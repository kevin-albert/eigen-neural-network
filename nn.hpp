#ifndef inc_nn_hpp
#error "include \"nn.h\", not \"nn.hpp\""
#else

#ifndef nn_hpp 
#define nn_hpp 

#include <cmath>
#include <iostream>

#define MAX_VECTOR_STACK 1000

namespace nn {


    template<class ..._>
    inline void pass(_&&...) {}


    inline float sigmoid(float x) {
        return x < -45 ? 0 :
               x >  45 ? 1 :
               1.0 / (1.0 + std::exp(-x));
    }


    inline float dsigmoid(float x) {
        return (1.0 - x) * x;
    }




    template<size_t A, size_t B>
    inline static auto _forwardstep(Connection<A,B> &connection) {
        return connection.lower.Z * connection.W;
    }


    template<size_t A, size_t B, class... C>
    inline static auto _forwardstep(Connection<A,B> &connection, C&... connections) {
        return connection.lower.Z * connection.W + _forwardstep(connections...);
    }


    template<size_t A, size_t B, class... C>
    void forwardstep(Connection<A,B> &first, C&... connections) {
        first.upper.Z = first.upper.B + _forwardstep(first, connections...);
        for (int i = 0; i < B; ++i) {
            first.upper.Z(0, i) = sigmoid(first.upper.Z(0, i));
        }
    }



    
    template<size_t N>
    void backprop_start(Layer<N> &out, const Eigen::MatrixXf &Y) {
        out.D = (out.Z - Y).transpose();
    }

    template<size_t A, size_t B>
    int _backwardstep(Connection<A,B> &connection) {
        connection.lower.D = connection.W * connection.upper.D;
        return 0;
    }




    template<size_t A, size_t B, class... C>
    void backwardstep(Connection<A,B> &first, C&... connections) {
        _backwardstep(first);
        pass( _backwardstep(connections)... );
        for (int i = 0; i < A; ++i) {
            first.lower.D(i, 0) *= dsigmoid(first.lower.Z(0, i));
        }
    }


    template<size_t A, size_t B>
    static inline int _updateweights(const float eta, Connection<A,B> &connection) {
        connection.W += -eta * (connection.upper.D * connection.lower.Z).transpose();
        connection.upper.B += -eta * connection.upper.D.transpose();
        return 0;
    }
    

    template<class... C>
    void updateweights(const float eta, C&... connections) {
        pass( _updateweights(eta, connections)... );
    }


    // error amount - sum of squares
    template<size_t N>
    float error(const Layer<N> &output, const Eigen::MatrixXf &Y) {
        return (Y-output.Z).squaredNorm();
    }
}

#endif
#endif