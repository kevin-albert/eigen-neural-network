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




    template<class A, class B>
    inline static auto _forwardstep(Connection<A,B> &connection) {
        return connection.lower.Z * connection.W;
    }


    template<class A, class B, class... C>
    inline static auto _forwardstep(Connection<A,B> &connection, C&... connections) {
        return connection.lower.Z * connection.W + _forwardstep(connections...);
    }


    template<class A, class B, class... C>
    void forwardstep(Connection<A,B> &first, C&... connections) {
        first.upper.Z = first.upper.B + _forwardstep(first, connections...);
        for (int i = 0; i < B::size; ++i) {
            first.upper.Z(0, i) = sigmoid(first.upper.Z(0, i));
        }
    }




    template<size_t N>
    void calc_output_delta(OutputLayer<N> &out) {
        out.D = (out.Z - out.Y).transpose();
    }


    template<size_t N>
    void batch_reset_output_delta(OutputLayer<N> &out) {
        out.D.setZero();
    }


    template<size_t N>
    void batch_add_output_delta(OutputLayer<N> &out) {
        out.D += (out.Z - out.Y).transpose();
    }


    template<class A, class B>
    int _backwardstep(Connection<A,B> &connection) {
        connection.lower.D = connection.W * connection.upper.D;
        return 0;
    }




    template<class A, class B, class... C>
    void backwardstep(Connection<A,B> &first, C&... connections) {
        _backwardstep(first);
        pass( _backwardstep(connections)... );
        for (int i = 0; i < A::size; ++i) {
            first.lower.D(i, 0) *= dsigmoid(first.lower.Z(0, i));
        }
    }


    template<class A, class B>
    static inline int _updateweights(const float eta, const float alpha, 
                                     Connection<A,B> &connection) {

        connection.W += alpha * connection.M;
        connection.M = -eta * (connection.upper.D * connection.lower.Z).transpose();
        connection.W += connection.M;
        connection.upper.B += alpha * connection.upper.M;
        connection.upper.M = -eta * connection.upper.D.transpose();
        connection.upper.B += connection.upper.M;
        return 0;
    }
    

    template<class... C>
    void updateweights(const float eta, const float alpha, C&... connections) {
        pass( _updateweights(eta, alpha, connections)... );
    }


    // error amount - sum of squares
    template<size_t N>
    float error(const Layer<N> &out) {
        return (out.Y-out.Z).squaredNorm();
    }
}

#endif
#endif