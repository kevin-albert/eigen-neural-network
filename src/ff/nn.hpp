#ifndef inc_nn_hpp
#error "include \"nn.h\", not \"nn.hpp\""
#else

#ifndef nn_hpp 
#define nn_hpp 

#include <cmath>
#include <iostream>

namespace nn {


    template<class ..._>
    inline void pass(_&&...) {}


    inline double sigmoid(double x) {
        return x < -45 ? 0 :
               x >  45 ? 1 :
               1.0 / (1.0 + std::exp(-x));
    }


    inline double dsigmoid(double x) {
        return (1.0 - x) * x;
    }




    template<class A, class B>
    inline static auto _forwardstep(Connection<A,B> &connection) {
        return connection.lower.Z * connection.W;
    }


    template<class A, class B, class... C>
    inline static auto _forwardstep(Connection<A,B> &connection, C&... connections) {
        return _forwardstep(connection) + _forwardstep(connections...);
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
    inline static auto _backwardstep(Connection<A,B> &connection) {
        return connection.W * connection.upper.D;
    }

    template<class A, class B, class... C>
    inline static auto _backwardstep(Connection<A,B> &connection, C&... connections) {
        return _backwardstep(connection) + _backwardstep(connections...);
    }

    template<class A, class B, class... C>
    void backwardstep(Connection<A,B> &first, C&... connections) {
        first.lower.D = _backwardstep(first, connections...);
        for (int i = 0; i < A::size; ++i) {
            first.lower.D(i, 0) *= dsigmoid(first.lower.Z(0, i));
        }
    }



    template<class A, class B>
    int _batch_reset_gradients(Connection<A, B> &connection) {
        connection.lower.D.setZero();
        return 0;
    }

    template<class... C>
    void batch_reset_gradients(C&... connections) {
        pass( _batch_reset_gradients(connections)... );
    }

    

    template<class A, class B, class... C>
    void batch_backwardstep(Connection<A,B> &first, C&... connections) {
        first.tmp = _backwardstep(first, connections...);
        for (int i = 0; i < A::size; ++i) {
            first.lower.D(i, 0) += first.tmp(i, 0) * dsigmoid(first.lower.Z(0, i));
        }
    }




    template<class A, class B>
    static inline int _updateweights(const double eta, const double alpha, 
                                     const double weight_factor, 
                                     Connection<A,B> &connection) {

        connection.W += alpha * connection.M;
        connection.M = -eta * (connection.upper.D * connection.lower.Z).transpose();
        connection.W += connection.M;
        if (weight_factor < 1) connection.W *= weight_factor;
        connection.upper.B += alpha * connection.upper.M;
        connection.upper.M = -eta * connection.upper.D.transpose();
        connection.upper.B += connection.upper.M;
        return 0;
    }
    

    template<class... C>
    void updateweights(const double eta, const double alpha, 
                       const double weight_factor, C&... connections) {
        pass( _updateweights(eta, alpha, weight_factor, connections)... );
    }



    // error amount - sum of squares
    template<size_t N>
    double error(const Layer<N> &out) {
        return (out.Y-out.Z).squaredNorm();
    }
}

#endif
#endif