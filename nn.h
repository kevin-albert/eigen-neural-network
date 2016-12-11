#ifndef nn_h
#define nn_h

#include "Eigen/Dense"

namespace nn {

    template<size_t N>
    struct Layer {

        // activations
        Eigen::MatrixXf Z;

        // biases
        Eigen::MatrixXf B;

        // gradients
        Eigen::MatrixXf D;

        Layer():
                Z(Eigen::MatrixXf::Random(1,N) * 0.1),
                B(Eigen::MatrixXf::Random(1,N) * 0.1),
                D(Eigen::MatrixXf(N,1)) {
            for (int i = 0; i < N; ++i) {
                Z(0,i) += 0.5;
            }
        }
    };


    template<size_t A, size_t B>
    struct Connection {

        Layer<A> &lower;
        Layer<B> &upper;

        Eigen::MatrixXf W;
        
        Connection(Layer<A> &lower, Layer<B> &upper): 
                lower(lower), upper(upper), W(Eigen::MatrixXf::Random(A,B) * 0.1) {}
    };

    template<size_t A, size_t B>
    Connection<A,B> connect(Layer<A> &lower, Layer<B> &upper) {
        return Connection<A,B>(lower, upper);
    }

    template<size_t N>
    inline Eigen::MatrixXf makeYVector(const Layer<N> &outputLayer) {
        return Eigen::MatrixXf::Zero(1,N);
    }

    // compute a forward pass from one layer to another
    template<size_t A, size_t B, class ...C>
    void forwardstep(Connection<A,B> &first, C&... args);

    // // calculate output gradients from expected values
    template<size_t N>
    void backprop_start(Layer<N> &out, const Eigen::MatrixXf &Y);

    // compute a backward pass from one layer to another
    template<size_t A, size_t B, class... C>
    void backwardstep(Connection<A,B> &first, C&... connections);
    
    // update weights for connections based on gradients
    template<class... C>
    void updateweights(const float eta, C&... connection);

    // error amount - sum of squares
    template<size_t N>
    float error(const Layer<N> &output, const Eigen::MatrixXf &Y);

}

#define inc_nn_hpp
#include "nn.hpp"
#undef inc_nn_hpp

#endif