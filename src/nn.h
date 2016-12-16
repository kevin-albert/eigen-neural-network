#ifndef nn_h
#define nn_h

#include "Eigen/Dense"

namespace nn {

    //
    // 3 layer types:
    //      Input
    //      Hidden
    //      Output
    //

    template<size_t N>
    struct LayerBase {

        // activations
        Eigen::MatrixXf Z;

        LayerBase(Eigen::MatrixXf Z): Z(Z) {}

        static constexpr size_t size = N;
    };


    template<size_t N>
    struct InputLayer: LayerBase<N> {
        InputLayer(): LayerBase<N>(Eigen::MatrixXf::Zero(1,N)) {}
    };


    template<size_t N>
    struct Layer: public LayerBase<N> {

        // biases
        Eigen::MatrixXf B;

        // gradients
        Eigen::MatrixXf D;

        // bias momentum
        Eigen::MatrixXf M;

        // previous bias deltas
        // Eigen::MatrixXf M;

        Layer(): LayerBase<N>(Eigen::MatrixXf::Random(1,N) * 0.1),
                B(Eigen::MatrixXf::Random(1,N) * 0.1),
                D(Eigen::MatrixXf(N,1)),
                M(Eigen::MatrixXf::Zero(1,N)) {

            for (int i = 0; i < N; ++i) {
                LayerBase<N>::Z(0,i) += 0.5;
            }
        }
    };


    template<size_t N>
    struct HiddenLayer: public Layer<N> {
        HiddenLayer(): Layer<N>() {}
    };


    template<size_t N>
    struct OutputLayer: public Layer<N> {

        // expected values
        Eigen::MatrixXf Y;

        OutputLayer(): 
                Layer<N>(),
                Y(Eigen::MatrixXf::Zero(1,N)) {}

    };


    template<class A, class B>
    struct Connection {

        // lower layer
        A &lower;

        // upper layer
        B &upper;

        // weight matrix
        Eigen::MatrixXf W;
        
        // weight momentum
        Eigen::MatrixXf M;

        Connection(A &lower, B &upper): 
                lower(lower), 
                upper(upper), 
                W(Eigen::MatrixXf::Random(A::size, B::size) * 0.1),
                M(Eigen::MatrixXf::Zero(A::size, B::size)) {}
    };

    // connect two layers
    template<class A, class B>
    Connection<A,B> connect(A &lower, B &upper) {
        return Connection<A,B>(lower, upper);
    }

    // compute a forward pass from one layer to another
    template<class A, class B, class ...C>
    void forwardstep(Connection<A,B> &first, C&... args);

    // calculate the output delta to begin propagating gradients downward
    template<size_t N>
    void calc_output_delta(Layer<N> &out);

    // for minibatch training - reset the current output delta
    template<size_t N>
    void batch_reset_output_delta(Layer<N> &out);

    // for minibatch training - add a Y vector to the current output delta
    template<size_t N>
    void batch_add_output_delta(Layer<N> &out);

    // compute a backward pass from one layer to another
    template<class A, class B, class... C>
    void backwardstep(Connection<A,B> &first, C&... connections);
    
    // update weights for connections based on gradients
    template<class... C>
    void updateweights(const float eta, const float alpha, C&... connection);

    // error amount - sum of squares
    template<size_t N>
    float error(const Layer<N> &out);

}

#define inc_nn_hpp
#include "nn.hpp"
#undef inc_nn_hpp

#endif