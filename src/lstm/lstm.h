#ifndef lstm_h
#define lstm_h

#include <iostream>
#include <cmath>
#include <random>
#include <vector>
#include <algorithm>

#include "../../lib/Eigen/Dense"
#include "util.h"

using Vec = Eigen::VectorXf;
using Mat = Eigen::MatrixXf;

template<class V>
void dbg(std::string lbl, V v) {
    std::cout << lbl << ": ";
    for (int i = 0; i < v.size(); ++i) {
        if (i) std::cout << ", ";
        std::cout << v[i];
    }
    std::cout << '\n';
}

//http://arunmallya.github.io/writeups/nn/lstm/index.html

template <int X, int N>
struct State {
    // everything goes in one vector.
    // easy :)
    Vec data {Vec::Zero(6*N)};
    Vec I{Vec::Zero(N+X)};

    auto z() -> decltype(data.segment(   0, 4*N)) { return data.segment(   0, 4*N); }

    // IMPORTANT!
    // a, i, f, o must remain in this order

    // new cell input
    auto a() -> decltype(data.segment(   0,   N)) { return data.segment(   0,   N); }
    
    // gates
    auto i() -> decltype(data.segment(   N,   N)) { return data.segment(   N,   N); }
    auto f() -> decltype(data.segment( 2*N,   N)) { return data.segment( 2*N,   N); }
    auto o() -> decltype(data.segment( 3*N,   N)) { return data.segment( 3*N,   N); }
    
    // previous cell state
    auto cp() -> decltype(data.segment( 4*N,   N)) { return data.segment( 4*N,   N); }
    // new cell state
    auto c() -> decltype(data.segment( 5*N,   N)) { return data.segment( 5*N,   N); }

    // I
    auto h() -> decltype(I.segment(X, N)) { return I.segment(X, N); }

    void after(State<X,N> &last) {
        h() = last.h();
        cp() = last.c();
    }
};


template <int X, int N>
struct Gradients {
    //
    // partial derivatives
    // notation: 
    //                  ∂E
    // d_Foo = δFoo =  ---- = partial derivative of error WRT Foo
    //                 ∂Foo

    // δh 
    Vec d_h{Vec::Zero(N)}; 
    
    // δo
    Vec d_o{Vec::Zero(N)}; 
    
    // δc, δc{t-1}
    Vec d_c{Vec::Zero(N)}; 
    Vec d_cp{Vec::Zero(N)}; 
    
    // δf
    Vec d_f{Vec::Zero(N)}; 
    
    // δi
    Vec d_i{Vec::Zero(N)}; 
    
    // δa
    Vec d_a{Vec::Zero(N)};

    // used in computational step during backprop
    // partial derivatives of all *linear* state values (^ means linear here):
    // [ δa^, δi^, δf^, δo^ ]
    Vec d_z{Vec::Zero(4*N)};

    // δW
    Mat d_W{Mat::Zero(4*N,2*N)};

    // δx, δh{t-1}
    Vec d_I{Vec::Zero(X+N)};
    auto  d_x() -> decltype(d_I.segment(0, X)) { return d_I.segment(0, X); }
    auto d_hp() -> decltype(d_I.segment(X, N)) { return d_I.segment(X, N); }

    void before(Gradients<X,N> &next) {
        d_h += next.d_hp();
    }
};


template<int X, int N> class LSTM {
public:

    LSTM() {
        // initialize weights     
        std::default_random_engine rng;
        std::normal_distribution<float> dist(0,1.0/sqrt(X));
        for (int i = 0; i < 4*N * 2*N; ++i) {
            W.data()[i] = dist(rng);
        }
    }


    // forward pass
    void forward_pass(Vec x, State<X,N> &state) {

        // get weight matrices                    // // another possible approach:
        auto Wxc = W.block( 0*N,   0,   N,   X);  // // | a^ |   | Wxc Whc |
        auto Whc = W.block( 0*N,   X,   N,   N);  // // | i^ |   | Wxi Whi |   | x       |
        auto Wxi = W.block( 1*N,   0,   N,   X);  // // | f^ | = | Wxf Whf | X | h^{t-1} |
        auto Whi = W.block( 1*N,   X,   N,   N);  // // | o^ |   | Wxo Who |
        auto Wxf = W.block( 2*N,   0,   N,   X);  // state.z() = W * state.I(); 
        auto Whf = W.block( 2*N,   X,   N,   N);  // state.i() = g(state.i() + i_b);
        auto Wxo = W.block( 3*N,   0,   N,   X);  // state.f() = ...
        auto Who = W.block( 3*N,   X,   N,   N);  // state.o() = ...
                                                  // // maybe we'll try it and see if things improve...

        // calculate new cell input & gates
        auto h = state.h();
        
        // a = tanh(Wxc * x + Whc * h + c_b)
        state.a() = Wxc * x + Whc * h + c_b; mod_inplace<N>(state.a(), std::tanhf);

        // i = g(Wxi * x + Whi * h + i_b)
        state.i() = Wxi * x + Whi * h + i_b; mod_inplace<N>(state.i(), sigf);

        // f = g(Wxf * x + Whi * h + f_b)
        state.f() = Wxf * x + Whf * h + f_b; mod_inplace<N>(state.f(), sigf);

        // o = g(Wxo * x + Who * h + o_b)
        state.o() = Wxo * x + Who * h + o_b; mod_inplace<N>(state.o(), sigf);

        // update memory cell
        // c = i ⊙ a + f ⊙ c^{t-1}
        state.c() = state.i().cwiseProduct(state.a()) + 
                    state.f().cwiseProduct(state.cp());
        // output
        // h = o ⊙ tanh(c)
        state.h() = state.c(); mod_inplace<N>(state.h(), std::tanhf);
        state.h() = state.o().cwiseProduct(state.h());
    }


    // backward pass
    void backward_pass(Vec x, State<X,N> &state, Gradients<X,N> &grads) {

        auto a = state.a();     // cell input
        auto i = state.i();     // input gate
        auto f = state.f();     // forget gate
        auto o = state.o();     // output gate
        auto c = state.c();     // cell state
        auto cp = state.cp();   // last cell state
        auto h = state.h();     // cell output

        ///////////////////////////////////////////////////////////////////////
        // chain rule time                                                   //
        ///////////////////////////////////////////////////////////////////////
        
        // calculate δo
        //  h = o ⊙ tanh(c)
        // δo = δh ⊙ tanh'(c)
        //    = δh ⊙ (1-h^2)
        
        // set tmp to 1-h^2 
        tmp = c; mod_inplace<N>(tmp, std::tanhf); 
        // NOTE - we reuse tmp in grads.d_c down below
        grads.d_o = grads.d_h.cwiseProduct(tmp);

        // calculate δc
        //  h = o ⊙ tanh(c)
        // δc = δc{t+1} + δh ⊙ o ⊙ tanh'(c)
        //    = δc{t+1} + δh ⊙ o ⊙ (1-h^2)
        mod_inplace<N>(tmp, dtanhfi);
        grads.d_c = grads.d_cp + grads.d_h.cwiseProduct(o).cwiseProduct(tmp);
        // and propagate to t-1
        grads.d_cp = grads.d_c.cwiseProduct(f);

        // calculate δi
        //  c = i ⊙ a + f ⊙ c^{t-1}
        // δi = δc ⊙ a 
        grads.d_i = grads.d_c.cwiseProduct(a);

        // calculate δf
        //  c = i ⊙ a + f ⊙ c^{t-1}
        // δf = δc ⊙ c^{t-1}
        // remember cp = c^{t-1}
        grads.d_f = grads.d_c.cwiseProduct(cp);

        // calculate δa
        //  c = i ⊙ a + f ⊙ c^{t-1}
        // δa = δc ⊙ i
        grads.d_a = grads.d_c.cwiseProduct(i);

        // the linear component of the forward pass looks like:
        // z^ = W x I 
        // so in order to calc weight gradients and δh^{t-1}, we want to 
        // calculate δz:
        //
        // δa^ = δa ⊙ tanh'(a^) = δa ⊙ (1-a^2)
        // δi^ = δi ⊙ σ'(i^)    = δi ⊙ i ⊙ (1-i)
        // δf^ = δf ⊙ σ'(f^)    = δf ⊙ f ⊙ (1-f)
        // δo^ = δo ⊙ σ'(o^)    = δo ⊙ o ⊙ (1-o)
        // δz = [ δa^, δi^, δf^, δo^ ]
        // 
        auto ONE = Vec::Constant(N,1);
        tmp = a; mod_inplace<N>(tmp, dtanhfi);
        grads.d_z << grads.d_a.cwiseProduct(tmp),
                     grads.d_i.cwiseProduct(i).cwiseProduct(ONE - i),
                     grads.d_f.cwiseProduct(f).cwiseProduct(ONE - f),
                     grads.d_o.cwiseProduct(o).cwiseProduct(ONE - o);

        // z  = W x I
        // δW = δz X I^T
        state.I.segment(0,X) = x;
        grads.d_W = grads.d_z * state.I.transpose();
        
        // δI = [δx, δh^{t-1}] = δW^T X δz
        grads.d_I = grads.d_W.transpose() * grads.d_z;
    }


    void update(float lr, int n_steps, Gradients<X,N> *gradients) {

        // apply gradients from each timestep
        for (int i = 0; i < n_steps; ++i) {
            W   -= lr * gradients[i].d_W;
            c_b -= lr * gradients[i].d_c;
            i_b -= lr * gradients[i].d_i;
            f_b -= lr * gradients[i].d_f;
            o_b -= lr * gradients[i].d_o;
        }
    }

private:
    // weights
    Mat W{Mat(4*N, X+N)};

    // biases
    Vec c_b{Vec::Random(N)}; 
    Vec i_b{Vec::Random(N)}; 
    Vec f_b{Vec::Constant(N,1)}; 
    Vec o_b{Vec::Random(N)};

    // used in intermediate calculations
    Vec tmp{N};
};

#endif