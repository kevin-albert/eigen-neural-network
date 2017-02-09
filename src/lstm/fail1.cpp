#include <iostream>
#include <string>

#include "lstm.h"

#define D 50

int main(void) {

    Mat X = Mat::Constant(4,3,-1);
    X(1,0) = 1;
    X(2,1) = 1;
    X(3,2) = 1;

    LSTM<3,D> lstm;
    int epochs = 10000;

    for (int e = 0; e < epochs; ++e)
    {

        // std::cout << "EPOCH: " << e << "\n";
        State<3,D> state[4];
        Gradients<3,D> grads[4];

        // sequence of length 4
        // 1 2 1 3
        Mat y = Mat::Constant(4, D, -1);
        y(0, 0) = 1;
        y(1, 1) = 1;
        y(2, 2) = 1;
        y(3, 3) = 1;

        float err = 0;
        for (int i = 0; i < 4; ++i)
        {
            if (i > 0) state[i].after(state[i - 1]);
            Vec x = X.row(i).transpose();
            lstm.forward_pass(x, state[i]);
            err += (state[i].h() - y.row(i).transpose()).squaredNorm();
            grads[i].d_h = Vec::Constant(D, 0.5).cwiseProduct((state[i].h() - y.row(i).transpose()));
        }
        if (e % 100 == 99) std::cout << "[" << e << "] error:      " << err << "\n";

        for (int i = 3; i >= 0; --i)
        {
            if (i < 3) grads[i].before(grads[i + 1]);
            Vec x = X.row(i).transpose();
            lstm.backward_pass(x, state[i], grads[i]);
        }
        lstm.update(0.1, 4, grads);
        // std::cout << "\n\n";
    }
}



/*
int main(void) {
    LSTM<1> lstm;
    lstm.W.setConstant(1);
    lstm.c_b << 0;
    lstm.i_b << 0;
    lstm.f_b << 0;
    lstm.o_b << 0;

    State<1> state;
    state.x() << 0.5;
    state.h() << 0.5;
    state.cp() << -1;

    std::cout << "forward\n";
    lstm.forward_pass(state);

    Gradients<1> grads;
    
    dbg("x", state.x());
    dbg("a", state.a());
    dbg("c", state.c());
    dbg("i", state.i());
    dbg("f", state.f());
    dbg("o", state.o());
    dbg("h", state.h());
    std::cout << "\n";
    dbg("c_b", lstm.c_b);
    dbg("i_b", lstm.i_b);
    dbg("f_b", lstm.f_b);
    dbg("o_b", lstm.o_b);
    std::cout << "\nW:\n" << lstm.W << "\n\n";

    std::cout << "\n\n";
    grads.d_h << state.h()[0] - 0.5;


    std::cout << "backward\n";
    lstm.backward_pass(state, grads);

    dbg("d_h", grads.d_h);
    dbg("d_o", grads.d_o);
    dbg("d_c", grads.d_c);
    dbg("d_cp", grads.d_cp);
    dbg("d_f", grads.d_f);
    dbg("d_i", grads.d_i);
    dbg("d_a", grads.d_a);
    std::cout << "dW:\n" << grads.d_W << "\n\n"; 

    std::cout << "update\n";
    lstm.update(1, 1, &grads);
    state.x() << 0.5;
    state.h() << 0.5;
    std::cout << "forward\n";
    lstm.forward_pass(state);

    dbg("x", state.x());
    dbg("a", state.a());
    dbg("c", state.c());
    dbg("i", state.i());
    dbg("f", state.f());
    dbg("o", state.o());
    dbg("h", state.h());
    std::cout << "\n";
    dbg("c_b", lstm.c_b);
    dbg("i_b", lstm.i_b);
    dbg("f_b", lstm.f_b);
    dbg("o_b", lstm.o_b);
    std::cout << "\nW:\n" << lstm.W << "\n\n";
}
*/
