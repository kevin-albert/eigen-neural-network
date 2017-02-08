#include <iostream>
#include <string>

#include "lstm.h"

int main(void) {

    LSTM<1> lstm;

    for (int e = 0; e < 10; ++e) {

        std::cout << "EPOCH: " << e << "\n";
        State<1> state[2];
        Gradients<1> grads[2];

        // sequence of length 2        
        state[0].x()[0] = 1;
        state[1].x()[0] = -1;

        for (int i = 0; i < 2; ++i) {
            if (i > 0) state[i].after(state[i-1]);
            lstm.forward_pass(state[i]);
            int j = 1-i;
            
            grads[i].d_h = state[i].h()-state[j].x();
            float err = std::abs(grads[i].d_h[0]);

            std::cout << "[" << i << "] expected:   " << state[j].x() << "\n";
            std::cout << "[" << i << "] out:        " << state[i].h() << "\n";
            std::cout << "[" << i << "] error:      " << err << "\n";
        }

        lstm.backward_pass(state[1], grads[1]);
        grads[0].before(grads[1]);
        lstm.backward_pass(state[0], grads[0]);
        lstm.update(0.5, 2, grads);
        std::cout << "\n\n";
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