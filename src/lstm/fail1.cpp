#include <iostream>

#include "lstm.h"

int main(void) {

    // in:   1
    // out: -1
    LSTM<1> lstm;

    for (int e = 0; e < 100; ++e)
    {
        State<1> state[5];
        Gradients<1> grads[5];
        
        for (int i = 0; i < 5; ++i) {
            state[i].x()[0] = 1;
            lstm.forward_pass(state[i]);
            
            grads[i].d_h[0] = state[i].h()[0] - -1; 
            float err = std::abs(grads[i].d_h[0]);
            
            std::cout << "error:  " << err << "\n";
            std::cout << "output: " << state[i].h() << "\n\n";

            lstm.backward_pass(state[i], grads[i]);
        }

        lstm.update(0.1, 5, grads);
    }


    {

        State<1> state[5];
        for (int i = 0; i < 5; ++i) {
            lstm.forward_pass(state[i]);
            
            float err = std::abs(state[i].h()[0] - -1);

            std::cout << "error:  " << err << "\n";
            std::cout << "output: " << state[i].h() << "\n\n";
        }
    }
}