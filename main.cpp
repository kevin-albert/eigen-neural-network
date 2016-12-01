#include <iostream>

#include "nn.h"
#include "mnist.h"

const int TRAIN_SIZE = 60000;
const int TEST_SIZE = 10000;

class Network {
public:
    Network(const float eta): 
            eta(eta),
            ih(nn::connect(input, hidden1)),
            hh(nn::connect(hidden1, hidden2)),
            ho(nn::connect(hidden2, output)),
            Y(nn::makeYVector(output)) {}

    void set_input(const mnist::byte *image) {
        for (int i = 0; i < 784; ++i) {
            input.Z(0,1) = ::pow(((float) image[i]) / 0xff+0.1, 3);
        }
    }

    bool output_equals(const mnist::byte expected) const {
        int j = 0;
        float max_z = 0;
        for (int i = 0; i < 10; ++i) {
            if (output.Z(0,i) > max_z) {
                max_z = output.Z(0,i);
                j = i;
            }
        }
        return expected == j;
    }

    void forwardpass() {
        nn::forwardstep(ih);
        nn::forwardstep(hh);
        nn::forwardstep(ho);
    }

    void backprop(const mnist::byte expected) {
        for (int i = 0; i < 10; ++i) {
            Y(0,i) = (i == expected ? 1 : 0);
        }
        nn::backprop_start(output, Y);
        nn::backwardstep(ho);
        nn::backwardstep(hh);
        nn::backwardstep(ih);
        nn::updateweights(eta, ih, hh, ho);
    }

// private:
    const float eta;
    nn::Layer<784> input;
    nn::Layer<200> hidden1;
    nn::Layer<200> hidden2;
    nn::Layer<10> output;
    decltype(nn::connect(input, hidden1)) ih;
    decltype(nn::connect(hidden1, hidden2)) hh;
    decltype(nn::connect(hidden2, output)) ho;
    decltype(nn::makeYVector(output)) Y; 
};

int main(void) {
    
    using namespace std::placeholders;

    Network network(0.01);

    int num_train = 0,
        num_test = 0,
        num_error = 0,
        num_itor = 0;

    auto process = [&](const bool train, const mnist::byte label, 
                       const mnist::byte *image) {
        
        // set input
        network.set_input(image);

        if (train) {
            // backprop / re-run until we get the right number
            int max_iterations = 100;
            bool error = true;
            ++num_train;
            while (error && --max_iterations) {
                network.forwardpass();
                network.backprop(label);
                error = !network.output_equals(label);
                // std::cout << network.Y << " | " << network.output.Z << " | " << (error ? "F":"P") <<'\n';
                ++num_itor;
            }
        } else {
            // increment num_error if we got it wrong
            ++num_test;
            network.forwardpass();
            if (!network.output_equals(label)) {
                ++num_error;
            }
        }
    };

    std::cout << "training...\n";

    // training set
    mnist::scan("train-labels-idx1-ubyte", "train-images-idx3-ubyte", 
                TRAIN_SIZE, std::bind(process, true, _1, _2));

    std::cout << "num_train:        " << num_train << '\n'
              << "testing...\n";

    // testing set
    mnist::scan("t10k-labels-idx1-ubyte", "t10k-images-idx3-ubyte", 
                TEST_SIZE, std::bind(process, false, _1, _2));

    std::cout << "num_test:         " << num_test << '\n'
              << "num_error:        " << num_error << '\n'
              << "avg iterations:   " << (num_itor / num_train) << '\n'
              << "error %:          " << (100*(float) num_error / num_test) << '\n';
    return 0;
}
