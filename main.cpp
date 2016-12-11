#include <iostream>

#include "nn.h"
#include "mnist.h"

// data sizes
const int TRAIN_SIZE = 60000;
const int TEST_SIZE = 10000;

// training parameters
const int TRAIN_MAX_ITERATIONS = 10;
const float TRAIN_MAX_ERROR = 0.01;
const float ETA = 0.9;

// increase contrast
#define PIXEL_XFORM(pixel) (::pow((pixel) + 0.1, 3))

class Network {
public:
    Network(const float eta): 
            eta(eta),
            ih(nn::connect(input, hidden1)),
            hh1(nn::connect(hidden1, hidden2)),
            hh2(nn::connect(hidden2, hidden3)),
            ss(nn::connect(hidden1, output)),
            ho(nn::connect(hidden3, output)),
            Y(nn::makeYVector(output)) {}

    void set_input(const mnist::byte *image) {
        for (int i = 0; i < 784; ++i) {
            input.Z(0,i) = PIXEL_XFORM(((float) image[i]) / 0xff);
        }
    }

    void set_y(const mnist::byte expected) {
        Y.setZero();
        Y(0, expected) = 1;
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

    float error() const {
        return nn::error(output, Y);
    }

    void forwardpass() {
        nn::forwardstep(ih);
        nn::forwardstep(hh1);
        nn::forwardstep(hh2);
        nn::forwardstep(ho, ss);
    }

    void backprop() {
        nn::backprop_start(output, Y);
        nn::backwardstep(ho, ss);
        nn::backwardstep(hh2);
        nn::backwardstep(hh1);
        nn::backwardstep(ih);
        nn::updateweights(eta, ih, hh1, hh2, ss, ho);
    }

private:
    const float eta;
    nn::Layer<784> input;
    nn::Layer<200> hidden1;
    nn::Layer<200> hidden2;
    nn::Layer<100> hidden3;
    nn::Layer<10> output;
    decltype(nn::connect(input, hidden1)) ih;
    decltype(nn::connect(hidden1, hidden2)) hh1;
    decltype(nn::connect(hidden2, hidden3)) hh2;
    decltype(nn::connect(hidden3, output)) ho;
    decltype(nn::connect(hidden1, output)) ss;
    decltype(nn::makeYVector(output)) Y; 
};

int main(int argc, char **argv) {
    
    using namespace std::placeholders;

    float eta = argc == 1 ? ETA : atof(argv[1]);
    std::cout << "eta: " << eta << '\n';

    Network network(eta);

    int num_train = 0,
        num_test,
        num_error,
        num_itor;

    auto process = [&](const bool train, const mnist::byte label, 
                       const mnist::byte *image) {
        
        // set input
        network.set_input(image);
        network.set_y(label);

        if (train) {
            // backprop / re-run until we get the right number
            int max_iterations = TRAIN_MAX_ITERATIONS;
            ++num_train;
            do {
                // set input
                network.forwardpass();
                network.backprop();
                ++num_itor;
            } while (network.error() > TRAIN_MAX_ERROR && --max_iterations > 0);
        } else {
            network.forwardpass();
            
            // increment num_error if we got it wrong
            ++num_test;
            if (!network.output_equals(label)) {
                ++num_error;
            }
        }
    };

    std::cout << "training...\n";

    for (int i = 0; i < 1000; ++i) {

        num_test = num_error = num_itor = 0;

        // training set
        mnist::scan("train-labels-idx1-ubyte", "train-images-idx3-ubyte", 
                    TRAIN_SIZE, std::bind(process, true, _1, _2));

        if (1) {//(i % 10 == 9) {
            std::cout << "num_train:        " << num_train << '\n'
                      << "testing...\n";

            // testing set
            mnist::scan("t10k-labels-idx1-ubyte", "t10k-images-idx3-ubyte", 
                        TEST_SIZE, std::bind(process, false, _1, _2));

            std::cout << "num_test:         " << num_test << '\n'
                      << "num_error:        " << num_error << '\n'
                      << "avg iterations:   " << (num_itor / num_train) << '\n'
                      << "error %:          " << (100*(float) num_error / num_test) << '\n';
        }
    }


    return 0;
}
