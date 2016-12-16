#ifndef _network_h
#define _network_h

#include "mnist.h"
#include "nn.h"

class Network {

public:
    Network():
        ih(nn::connect(input,h1)),
        hh(nn::connect(h1,h2)),
        ho(nn::connect(h2,output))
    {}

    void set_image(const mnist::byte *image) {
        for (int i = 0; i < 784; ++i) {
            input.Z(0,i) = ::pow((float)image[i] / 0xff, 3);
        }
    }

    void set_label(const mnist::byte label) {
        output.Y.setZero();
        output.Y(0,label) = 1;
    }

    mnist::byte get_output() {
        float max = 0;
        mnist::byte result;
        for (int i = 0; i < 10; ++i) {
            if (output.Z(0,i) > max) {
                result = i;
                max = output.Z(0,i);
            }
        }
        return result;
    }

    void forwardpass() {
        nn::forwardstep(ih);
        nn::forwardstep(hh);
        nn::forwardstep(ho);
    }

    void backwardpass(const float eta, const float alpha) {
        nn::calc_output_delta(output);
        nn::backwardstep(ho);
        nn::backwardstep(hh);
        nn::updateweights(eta, alpha, ih, hh, ho);
    }

    void batch_backwardpass() {
        nn::calc_output_delta(output);
        nn::batch_backwardstep(ho);
        nn::batch_backwardstep(hh);            
    }

    void batch_update_reset(const float eta, const float alpha) {
        nn::updateweights(eta, alpha, ih, hh, ho);
        nn::batch_reset_gradients(hh, ho);
    }

private:
    nn::InputLayer<784> input;
    nn::HiddenLayer<200> h1;
    nn::HiddenLayer<100> h2;
    nn::OutputLayer<10> output;
    decltype(nn::connect(input,h1)) ih;
    decltype(nn::connect(h1,h2)) hh;
    decltype(nn::connect(h2,output)) ho;
};

#endif