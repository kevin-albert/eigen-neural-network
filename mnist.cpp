
#include <string>
#include <fstream>
#include <stdexcept>

#include "mnist.h"

#include <iostream>


namespace mnist {

    const size_t LBL_HEADER_SIZE    = 8;
    const size_t IMG_HEADER_SIZE    = 16;
    const size_t LBL_SIZE           = 1;
    const size_t IMG_SIZE           = 28*28;

    void scan(const std::string &label_file, const std::string &image_file, 
              const int N, std::function<void(const byte, const byte*)> fn) {
        
        std::ifstream labels(label_file);
        std::ifstream images(image_file);

        if (!labels.is_open()) {
            throw std::runtime_error("unable to open " + label_file);
        }

        if (!images.is_open()) {
            throw std::runtime_error("unable to open " + label_file);
        }

        labels.seekg(LBL_HEADER_SIZE, labels.beg);
        images.seekg(IMG_HEADER_SIZE, images.beg);

        byte label;
        byte image[784];

        for (int i = 0; i < N; ++i) {
            labels.read((char*)&label, LBL_SIZE);
            images.read((char*) image, IMG_SIZE);
            if (labels && images) {
                fn(label, image);
            } else {
                throw std::runtime_error("read failed");
            }
        }
    }
}