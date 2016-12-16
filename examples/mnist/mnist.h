#ifndef mnist_h
#define mnist_h

#include <string>
#include <fstream>
#include <stdexcept>
#include <iostream>

namespace mnist {

    using byte = unsigned char;

    class DB {
    public:
        DB(const std::string &label_file, const std::string &image_file):
                labels(label_file),
                images(image_file) {
            if (!labels.is_open()) {
                throw std::runtime_error("unable to open " + label_file);
            }

            if (!images.is_open()) {
                throw std::runtime_error("unable to open " + label_file);
            }

            reset();
        }

        byte next_label() {
            labels.read((char*)&label, LBL_SIZE);
            if (!labels) {
                throw std::runtime_error("unable to read label");
            }
            return label;
        }

        byte *next_image() {
            images.read((char*)&image, IMG_SIZE);
            if (!images) {
                throw std::runtime_error("unable to read label");
            }
            return image;
        }

        void reset() {
            labels.seekg(LBL_HEADER_SIZE, labels.beg);
            images.seekg(IMG_HEADER_SIZE, images.beg);
        }

    private:
        static constexpr size_t LBL_HEADER_SIZE    = 8;
        static constexpr size_t IMG_HEADER_SIZE    = 16;
        static constexpr size_t LBL_SIZE           = 1;
        static constexpr size_t IMG_SIZE           = 28*28;

        byte label;
        byte image[IMG_SIZE];

        std::ifstream labels;
        std::ifstream images;
    };
}

#endif