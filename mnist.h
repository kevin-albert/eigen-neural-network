#ifndef mnist_h
#define mnist_h

#include <functional>
#include <string>

namespace mnist {

    using byte = unsigned char;

    // scan a dataset and invoke a callback function for each image / label pair
    void scan(const std::string &label_file, const std::string &image_file, 
              const int N, std::function<void(const byte, const byte*)> fn);

}

#endif