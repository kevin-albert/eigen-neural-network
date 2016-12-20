#include <iostream>
#include <iomanip>
#include <unistd.h>
#include <cmath>
#include <signal.h>

#include "mnist.h"
#include "network.h"
#include "hack.h"

const int MAX_TRAIN_SIZE = 60000;
const int MAX_TEST_SIZE = 10000;
const double pi = 3.1415926535897;


// params
float eta = 0.1;
float alpha = 0;
int num_train = MAX_TRAIN_SIZE;
int num_test = MAX_TEST_SIZE;
int batch_size = 1;
float batch_flux_rate = 0.2;
float batch_flux_amount = 0;
float batch_size_decay = 1;
int num_epochs = 1;
bool verbose = false;

volatile bool has_signal = false;
void onsignal(int);
void menu();

int main(int argc, char **argv) {



    signal(SIGINT, onsignal);

    int c;
    while ((c = getopt(argc, argv, "e:a:t:n:b:f:F:d:E:vh")) != -1) {
        switch (c) {
            case_float_arg('e', eta, eta > 0 && eta <= 1);
            case_float_arg('a', alpha, alpha >= 0 && alpha <= 1);
            case_int_arg('t', num_train, num_train >= 0 && num_train <= MAX_TRAIN_SIZE);
            case_int_arg('n', num_test, num_test >= 0 && num_test <= MAX_TEST_SIZE);
            case_int_arg('b', batch_size, batch_size > 0);
            case_float_arg('f', batch_flux_rate, batch_flux_rate > 0 && batch_flux_rate < 1);
            case_float_arg('F', batch_flux_amount, true);
            case_float_arg('d', batch_size_decay, batch_size_decay >= 0);
            case_int_arg('E', num_epochs, num_epochs > 0);
            case 'h':
                std::cout << "usage: " << argv[0] << " [-eatnbEvh]\n"
                          << "    -e eta                [0.1]\n"
                          << "    -a alpha              [0.0]\n"
                          << "    -t num_train          [60000]\n"
                          << "    -n num_test           [10000]\n"
                          << "    -b batch_size         [1]\n"
                          << "    -f batch_flux_rate    [0.2]\n"
                          << "    -F batch_flux_amount  [0.0]\n"
                          << "    -d batch_size_decay   [0.0]\n"
                          << "    -E num_epochs         [1]\n"
                          << "    -v verbose            [false]\n"
                          << "    -h help\n"
                          << "\n";
                return EXIT_SUCCESS;
            case 'v':
                verbose = true;
                break;
            case '?':
                print_usage_exit(argv[0]);
            default:
                abort();
        }
    }

    std::cout << "parameters:\n"
              << "    eta: (-e)                 " << eta << "\n"
              << "    alpha: (-a)               " << alpha << "\n"
              << "    num_train: (-t)           " << num_train << "\n"
              << "    num_test: (-n)            " << num_test << "\n"
              << "    batch_size: (-b)          " << batch_size << "\n"
              << "    batch_flux_rate: (-f)     " << batch_flux_rate << "\n"
              << "    batch_flux_amount: (-F)   " << batch_flux_amount << "\n"
              << "    batch_size_decay: (-d)    " << batch_size_decay << "\n"
              << "    num_epochs: (-E)          " << num_epochs << "\n"
              << "    threads:                  " << Eigen::nbThreads() << "\n"
              << "\n";
    
    Network network;
    mnist::DB train_data("train-labels-idx1-ubyte", "train-images-idx3-ubyte");
    mnist::DB test_data("t10k-labels-idx1-ubyte", "t10k-images-idx3-ubyte");

    int batch_idx = 0;
    double error_rate = 1;

    for (int epoch = 0; epoch < num_epochs; ++epoch) {

        double multiplier = batch_size_decay == 0 ? 1 : 
                std::pow(1.0 - (double) epoch / num_epochs, batch_size_decay);

        int real_batch_size = std::max(
                            1,
                            (int) (
                            batch_size 
                                * multiplier 
                                * (1.0 + batch_flux_amount * std::sin(2.0 * pi * epoch * batch_flux_rate)) / 2));
                  
        //
        // train
        //
        for (int i = 0; i < num_train; ++i) {
            if (has_signal) {
                menu();
            }
            network.set_label(train_data.next_label());
            network.set_image(train_data.next_image());
            network.forwardpass();

            if (real_batch_size == 1) {
                network.backwardpass(eta, alpha);
            } else {
                network.batch_backwardpass();
                if (++batch_idx == real_batch_size) {
                    network.batch_update_reset(eta, alpha);
                    batch_idx = 0;
                }
            }
        }

        //
        // test
        // 
        int error_count = 0;
        
        for (int i = 0; i < num_test; ++i) {
            if (has_signal) {
                menu();
            }
            auto label = test_data.next_label();
            network.set_label(label);
            network.set_image(test_data.next_image());
            network.forwardpass();
            if (label != network.get_output()) {
                ++error_count;
            }
        }

        error_rate = (double) error_count / num_test;

        std::cout << "epoch: " << std::setw(8) << (epoch+1) << ", "
                  << "error rate: " << error_rate
                  << "\n";

        train_data.reset();
        test_data.reset();
    }

}


void onsignal(int signal) {
    if (has_signal) {
        // quit 
        std::cout << "\n";
        exit(EXIT_SUCCESS);
    }
    has_signal = true;
}


void menu() {
    std::cout << "\n"
              << "parameters:\n"
              << "    eta: (-e)                 " << eta << "\n"
              << "    alpha: (-a)               " << alpha << "\n"
              << "    num_train: (-t)           " << num_train << "\n"
              << "    num_test: (-n)            " << num_test << "\n"
              << "    batch_size: (-b)          " << batch_size << "\n"
              << "    batch_flux_rate: (-f)     " << batch_flux_rate << "\n"
              << "    batch_flux_amount: (-F)   " << batch_flux_amount << "\n"
              << "    batch_size_decay: (-d)    " << batch_size_decay << "\n"
              << "    num_epochs: (-E)          " << num_epochs << "\n"
              << "    threads:                  " << Eigen::nbThreads() << "\n";

    char option = '?';
menu_select:
    std::cout << "\n[c]ontinue, [q]uit, or [e]dit: ";
    std::cin >> option;

    switch (option) {
        case 'e':
            std::cout << "flag to modify: ";
            std::cin >> option;
            switch (option) {
                    case_input_value('e', eta, eta > 0 && eta <= 1);
                    case_input_value('a', alpha, alpha >= 0 && alpha <= 1);
                    case_input_value('t', num_train, num_train >= 0 && num_train <= MAX_TRAIN_SIZE);
                    case_input_value('n', num_test, num_test >= 0 && num_test <= MAX_TEST_SIZE);
                    case_input_value('b', batch_size, batch_size > 0);
                    case_input_value('f', batch_flux_rate, batch_flux_rate > 0 && batch_flux_rate < 1);
                    case_input_value('F', batch_flux_amount, true);
                    case_input_value('d', batch_size_decay, batch_size_decay >= 0);
                    case_input_value('E', num_epochs, num_epochs > 0);
                    default:
                        std::cout << "invalid option: " << option << "\n";
                        goto menu_select;
            }
            break;
        case 'c':
            break;
        case 'q':
            exit(EXIT_SUCCESS);
        default:
            goto menu_select;
    }

    has_signal = false;
    return;
}