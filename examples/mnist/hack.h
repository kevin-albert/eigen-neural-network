#ifndef _hack_h
#define _hack_h

#include <string>
#include <iostream>

// 
// <HACK> 
// argument parsing code
// please don't judge
//
#define fail_usage(message) \
    {std::cerr << message << "\n"; print_usage_exit(argv[0]);}
#define gen_case_arg(flag, var, test, parse_fn) case flag:\
    var = parse_fn(argv[0], #var);\
    if (!(test)) {\
        std::cerr << "invalid " #var ": " << ::optarg\
                  << ", failed check: " #test;\
        print_usage_exit(argv[0]);\
    }\
    break
#define case_int_arg(flag, var, test) gen_case_arg(flag, var, test, parse_int)
#define case_float_arg(flag, var, test) gen_case_arg(flag, var, test, parse_float)

void print_usage_exit(char *execname) {
    std::cerr << "usage: " << execname << " [-easinNbEv]\n";
    exit(EXIT_FAILURE);
}

int parse_int(char *execname, const std::string &name) {
    const char *p = ::optarg;
    if (!p) goto fail;
    if (*p == '-') ++p;
    if (!p) goto fail;
    while (*p) {
        if (*p < '0' || *p > '9') goto fail;
        ++p;
    }
    return atoi(::optarg);
fail:
    std::cerr << "invalid value for " << name << ": " << ::optarg 
              << " (expected integer)\n";
    print_usage_exit(execname);
    return 0;
}

float parse_float(char *execname, const std::string &name) {
    const char *p = ::optarg;
    if (!p) goto fail;
    if (*p == '-') ++p;
    if (!p) goto fail;
    while (*p) {
        if (*p == '.') { ++p; break; }
        if (*p < '0' || *p > '9') goto fail;
        ++p;
    }
    while (*p) {
        if (*p < '0' || *p > '9') goto fail;
        ++p;
    }
    return atof(::optarg);
fail:
    std::cerr << "invalid value for " << name << ": " << ::optarg 
              << " (expected number)\n";
    print_usage_exit(execname);
    return 0;
}


#endif