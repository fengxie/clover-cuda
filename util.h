
#pragma once

#include <iostream>

typedef enum {
    SUCCESS = 0,
    SUCCESS_NO_REFCHECK = 1,
    FAIL_UNKNOWN = 10,
    FAIL_REFCHECK = 11,
    FAIL_CUDA_RUNTIME = 100,
} Error;

template<typename ... Args>
inline void _assert(bool cond, int code, const char *file, int line, const std::string &format, Args ... args) {
    // FIXME: need to move this buffer to one single .cpp file
    static char buf[1024];

    if (!cond) {
        snprintf(buf, 1024, format.c_str(), args ...);
        std::cerr << file << ":" << line << " ASSERT: " << buf << std::endl;
        exit(code);
    }
}

#define CUDA_ASSERT(cond, format, ...)                                                            \
    do {                                                                                          \
        cudaError_t __err = (cond);                                                               \
        _assert((__err == cudaSuccess), FAIL_CUDA_RUNTIME, __FILE__, __LINE__, \
                ("%s = %d: %s: " format), #cond, __err, cudaGetErrorString(__err), ## __VA_ARGS__);   \
    } while (0)

