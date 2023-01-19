#ifndef __MATRIX_HPP__
#define __MATRIX_HPP__

#include <cstdint>

namespace matrix {

struct Matrix {
  int32_t M, N;
  float *data;
};

} // namespace matrix

#endif
