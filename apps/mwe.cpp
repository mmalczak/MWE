#include "matrix.hpp"

#include <iostream>

int main(int argc, char *argv[]) {
  matrix::Matrix *m = matrix::generateMatrix(2, 2);
  matrix::printMatrix(2, 2, m->data);
  float data[] = {1, 2, 3, 4};
  matrix::printMatrix(2, 2, data);
}
