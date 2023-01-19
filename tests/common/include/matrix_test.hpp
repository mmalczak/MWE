#include "src/cuda/matrix/include/matrix.hpp"
#include "tests/common/include/blob_files.hpp"

class MatrixTest {
public:
  matrix::Matrix construct_matrix(int32_t M, int32_t N);

  matrix::Matrix construct_matrix(int32_t M, int32_t N, std::string path);

  void load_data(std::string tc_name);

  void free_data();

  matrix::Matrix a;
  matrix::Matrix b;
  matrix::Matrix c;
};
