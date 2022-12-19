#include "common/include/utils.hpp"

#include <cstdint>
#include <fstream>
#include <cstdio>
#include <string>
#include <vector>

namespace utils {

void print(int32_t M, int32_t N, float *data) {
  for (int32_t j = 0; j < M; j++) {
    for (int32_t i = 0; i < N; i++) {
      printf("%f ", *data++);
    }
    printf("\n");
  }
  printf("\n");
}

template <typename T> void read_blob_file(const std::string &filename, std::vector<T> &out) {
  const std::string blobs_dir = std::getenv("BLOBS_DIR");
  if (blobs_dir.empty()) {
    fprintf(stderr, "Cannot read BLOBS_DIR env variable!\n");
    exit(1);
  }
  const std::string blob_extension = ".bin";

  std::string full_path = blobs_dir + filename + blob_extension;
  std::ifstream file(full_path, std::ios::in | std::ios::binary | std::ios::ate);
  if (file.fail() || !file.is_open()) {
    fprintf(stderr, "Cannot open file (%s) or file doesn't exist!\n", full_path.c_str());
    exit(1);
  }

  std::streampos size = file.tellg();

  file.seekg(0, std::ios::beg);

  uint32_t items = size / (sizeof(T) / sizeof(char));
  for (uint32_t i = 0; i < items; i++) {
    T memblock{};

    file.read(reinterpret_cast<char *>(&memblock), sizeof(T));
    out.push_back(T(memblock));
  }

  file.close();
}

template <typename T> void load(std::string filename, T *dst) {
  {
    std::vector<T> d;
    read_blob_file(filename, d);
    std::copy(d.begin(), d.end(), dst);
  }
}

template void read_blob_file(const std::string &path, std::vector<float> &out);
template void read_blob_file(const std::string &path, std::vector<int8_t> &out);
template void read_blob_file(const std::string &path, std::vector<int16_t> &out);
template void read_blob_file(const std::string &path, std::vector<int32_t> &out);

template void load(std::string filename, float *dst);
template void load(std::string filename, int8_t *dst);
template void load(std::string filename, int16_t *dst);
template void load(std::string filename, int32_t *dst);

} // namespace utils
