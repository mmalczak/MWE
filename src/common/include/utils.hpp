#include <cstdint>
#include <string>
#include <vector>

namespace utils {

void print(int32_t M, int32_t N, float *data);

template <typename T> void read_blob_file(const std::string &filename, std::vector<T> &out);

template <typename T> void load(std::string filename, T *dst);

} // namespace utils
