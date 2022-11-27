#include <string>
#include <vector>

namespace utils {

void print(int M, int N, float *data);

template <typename T> void read_blob_file(const std::string &filename, std::vector<T> &out);

} // namespace utils
