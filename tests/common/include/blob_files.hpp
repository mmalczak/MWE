#include <string>
#include <vector>

namespace blob_files {

template <typename T> void read(const std::string &filename, std::vector<T> &out);

template <typename T> void load(std::string filename, T *dst);

} // namespace blob_files
