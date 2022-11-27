#!/bin/sh

echo "C/C++ code formatting..."
find -type f -not -path "./build/*" -name "*.cc" -o -name "*.hpp" -o -name "*.cpp" | xargs clang-format -i -Werror
echo "Done"

#echo "Python code formatting..."
#find -type f -name "*.py" | xargs black -q
#echo "Done"

echo "CMake files formatting..."
find -type f -not -path "./build/*" -name "CMakeLists.txt" -o -name "*.cmake" | xargs cmake-format -i -l info
echo "Done"
