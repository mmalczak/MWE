#include <iostream>
#include <mwe/matrix.hpp>

int main()
{
    matrix::Matrix* m = matrix::generateMatrix(2, 2);
    matrix::printMatrix(2, 2, m->data);
    float data[] = {1, 2, 3, 4};
    matrix::printMatrix(2, 2, data);
    return 0;
}
