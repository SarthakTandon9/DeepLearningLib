#include <Eigen/Dense>
#include <iostream>

int main() {
    Eigen::Matrix2d mat;
    mat << 1, 2,
           3, 4;

    std::cout << "Eigen is installed and working properly.\n";
    std::cout << "Here is a 2x2 matrix:\n" << mat << std::endl;

    return 0;
}
