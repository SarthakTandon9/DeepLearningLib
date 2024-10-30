#ifndef TENSOR_H
#define TENSOR_H

#include <eigen3/Eigen/Dense>
#include<vector>

namespace DeepFinanceDL
{
    namespace Core
    {
        class Tensor
        {
            public:
                //constructors
                Tensor() = default;
                Tensor(int rows, int cols);
                Tensor(const Eigen::MatrixXd& matrix);

                //initialization
                void random_init(double min = -1.0, double max = 1.0);
                void zeroes_init();
                void ones_init();

                //getter
                Eigen::MatrixXd& get_matrix();
                const Eigen::MatrixXd& get_matrix() const;

                Tensor operator*(const Tensor& other) const;
                Tensor operator+(const Tensor& other) const; // Matrix addition
                Tensor operator-(const Tensor& other) const; // Matrix subtraction
                Tensor operator*(double scalar) const;        // Scalar multiplication

                int rows() const;
                int cols() const;
            private:
                Eigen::MatrixXd matrix_;
        };
    }
}

#endif