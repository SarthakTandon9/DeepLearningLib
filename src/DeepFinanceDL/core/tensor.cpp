#include "DeepFinanceDL/core/tensor.h"
#include <random>

namespace DeepFinanceDL
{
    namespace Core
    {
        Tensor::Tensor(int rows, int cols) : matrix_(rows, cols) {}
        Tensor::Tensor(const Eigen::MatrixXd& matrix) : matrix_(matrix) {}

        void Tensor::random_init(double min, double max)
        {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<> dis(min, max);
            matrix_ = Eigen::MatrixXd::NullaryExpr(matrix_.rows(), matrix_.cols(), [&]() { return dis(gen); });
        }

        void Tensor::zeroes_init()
        {
            matrix_.setZero();
        } 
        
        void Tensor::ones_init() 
        {
            matrix_.setOnes();
        }     
        
        Eigen::MatrixXd& Tensor::get_matrix()
        {
            return matrix_;
        } 

        const Eigen::MatrixXd& Tensor::get_matrix() const
        {
            return matrix_;
        } 

        Tensor Tensor::operator*(const Tensor& other) const
        {
            return Tensor(this->matrix_ * other.matrix_);
        } 

        Tensor Tensor::operator+(const Tensor& other) const
        {
            return Tensor(this->matrix_ + other.matrix_);
        }

        Tensor Tensor::operator-(const Tensor& other) const
        {
            return Tensor(this->matrix_ - other.matrix_);
        }

        Tensor Tensor::operator*(double scalar) const
        {
            return Tensor(this->matrix_ * scalar);
        }      

        int Tensor::rows() const
        {
            return matrix_.rows();
        }

        int Tensor::cols() const
        {
            return matrix_.cols();
        }
    }
}

