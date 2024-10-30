#include "DeepFinanceDL/layers/dense.h"
#include <random>
#include <cmath>

namespace DeepFinanceDL {
namespace Layers {

Dense::Dense(int input_size, int output_size, std::shared_ptr<Activations::Activation> activation)
    : input_size_(input_size), output_size_(output_size), activation_(activation) {

    double limit = std::sqrt(6.0 / (input_size + output_size));
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-limit, limit);

    weights_ = Eigen::MatrixXd(input_size, output_size).unaryExpr([&](double) { return dis(gen); });
    biases_ = Eigen::VectorXd(output_size).unaryExpr([&](double) { return dis(gen); });
}

Eigen::MatrixXd Dense::forward(const Eigen::MatrixXd& input) {
    input_cache_ = input;
    Z_cache_ = (input * weights_).rowwise() + biases_.transpose();
    return activation_->forward(Z_cache_);
}

Eigen::MatrixXd Dense::backward(const Eigen::MatrixXd& grad_output, double learning_rate) {
    Eigen::MatrixXd activation_grad = activation_->backward(Z_cache_);
    Eigen::MatrixXd grad_Z = grad_output.cwiseProduct(activation_grad);

    grad_weights_ = input_cache_.transpose() * grad_Z;
    grad_biases_ = grad_Z.colwise().sum();

    weights_ -= learning_rate * grad_weights_;
    biases_ -= learning_rate * grad_biases_;

    Eigen::MatrixXd grad_input = grad_Z * weights_.transpose();
    return grad_input;
}

} 
}
