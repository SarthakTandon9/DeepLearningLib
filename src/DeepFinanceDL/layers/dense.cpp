#include "DeepFinanceDL/layers/dense.h"
#include <random>
#include <cmath>

namespace DeepFinanceDL {
namespace Layers {

// Constructor: Initialize weights and biases with Xavier initialization
Dense::Dense(int input_size, int output_size, std::shared_ptr<Activations::Activation> activation)
    : input_size_(input_size), output_size_(output_size), activation_(activation) {

    double limit = std::sqrt(6.0 / (input_size + output_size));
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-limit, limit);

    weights_ = Eigen::MatrixXd(input_size, output_size).unaryExpr([&](double) { return dis(gen); });
    biases_ = Eigen::VectorXd(output_size).unaryExpr([&](double) { return dis(gen); });
}

// Forward pass
Eigen::MatrixXd Dense::forward(const Eigen::MatrixXd& input) {
    input_cache_ = input;
    Z_cache_ = (input * weights_).rowwise() + biases_.transpose();
    if (activation_) {
        return activation_->forward(Z_cache_);
    } else {
        return Z_cache_; // Linear activation
    }
}

// Backward pass
Eigen::MatrixXd Dense::backward(const Eigen::MatrixXd& grad_output, double learning_rate) {
    Eigen::MatrixXd grad_Z;

    if (activation_) {
        // Compute gradient of activation
        Eigen::MatrixXd activation_grad = activation_->backward(Z_cache_);
        grad_Z = grad_output.cwiseProduct(activation_grad);
    } else {
        grad_Z = grad_output; // Linear activation derivative is 1
    }

    // Compute gradients w.r.t weights and biases
    grad_weights_ = input_cache_.transpose() * grad_Z;
    grad_biases_ = grad_Z.colwise().sum();

    // Update weights and biases
    weights_ -= learning_rate * grad_weights_;
    biases_ -= learning_rate * grad_biases_;

    // Compute gradient to pass to the previous layer
    Eigen::MatrixXd grad_input = grad_Z * weights_.transpose();
    return grad_input;
}

} // namespace Layers
} // namespace DeepFinanceDL
