#ifndef DEEPFINANCEDL_LAYERS_DENSE_H
#define DEEPFINANCEDL_LAYERS_DENSE_H

#include "layer.h"
#include "DeepFinanceDL/activations/activation.h"

namespace DeepFinanceDL {
namespace Layers {

class Dense : public Layer {
public:
    Dense(int input_size, int output_size, std::shared_ptr<Activations::Activation> activation);
    ~Dense() override {}

    Eigen::MatrixXd forward(const Eigen::MatrixXd& input) override;
    Eigen::MatrixXd backward(const Eigen::MatrixXd& grad_output, double learning_rate) override;

private:
    int input_size_;
    int output_size_;
    Eigen::MatrixXd weights_;
    Eigen::VectorXd biases_;
    Eigen::MatrixXd input_cache_;
    Eigen::MatrixXd Z_cache_;

    // Gradients
    Eigen::MatrixXd grad_weights_;
    Eigen::VectorXd grad_biases_;

    // Activation function
    std::shared_ptr<Activations::Activation> activation_;
};

} // namespace Layers
} // namespace DeepFinanceDL

#endif // DEEPFINANCEDL_LAYERS_DENSE_H
