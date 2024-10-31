#include "DeepFinanceDL/activations/relu.h"

namespace DeepFinanceDL {
namespace Activations {

Eigen::MatrixXd ReLU::forward(const Eigen::MatrixXd& Z) {
    return Z.array().max(0.0);
}

Eigen::MatrixXd ReLU::backward(const Eigen::MatrixXd& Z) {
    Eigen::MatrixXd derivative = (Z.array() > 0).cast<double>();
    return derivative;
}

} // namespace Activations
} // namespace DeepFinanceDL
