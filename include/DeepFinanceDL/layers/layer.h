#ifndef DEEPFINANCEDL_LAYERS_LAYER_H
#define DEEPFINANCEDL_LAYERS_LAYER_H

#include <eigen3/Eigen/Dense>
#include <memory>
#include "DeepFinanceDL/core/tensor.h"
#include "DeepFinanceDL/activations/activation.h"

namespace DeepFinanceDL {
namespace Layers {

class Layer {
public:
    virtual ~Layer() {}
    virtual Eigen::MatrixXd forward(const Eigen::MatrixXd& input) = 0;
    virtual Eigen::MatrixXd backward(const Eigen::MatrixXd& grad_output, double learning_rate) = 0;
};

} 
} 

#endif // DEEPFINANCEDL_LAYERS_LAYER_H
