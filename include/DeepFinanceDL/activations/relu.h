#ifndef DEEPFINANCEDL_ACTIVATIONS_RELU_H
#define DEEPFINANCEDL_ACTIVATIONS_RELU_H
#include "activation.h"
#include <eigen3/Eigen/Dense>

namespace DeepFinanceDL {
namespace Activations {

class ReLU : public Activation{
public:
    Eigen::MatrixXd forward(const Eigen::MatrixXd& Z);
    Eigen::MatrixXd backward(const Eigen::MatrixXd& Z);
};

} 
} 

#endif 
