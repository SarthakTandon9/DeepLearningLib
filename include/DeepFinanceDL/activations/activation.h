#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <eigen3/Eigen/Dense>

namespace DeepFinanceDL
{
    namespace Avtivations
    {
        class Activation
        {
            public:
                virtual ~Activation() {}
                virtual Eigen::MatrixXd forward(const Eigen::MatrixXd& Z) = 0;
                virtual Eigen::MatrixXd backward(const Eigen::MatrixXd& Z) = 0;
        };
    }
}

#endif