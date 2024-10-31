#ifndef DEEPFINANCEDL_MODELS_MODEL_H
#define DEEPFINANCEDL_MODELS_MODEL_H

#include <vector>
#include <memory>
#include "DeepFinanceDL/layers/layer.h"
#include "DeepFinanceDL/optimizers/optimizer.h"

namespace DeepFinanceDL
{
    namespace Models
    {
        class Model
        {
            public:
                virtual ~Model(){}
                virtual void add_layer(std::shared_ptr<Layers::Layer> layer) = 0;
                virtual Eigen::MatrixXd predict(const Eigen::MatrixXd& input) = 0;
                virtual void train(const Eigen::MatrixXd& X, const Eigen::MatrixXd& y, int epochs, double learning_rate) = 0;
        };
    }
}

#endif