#ifndef DEEPFINANCEDL_MODELS_FEEDFORWARD_H
#define DEEPFINANCEDL_MODELS_FEEDFORWARD_H

#include "model.h"
#include "DeepFinanceDL/layers/layer.h"
#include "DeepFinanceDL/optimizers/optimizer.h"


namespace DeepFinanceDL
{
    namespace Models
    {
        class Feedforward : public Model
        {
            public:
                Feedforward(std::shared_ptr<Optimizers::Optimizer> optimizer);
                ~Feedforward() override {}

                void add_layer(std::shared_ptr<Layers::Layer> layer) override;
                Eigen::MatrixXd predict(const Eigen::MatrixXd& input) override;
                void train(const Eigen::MatrixXd& X, const Eigen::MatrixXd& y, int epochs, double learning_rate) override;
            private:
                std::vector<std::shared_ptr<Layers::Layer>> layers_;
                std::shared_ptr<Optimizers::Optimizer> optimizer_;
        };
    }
}

#endif