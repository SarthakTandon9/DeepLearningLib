#include "DeepFinanceDL/models/feedforward.h"
#include <iostream>

namespace DeepFinanceDL
{
    namespace Models
    {
        Feedforward::Feedforward(std::shared_ptr<Optimizers::Optimizer> optimizer)
            : optimizer_(optimizer) {}

        void Feedforward::add_layer(std::shared_ptr<Layers::Layer> layer) {
            layers_.emplace_back(layer);
        }

        Eigen::MatrixXd Feedforward::predict(const Eigen::MatrixXd& input) {
            Eigen::MatrixXd output = input;
            for (auto& layer : layers_) {
                output = layer->forward(output);
            }
            return output;
        }

        void Feedforward::train(const Eigen::MatrixXd& X, const Eigen::MatrixXd& y, int epochs, double learning_rate) {
            for (int epoch = 0; epoch < epochs; ++epoch) {
                // Forward pass
                Eigen::MatrixXd output = predict(X);

                // Compute loss (MSE)
                Eigen::MatrixXd loss = output - y;
                double mse = loss.array().square().mean();

                // Backward pass
                Eigen::MatrixXd grad = 2.0 * loss / loss.rows();
                for (auto it = layers_.rbegin(); it != layers_.rend(); ++it) {
                    grad = (*it)->backward(grad, learning_rate);
                }

                // Optionally, use the optimizer to update parameters if implemented separately
                // optimizer_->update(layers_, learning_rate);

                // Logging
                if ((epoch + 1) % 100 == 0 || epoch == 0) {
                    std::cout << "Epoch " << epoch + 1 << "/" << epochs << " - MSE: " << mse << std::endl;
                }
            }
        }
    }
}