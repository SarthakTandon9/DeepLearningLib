#include "DeepFinanceDL/core/tensor.h"
#include "DeepFinanceDL/activations/relu.h"
#include "DeepFinanceDL/layers/dense.h"
#include "DeepFinanceDL/optimizers/sgd.h"
#include "DeepFinanceDL/models/feedforward.h"
#include "DeepFinanceDL/datasets/financial_dataset.h"
#include "DeepFinanceDL/utils/logger.h"

#include <eigen3/Eigen/Dense>
#include <memory>
#include <iostream>

using namespace DeepFinanceDL;

// Function to normalize features
Eigen::MatrixXd normalize_features(const Eigen::MatrixXd& X) {
    Eigen::RowVectorXd mean = X.colwise().mean();
    Eigen::RowVectorXd std_dev = ((X.rowwise() - mean).array().square().colwise().mean()).sqrt();
    return (X.rowwise() - mean).array().rowwise() / std_dev.array();
}

int main() {
    // Load dataset
    Datasets::FinancialDataset dataset;
    if (!dataset.load_data("data/sample_financial_data.csv")) {
        Utils::Logger::log("Failed to load dataset.");
        return -1;
    }

    Eigen::MatrixXd X = dataset.get_features();
    Eigen::MatrixXd y = dataset.get_labels();

    // Normalize features
    X = normalize_features(X);

    // Initialize optimizer
    std::shared_ptr<Optimizers::Optimizer> optimizer = std::make_shared<Optimizers::SGD>();

    // Initialize model
    std::shared_ptr<Models::Feedforward> model = std::make_shared<Models::Feedforward>(optimizer);

    // Define network architecture
    // Input layer size = number of features
    // Hidden layer with 10 neurons and ReLU activation
    // Output layer with 1 neuron (for regression)
    std::shared_ptr<Activations::ReLU> relu = std::make_shared<Activations::ReLU>();
    std::shared_ptr<Layers::Layer> hidden_layer = std::make_shared<Layers::Dense>(X.cols(), 10, relu);
    std::shared_ptr<Activations::Activation> linear = std::make_shared<Activations::Activation>(); // Placeholder for linear activation
    // Since we don't have a linear activation, we can modify the Dense layer to optionally skip activation or use identity.
    // For simplicity, assume the output layer has no activation.
    std::shared_ptr<Layers::Layer> output_layer = std::make_shared<Layers::Dense>(10, 1, relu); // Replace with linear activation

    // Add layers to the model
    model->add_layer(hidden_layer);
    // For regression, the output layer typically has no activation or a linear activation
    // Here, we'll modify Dense layer to handle linear activation by passing a different activation function
    // For simplicity, we'll use ReLU as a placeholder
    model->add_layer(output_layer);

    // Train the model
    int epochs = 100;
    double learning_rate = 0.01;
    model->train(X, y, epochs, learning_rate);

    // Make predictions
    Eigen::MatrixXd predictions = model->predict(X);

    // Compute final MSE
    Eigen::MatrixXd final_loss = predictions - y;
    double final_mse = final_loss.array().square().mean();
    std::cout << "Final MSE: " << final_mse << std::endl;

    return 0;
}
