#include "DeepFinanceDL/datasets/financial_dataset.h"
#include <fstream>
#include <sstream>
#include <vector>
#include <iostream>


namespace DeepFinanceDL
{
    namespace Datasets
    {
        bool FinancialDataset::load_data(const std::string& filepath)
        {
            std::ifstream file(filepath);
            if (!file.is_open()) {
                std::cerr << "Failed to open data file: " << filepath << std::endl;
                return false;
            }

            std::string line;
            std::vector<std::vector<double>> data_vec;
            std::vector<std::vector<double>> label_vec;

            while (std::getline(file, line)) {
                std::stringstream ss(line);
                std::string value;
                std::vector<double> row;

                while (std::getline(ss, value, ',')) {
                    row.push_back(std::stod(value));
                }

                if (row.empty()) continue;

                // Assume last column is the label
                std::vector<double> features(row.begin(), row.end() - 1);
                std::vector<double> label = { row.back() };

                data_vec.emplace_back(features);
                label_vec.emplace_back(label);
            }

            int rows = data_vec.size();
            if (rows == 0) {
                std::cerr << "No data found in file: " << filepath << std::endl;
                return false;
            }

            int feature_cols = data_vec[0].size();
            features_ = Eigen::MatrixXd(rows, feature_cols);
            labels_ = Eigen::MatrixXd(rows, 1);

            for (int i = 0; i < rows; ++i) {
                features_.row(i) = Eigen::VectorXd::Map(&data_vec[i][0], feature_cols);
                labels_.row(i) = Eigen::VectorXd::Map(&label_vec[i][0], 1);
            }

            return true;

        }
        Eigen::MatrixXd FinancialDataset::get_features() const {
            return features_;
        }

        Eigen::MatrixXd FinancialDataset::get_labels() const {
            return labels_;
        }
    }
} // namespace DeepFinanceDL
