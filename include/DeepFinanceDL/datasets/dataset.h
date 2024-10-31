#ifndef DEEPFINANCEDL_DATASETS_DATASET_H
#define DEEPFINANCEDL_DATASETS_DATASET_H

#include<eigen3/Eigen/Dense>
#include<string>

namespace DeepFinanceDL
{
    namespace Datasets
    {
        class Dataset {
            public:
                virtual ~Dataset() {}
                virtual bool load_data(const std::string& filepath) = 0;
                virtual Eigen::MatrixXd get_features() const = 0;
                virtual Eigen::MatrixXd get_labels() const = 0;
            };
    }
}

#endif

