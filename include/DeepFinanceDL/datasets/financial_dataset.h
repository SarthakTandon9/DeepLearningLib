#ifndef DEEPFINANCEDL_DATASETS_FINANCIALDATASET_H
#define DEEPFINANCEDL_DATASETS_FINANCIALDATASET_H

#include "dataset.h"

namespace DeepFinanceDL
{
    namespace Datasets
    {
        class FinancialDataset : public Dataset
        {
            public:
                bool load_data(const std::string& filepath) override;
                Eigen::MatrixXd get_features() const override;
                Eigen::MatrixXd get_labels() const override;

            private:
                Eigen::MatrixXd features_;
                Eigen::MatrixXd labels_;
        };
    }
}

#endif