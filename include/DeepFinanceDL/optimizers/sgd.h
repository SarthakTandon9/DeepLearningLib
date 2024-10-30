#ifndef DEEPFINANCEDL_OPTIMIZERS_SGD_H
#define DEEPFINANCEDL_OPTIMIZERS_SGD_H

#include "optimizer.h"

namespace DeepFinanceDL {
namespace Optimizers {

class SGD : public Optimizer {
public:
    void update(std::vector<std::shared_ptr<Layers::Layer>>& layers, double learning_rate) override;
};

} 
} 

#endif 
