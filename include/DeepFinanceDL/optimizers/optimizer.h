#ifndef DEEPFINANCEDL_OPTIMIZERS_OPTIMIZER_H
#define DEEPFINANCEDL_OPTIMIZERS_OPTIMIZER_H

#include <vector>
#include <memory>
#include "DeepFinanceDL/layers/layer.h"

namespace DeepFinanceDL {
namespace Optimizers {

class Optimizer {
public:
    virtual ~Optimizer() {}
    virtual void update(std::vector<std::shared_ptr<Layers::Layer>>& layers, double learning_rate) = 0;
};

}
} 

#endif
