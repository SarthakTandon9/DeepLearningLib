#include "DeepFinanceDL/optimizers/sgd.h"

// Note: In our current layer implementation, the Dense layer updates its own weights in the backward pass.
// Therefore, the optimizer might not need to do anything. However, if you choose to separate parameter updates,
// you can implement it here.

namespace DeepFinanceDL {
namespace Optimizers {

void SGD::update(std::vector<std::shared_ptr<Layers::Layer>>& layers, double learning_rate) {
   
}

}
} 
