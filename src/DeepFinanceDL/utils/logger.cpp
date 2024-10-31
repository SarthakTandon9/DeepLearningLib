#include "DeepFinanceDL/utils/logger.h"
#include <iostream>

namespace DeepFinanceDL {
    namespace Utils {
        void Logger::log(const std::string& message) {
            std::cout << message << std::endl;
        }
    } // namespace Utils
} // namespace DeepFinanceDL
