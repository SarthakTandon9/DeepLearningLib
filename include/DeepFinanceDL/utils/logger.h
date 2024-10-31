#ifndef DEEPFINANCEDL_UTILS_LOGGER_H
#define DEEPFINANCEDL_UTILS_LOGGER_H

#include <string>

namespace DeepFinanceDL {
    namespace Utils {
        class Logger {
        public:
            static void log(const std::string& message);
        };
    } // namespace Utils
} // namespace DeepFinanceDL

#endif // DEEPFINANCEDL_UTILS_LOGGER_H
