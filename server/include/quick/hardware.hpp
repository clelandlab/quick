#pragma once
#include "quick/common.hpp"
namespace quick {
std::unique_ptr<Backend> make_hardware(const Json &settings);
}
