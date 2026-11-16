#pragma once
#include "json.hpp"
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace quick {
using Json = nlohmann::json;
struct Error : std::runtime_error {
    std::string type;
    Error(std::string kind, const std::string &message)
        : std::runtime_error(message), type(std::move(kind)) {}
};
struct Array {
    std::string dtype;
    std::vector<std::size_t> shape;
    std::vector<std::uint8_t> bytes;
    std::size_t rows() const { return shape.empty() ? 1 : shape.front(); }
    std::size_t itemsize() const;
};
using Arrays = std::map<std::string, Array>;
struct Frame {
    std::uint8_t kind = 1;
    std::uint32_t request_id = 0;
    Json meta = Json::object();
    Arrays arrays;
};
constexpr std::uint8_t REQUEST = 1, RESPONSE = 2, DATA = 3, DONE = 4, ERROR = 5;
constexpr std::size_t MAX_METADATA = 1024 * 1024, MAX_PAYLOAD = 64 * 1024 * 1024, MAX_ARRAYS = 4096;
class Backend {
  public:
    virtual ~Backend() = default;
    virtual Json get_cfg() const = 0;
    virtual Frame call(const std::string &op, const Json &args = Json::object(),
                       const Arrays &arrays = {}) = 0;
};
// Byte-for-byte Python json.dumps(sort_keys=True, separators=(',', ':'), ensure_ascii=True).
std::string canonical_json(const Json &value);
Json parse_json(const std::string &text);
std::string sha256(const std::string &bytes);
std::string sha256(const std::vector<std::uint8_t> &bytes);
std::string sha256_parts(const std::vector<std::vector<std::uint8_t>> &parts);
std::string sha256_file(const std::string &path);
} // namespace quick
