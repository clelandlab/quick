#pragma once
#include "quick/common.hpp"
#include <memory>

namespace quick {
// Owns all TCP sessions and joins acquisition workers before releasing hardware.
class Server {
public:
    explicit Server(Backend& backend, const Json& settings = Json::object());
    ~Server();
    Server(const Server&) = delete;
    Server& operator=(const Server&) = delete;
    void run();
    void request_stop() noexcept;
    void shutdown();
    unsigned short port() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
