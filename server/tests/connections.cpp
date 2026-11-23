#include "quick/server.hpp"
#include "quick/wire.hpp"
#include <arpa/inet.h>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <exception>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <sys/socket.h>
#include <thread>

namespace {
using namespace quick;
using Clock = std::chrono::steady_clock;

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

class MockBackend : public Backend {
public:
    Json get_cfg() const override {
        return {{"gens", Json::array()}, {"readouts", Json::array()},
                {"tprocs", Json::array({{{"pmem_size", 8}, {"wmem_size", 8}, {"dmem_size", 8}}})}};
    }

    Frame call(const std::string& op, const Json& = Json::object(), const Arrays& = {}) override {
        if (op == "start_tproc") {
            std::lock_guard<std::mutex> lock(mutex);
            started = true;
            changed.notify_all();
        }
        if (op == "cleanup_round") {
            std::unique_lock<std::mutex> lock(mutex);
            cleanup_waiting = true;
            changed.notify_all();
            changed.wait(lock, [&] { return !cleanup_blocked; });
            ++cleanups;
        }
        Frame result;
        if (op == "get_tproc_counter") result.meta = {{"value", 0}};
        return result;
    }

    void wait_for_capture() {
        std::unique_lock<std::mutex> lock(mutex);
        require(changed.wait_for(lock, std::chrono::seconds(2), [&] { return started; }),
                "capture did not start");
    }

    void hold_cleanup(bool hold) {
        std::lock_guard<std::mutex> lock(mutex);
        cleanup_blocked = hold;
        changed.notify_all();
    }
    void wait_for_cleanup_start() {
        std::unique_lock<std::mutex> lock(mutex);
        require(changed.wait_for(lock, std::chrono::seconds(2), [&] { return cleanup_waiting; }),
                "evicted capture did not begin cleanup");
    }

    std::atomic<int> cleanups{0};
private:
    std::mutex mutex;
    std::condition_variable changed;
    bool started = false;
    bool cleanup_blocked = false, cleanup_waiting = false;
};

struct HeldCleanup {
    explicit HeldCleanup(MockBackend& value) : backend(value) { backend.hold_cleanup(true); }
    ~HeldCleanup() { backend.hold_cleanup(false); }
    MockBackend& backend;
};

class RunningServer {
public:
    explicit RunningServer(int limit)
        : server(backend, {{"port", 0}, {"max_clients", limit}, {"heartbeat_interval", 0.05}}),
          thread([this] { try { server.run(); } catch (...) { error = std::current_exception(); } }) {}
    ~RunningServer() {
        server.request_stop();
        thread.join();
        server.shutdown();
    }
    MockBackend backend;
    Server server;
private:
    std::exception_ptr error;
    std::thread thread;
};

class Client {
public:
    explicit Client(unsigned short port) : fd(::socket(AF_INET, SOCK_STREAM, 0)) {
        require(fd >= 0, "cannot create test client");
        sockaddr_in address{};
        address.sin_family = AF_INET;
        address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        address.sin_port = htons(port);
        if (::connect(fd, reinterpret_cast<sockaddr*>(&address), sizeof(address))) {
            auto message = std::string("cannot connect test client: ") + std::strerror(errno);
            close_socket(fd);
            throw std::runtime_error(message);
        }
        configure_socket(fd);
        socket_timeout(fd, 1.0);
    }
    ~Client() { close_socket(fd); }
    Client(const Client&) = delete;
    Client& operator=(const Client&) = delete;

    void send(Json meta, Arrays arrays = {}) {
        Frame frame;
        frame.request_id = ++request_id;
        frame.meta = std::move(meta);
        frame.arrays = std::move(arrays);
        send_frame(fd, frame);
    }
    Frame request(Json meta, Arrays arrays = {}) {
        send(std::move(meta), std::move(arrays));
        return receive();
    }
    Frame receive() {
        auto frame = recv_frame(fd);
        require(frame.request_id == request_id, "response has the wrong request ID");
        require(frame.kind == RESPONSE, "request failed: " + frame.meta.dump());
        return frame;
    }
    void expect_pending() {
        require(!readable(fd, 0.05), "takeover returned before capture cleanup completed");
    }
    Frame config(bool takeover = false) {
        return request({{"command", "config"}, {"takeover", takeover}});
    }
    void partial_frame() {
        require(::send(fd, "QK01", 4, MSG_NOSIGNAL) == 4, "cannot send partial frame");
    }
    void expect_closed() {
        const auto deadline = Clock::now() + std::chrono::seconds(1);
        while (Clock::now() < deadline) {
            if (!readable(fd, 0.02)) continue;
            char buffer[1024];
            const auto received = ::recv(fd, buffer, sizeof(buffer), MSG_DONTWAIT);
            if (received == 0 || (received < 0 && errno == ECONNRESET)) return;
            require(received >= 0 || errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR,
                    "unexpected receive error while waiting for eviction");
        }
        throw std::runtime_error("oldest client was not disconnected promptly");
    }

private:
    int fd;
    std::uint32_t request_id = 0;
};

std::string start_capture(Client& client, MockBackend& backend) {
    auto config = client.config();
    Json plan{{"schema", 1}, {"firmware_id", config.meta.at("firmware_id")},
              {"loop_dims", Json::array({1})}, {"counter_addr", 1}};
    Array pmem{"<i4", {1, 8}, std::vector<std::uint8_t>(32)};
    auto uploaded = client.request({{"command", "upload"}, {"plan", plan}}, {{"pmem", pmem}});
    client.send({{"command", "run"}, {"program_id", uploaded.meta.at("program_id")}, {"mode", "run"}});
    backend.wait_for_capture();
    return config.meta.at("session_id");
}

void wait_for_cleanup(Client& client, MockBackend& backend) {
    const auto deadline = Clock::now() + std::chrono::seconds(2);
    do {
        auto status = client.request({{"command", "status"}});
        if (!status.meta.at("busy").get<bool>()) {
            require(backend.cleanups.load() > 0, "evicted acquisition did not clean up");
            client.request({{"command", "reset_gens"}});
            return;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    } while (Clock::now() < deadline);
    throw std::runtime_error("board remained busy after acquisition cancellation");
}

void fifo() {
    RunningServer running(2);
    Client oldest(running.server.port());
    oldest.config();
    Client middle(running.server.port());
    middle.config();
    // Recent traffic must not change the admission order.
    oldest.request({{"command", "status"}});
    Client latest(running.server.port());
    // Admission itself must evict, even before the new client sends a request.
    oldest.expect_closed();
    latest.config();
    middle.request({{"command", "status"}});
    Client next(running.server.port());
    next.config();
    middle.expect_closed();
    latest.request({{"command", "status"}});
}

void single_limit() {
    RunningServer running(1);
    auto previous = std::make_unique<Client>(running.server.port());
    previous->config();
    for (int i = 0; i < 32; ++i) {
        auto latest = std::make_unique<Client>(running.server.port());
        latest->config(i % 2 == 0);
        previous->expect_closed();
        latest->request({{"command", "status"}});
        previous = std::move(latest);
    }
}

void stalled_clients() {
    RunningServer running(2);
    Client oldest(running.server.port());
    oldest.config();
    Client middle(running.server.port());
    middle.config();
    oldest.partial_frame();
    middle.partial_frame();
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    // A held overflow socket must not reserve the only path for another client.
    Client held(running.server.port());
    held.partial_frame();
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    Client latest(running.server.port());
    latest.config();
    oldest.expect_closed();
    middle.expect_closed();
    Client next(running.server.port());
    next.config(true);
    held.expect_closed();
    latest.request({{"command", "status"}});
}

void acquisition() {
    RunningServer running(1);
    Client acquiring(running.server.port());
    start_capture(acquiring, running.backend);
    Client latest(running.server.port());
    latest.config();
    acquiring.expect_closed();
    wait_for_cleanup(latest, running.backend);
    auto takeover = latest.config(true);
    require(takeover.meta.at("takeover_complete").get<bool>(), "takeover did not complete");
    latest.request({{"command", "status"}});
}

void acquisition_takeover() {
    RunningServer running(1);
    HeldCleanup held(running.backend);
    Client acquiring(running.server.port());
    start_capture(acquiring, running.backend);
    Client latest(running.server.port());
    latest.send({{"command", "config"}, {"takeover", true}});
    running.backend.wait_for_cleanup_start();
    latest.expect_pending();
    running.backend.hold_cleanup(false);
    auto config = latest.receive();
    require(config.meta.at("takeover_complete").get<bool>(), "takeover did not complete");
    acquiring.expect_closed();
    wait_for_cleanup(latest, running.backend);
}

void pending_takeover() {
    RunningServer running(1);
    HeldCleanup held(running.backend);
    Client acquiring(running.server.port());
    start_capture(acquiring, running.backend);
    Client pending(running.server.port());
    pending.send({{"command", "config"}, {"takeover", true}});
    running.backend.wait_for_cleanup_start();
    pending.expect_pending();
    Client latest(running.server.port());
    latest.send({{"command", "config"}, {"takeover", true}});
    pending.expect_closed();
    latest.expect_pending();
    running.backend.hold_cleanup(false);
    auto config = latest.receive();
    require(config.meta.at("takeover_complete").get<bool>(), "newest takeover did not complete");
    wait_for_cleanup(latest, running.backend);
}

void stop_at_limit() {
    RunningServer running(2);
    Client observer(running.server.port());
    observer.config();
    Client acquiring(running.server.port());
    const auto session = start_capture(acquiring, running.backend);
    Client controller(running.server.port());
    auto stopped = controller.request({{"command", "stop"}, {"session_id", session}});
    require(stopped.meta.at("stopped").get<bool>(), "stop request was rejected at capacity");
    observer.expect_closed();
    wait_for_cleanup(controller, running.backend);
}
} // namespace

int main(int argc, char** argv) {
    const std::pair<const char*, std::function<void()>> tests[] = {
        {"fifo", fifo}, {"single_limit", single_limit}, {"stalled_clients", stalled_clients},
        {"acquisition", acquisition}, {"acquisition_takeover", acquisition_takeover},
        {"pending_takeover", pending_takeover}, {"stop_at_limit", stop_at_limit}};
    if (argc != 2) {
        std::cerr << "usage: quick-connection-tests <case>\n";
        return 2;
    }
    for (const auto& test : tests) {
        if (test.first != std::string(argv[1])) continue;
        try {
            test.second();
            std::cout << test.first << " passed\n";
            return 0;
        } catch (const std::exception& error) {
            std::cerr << test.first << ": " << error.what() << '\n';
            return 1;
        }
    }
    std::cerr << "unknown case: " << argv[1] << '\n';
    return 2;
}
