#include "quick/server.hpp"
#include "quick/wire.hpp"
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <exception>
#include <iostream>
#include <list>
#include <mutex>
#include <netdb.h>
#include <netinet/in.h>
#include <random>
#include <set>
#include <sstream>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>

namespace quick {
namespace {
using Clock = std::chrono::steady_clock;
using Time = Clock::time_point;
double now_seconds() { return std::chrono::duration<double>(Clock::now().time_since_epoch()).count(); }
void fail(const std::string& message) { throw Error("ValueError", message); }
std::int64_t integer(const Json& value, std::int64_t low, std::int64_t high, const std::string& name) {
    if (!value.is_number_integer() || value.is_boolean()) fail("Invalid " + name);
    if (value.is_number_unsigned() && value.get<std::uint64_t>() > static_cast<std::uint64_t>(INT64_MAX)) fail("Invalid " + name);
    const auto n = value.get<std::int64_t>();
    if (n < low || n > high) fail("Invalid " + name);
    return n;
}
Json get(const Json& obj, const std::string& key, Json fallback = nullptr) {
    if (!obj.is_object()) return fallback;
    auto it = obj.find(key);
    return it == obj.end() ? fallback : *it;
}
std::string string_value(const Json& obj, const std::string& key, const std::string& fallback = "") {
    auto v = get(obj, key, fallback);
    if (!v.is_string()) fail("Invalid " + key);
    return v.get<std::string>();
}
bool boolean(const Json& obj, const std::string& key, bool fallback) {
    auto v = get(obj, key, fallback);
    if (!v.is_boolean()) fail(key + " must be a boolean");
    return v.get<bool>();
}
bool finite_number(const Json& v) { return v.is_number() && std::isfinite(v.get<double>()); }
std::int64_t total_shots(const Json& plan) {
    std::int64_t n = 1;
    for (const auto& dim : plan.at("loop_dims")) n *= dim.get<std::int64_t>();
    return n;
}
std::vector<std::uint8_t> bytes(const std::string& s) { return {s.begin(), s.end()}; }
Frame response(std::uint8_t kind, std::uint32_t id, Json meta, Arrays arrays = {}) {
    Frame frame; frame.kind = kind; frame.request_id = id; frame.meta = std::move(meta); frame.arrays = std::move(arrays); return frame;
}
std::string session_id() {
    // getrandom-backed random_device on Linux; no predictable ownership token.
    std::random_device random;
    static constexpr char hex[] = "0123456789abcdef";
    std::string id(32, '0');
    for (std::size_t i = 0; i < 16; ++i) {
        auto b = static_cast<unsigned char>(random());
        id[2*i] = hex[b >> 4]; id[2*i+1] = hex[b & 15];
    }
    return id;
}
bool transport_error(const Error& error) {
    return error.type == "OSError" || error.type == "EOFError" || error.type == "ConnectionError";
}
}

struct Server::Impl {
    // Unlike std::mutex, the board reservation may be handed from the request
    // thread to its capture worker. No C++ mutex is unlocked by a non-owner.
    struct BoardGate {
        std::mutex mutex;
        std::condition_variable available;
        bool occupied = false;
        bool try_lock() {
            std::lock_guard<std::mutex> guard(mutex);
            if (occupied) return false;
            occupied = true; return true;
        }
        void lock() {
            std::unique_lock<std::mutex> guard(mutex);
            available.wait(guard, [&] { return !occupied; }); occupied = true;
        }
        void unlock() {
            std::lock_guard<std::mutex> guard(mutex);
            occupied = false; available.notify_one();
        }
    };
    struct Connection {
        explicit Connection(int socket) : fd(socket), session(session_id()) {}
        ~Connection() { close_socket(fd); }
        void disconnect() { disconnected.store(true); ::shutdown(fd, SHUT_RDWR); }
        const int fd;
        const std::string session;
        std::atomic<bool> disconnected{false};
        bool reserved = false; // protected by state
        bool busy = false;
        bool registered = true;
    };
    struct Program { Json plan; Arrays arrays; std::size_t size = 0; };
    struct Capture {
        std::atomic<bool> cancel{false};
        std::mutex mutex;
        std::condition_variable changed;
        std::deque<Frame> output;
        bool done = false;
        std::exception_ptr error;
    };

    Backend& backend;
    Json cfg;
    std::string firmware_id;
    int listener = -1;
    unsigned short bound_port = 0;
    std::size_t queue_depth, cache_limit, client_limit;
    double heartbeat_interval;
    std::atomic<bool> stop{false};
    std::mutex state, lifecycle, shutdown_mutex;
    std::condition_variable lifecycle_changed;
    bool serving = false, shutdown_complete = false;
    BoardGate board;
    std::timed_mutex takeover_lock;
    std::list<std::shared_ptr<Connection>> connections;
    struct SessionThread { std::thread thread; std::shared_ptr<std::atomic<bool>> finished; };
    std::vector<SessionThread> sessions;
    std::list<std::pair<std::string, std::shared_ptr<Program>>> cache;
    std::size_t cache_size = 0;
    std::shared_ptr<Connection> experiment, active_connection;
    std::shared_ptr<Capture> active_capture;
    bool takeover_pending = false;
    double recover_until = 0;
    std::string loaded_id;
    bool loaded_envelopes = false;

    explicit Impl(Backend& value, const Json& settings) : backend(value) {
        queue_depth = integer(get(settings, "queue_depth", 8), 1, 64, "queue depth");
        cache_limit = integer(get(settings, "cache_bytes", 128*1024*1024), 1, 512*1024*1024, "cache size");
        client_limit = integer(get(settings, "max_clients", 8), 1, 32, "client limit");
        auto hb = get(settings, "heartbeat_interval", 1.0);
        if (!finite_number(hb) || hb.get<double>() <= 0) fail("Heartbeat interval must be positive seconds");
        heartbeat_interval = hb.get<double>();
        cfg = backend.get_cfg();
        firmware_id = sha256(canonical_json(cfg));
        auto host = string_value(settings, "host", "127.0.0.1");
        auto port = integer(get(settings, "port", 9999), 0, 65535, "port");
        addrinfo hints{}, *address = nullptr;
        hints.ai_family = AF_INET; hints.ai_socktype = SOCK_STREAM;
        int result = ::getaddrinfo(host.c_str(), std::to_string(port).c_str(), &hints, &address);
        if (result) throw Error("OSError", std::string("Cannot resolve listen address: ") + gai_strerror(result));
        listener = ::socket(AF_INET, SOCK_STREAM, 0);
        if (listener < 0) { freeaddrinfo(address); throw Error("OSError", "Cannot create listening socket"); }
        int one = 1; ::setsockopt(listener, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
        result = ::bind(listener, address->ai_addr, address->ai_addrlen);
        freeaddrinfo(address);
        if (result || ::listen(listener, static_cast<int>(client_limit))) {
            auto message = std::string("Cannot bind/listen: ") + std::strerror(errno);
            close_socket(listener); listener = -1; throw Error("OSError", message);
        }
        sockaddr_in local{}; socklen_t length = sizeof(local);
        ::getsockname(listener, reinterpret_cast<sockaddr*>(&local), &length);
        bound_port = ntohs(local.sin_port);
    }

    void run() {
        {
            std::lock_guard<std::mutex> lock(lifecycle);
            if (serving) throw Error("RuntimeError", "Server is already running");
            if (stop) return;
            serving = true;
        }
        try {
            while (!stop) {
                for (auto it = sessions.begin(); it != sessions.end();) {
                    if (it->finished->load()) { it->thread.join(); it = sessions.erase(it); }
                    else ++it;
                }
                if (!readable(listener, 0.2)) continue;
                int fd = ::accept(listener, nullptr, nullptr);
                if (fd < 0) {
                    if (stop) break;
                    if (errno == EINTR || errno == EAGAIN) continue;
                    throw Error("OSError", std::string("accept failed: ") + std::strerror(errno));
                }
                if (stop) { close_socket(fd); break; }
                configure_socket(fd);
                auto conn = std::make_shared<Connection>(fd);
                {
                    std::lock_guard<std::mutex> lock(state);
                    std::size_t normal = 0, reserved = 0;
                    for (const auto& existing : connections) existing->reserved ? ++reserved : ++normal;
                    if (normal >= client_limit) {
                        if (reserved) continue;
                        conn->reserved = true;
                    }
                    socket_timeout(fd, conn->reserved ? 2.0 : 5.0);
                    connections.push_back(conn);
                }
                try {
                    auto finished = std::make_shared<std::atomic<bool>>(false);
                    // Reserve before constructing a joinable thread so allocation
                    // failure cannot destroy a temporary joinable std::thread.
                    sessions.reserve(sessions.size()+1);
                    sessions.push_back({std::thread([this, conn, finished] { session(conn); finished->store(true); }), finished});
                }
                catch (...) {
                    std::lock_guard<std::mutex> lock(state);
                    connections.remove(conn); conn->registered = false; throw;
                }
            }
        } catch (...) {
            std::lock_guard<std::mutex> lock(lifecycle);
            serving = false; lifecycle_changed.notify_all(); throw;
        }
        std::lock_guard<std::mutex> lock(lifecycle);
        serving = false; lifecycle_changed.notify_all();
    }

    void request_stop() noexcept { stop.store(true); }
    void shutdown() {
        std::lock_guard<std::mutex> shutdown_guard(shutdown_mutex);
        if (shutdown_complete) return;
        request_stop();
        if (listener >= 0) ::shutdown(listener, SHUT_RDWR);
        {
            std::unique_lock<std::mutex> lock(lifecycle);
            lifecycle_changed.wait(lock, [this] { return !serving; });
        }
        {
            std::lock_guard<std::mutex> lock(state);
            if (active_capture) active_capture->cancel.store(true);
            for (const auto& conn : connections) conn->disconnect();
        }
        for (auto& session_thread : sessions) if (session_thread.thread.joinable()) session_thread.thread.join();
        sessions.clear();
        if (listener >= 0) { close_socket(listener); listener = -1; }
        std::lock_guard<BoardGate> lease(board);
        for (const auto& op : {"start_src", "stop_tproc", "reset_gens", "close"}) {
            try { backend.call(op, std::string(op) == "start_src" ? Json{{"src", "internal"}} : Json::object()); }
            catch (const std::exception& error) { std::cerr << "Shutdown " << op << ": " << error.what() << '\n'; }
        }
        shutdown_complete = true;
    }

    bool promote(const std::shared_ptr<Connection>& conn) {
        std::shared_ptr<Connection> retired;
        {
            std::lock_guard<std::mutex> lock(state);
            if (!conn->registered) return false;
            if (!conn->reserved) return true;
            std::size_t normal = 0;
            for (const auto& item : connections) if (!item->reserved) ++normal;
            if (normal >= client_limit) {
                auto it = std::find_if(connections.begin(), connections.end(), [](const auto& item) { return !item->reserved && !item->busy; });
                if (it == connections.end()) return false;
                retired = *it; retired->registered = false; connections.erase(it);
            }
            conn->reserved = false;
        }
        if (retired) retired->disconnect();
        socket_timeout(conn->fd, 5.0);
        return true;
    }

    void session(const std::shared_ptr<Connection>& conn) noexcept {
        try {
            while (!stop && !conn->disconnected) {
                bool reserved;
                {
                    std::lock_guard<std::mutex> lock(state);
                    if (!conn->registered) break;
                    reserved = conn->reserved;
                }
                if (!reserved && !readable(conn->fd, 0.2)) continue;
                {
                    std::lock_guard<std::mutex> lock(state);
                    if (!conn->registered) break;
                    conn->busy = true;
                }
                auto frame = recv_frame(conn->fd);
                if (frame.kind != REQUEST) throw Error("ProtocolError", "Expected a request frame");
                try {
                    bool taking_over = get(frame.meta, "command") == "config" && get(frame.meta, "takeover") == true;
                    if (reserved && get(frame.meta, "command") != "stop" && !taking_over && !promote(conn))
                        throw Error("BusyError", "Normal sessions are busy; reserved connection permits stop or takeover");
                    dispatch(conn, frame);
                } catch (const Error& error) {
                    if (transport_error(error)) throw;
                    send_frame(conn->fd, response(ERROR, frame.request_id, {{"error", error.type}, {"message", std::string(error.what()).substr(0, 4096)}}));
                } catch (const Json::exception& error) {
                    send_frame(conn->fd, response(ERROR, frame.request_id, {{"error", "ValueError"}, {"message", std::string(error.what()).substr(0, 4096)}}));
                } catch (const std::exception& error) {
                    send_frame(conn->fd, response(ERROR, frame.request_id, {{"error", "RuntimeError"}, {"message", std::string(error.what()).substr(0, 4096)}}));
                }
                {
                    std::lock_guard<std::mutex> lock(state);
                    conn->busy = false;
                    if (conn->reserved) break;
                }
            }
        } catch (const std::exception&) { /* EOF, malformed framing, or dead peer terminates only this session. */ }
        {
            std::lock_guard<std::mutex> lock(state);
            if (active_connection == conn && active_capture) active_capture->cancel.store(true);
            connections.remove(conn); conn->registered = false; conn->busy = false;
            if (experiment == conn) experiment.reset();
        }
        conn->disconnect();
    }

    void dispatch(const std::shared_ptr<Connection>& conn, const Frame& frame) {
        const auto& meta = frame.meta;
        auto command = string_value(meta, "command");
        if (!frame.arrays.empty() && command != "upload") throw Error("ProtocolError", "This command has no array payload");
        if (command == "config") {
            bool taken = boolean(meta, "takeover", false);
            if (taken) takeover(conn);
            send_frame(conn->fd, response(RESPONSE, frame.request_id, {{"cfg", cfg}, {"firmware_id", firmware_id}, {"session_id", conn->session},
                {"protocol", 1}, {"max_payload", MAX_PAYLOAD}, {"takeover_complete", taken}}));
        } else if (command == "upload") {
            auto result = upload(get(meta, "plan"), frame.arrays);
            send_frame(conn->fd, response(RESPONSE, frame.request_id, {{"program_id", result.first}, {"cached", result.second}}));
        } else if (command == "run") stream_run(conn, frame.request_id, meta);
        else if (command == "stop") {
            bool stopped;
            {
                std::lock_guard<std::mutex> lock(state);
                stopped = active_connection && get(meta, "session_id") == active_connection->session;
                if (stopped) active_capture->cancel.store(true);
            }
            send_frame(conn->fd, response(RESPONSE, frame.request_id, {{"stopped", stopped}}));
        } else if (command == "status") {
            Json result;
            {
                std::lock_guard<std::mutex> lock(state);
                double recovery = std::max(0.0, recover_until-now_seconds());
                result = {{"busy", bool(active_capture) || recovery > 0 || takeover_pending}, {"takeover_pending", takeover_pending},
                    {"recovery_seconds", recovery}, {"cached_programs", cache.size()}, {"cache_bytes", cache_size}, {"firmware_id", firmware_id}};
            }
            send_frame(conn->fd, response(RESPONSE, frame.request_id, std::move(result)));
        } else if (command == "read_mem" || command == "reset_gens") {
            claim_board();
            std::unique_lock<BoardGate> lease(board, std::adopt_lock);
            if (command == "read_mem") {
                auto memory = string_value(meta, "mem_sel", "dmem");
                if (memory != "pmem" && memory != "wmem" && memory != "dmem") fail("Invalid memory selector");
                auto capacity = cfg.at("tprocs").at(0).at(memory+"_size").get<std::int64_t>();
                auto addr = integer(get(meta, "addr", 0), 0, capacity-1, "memory address");
                auto length = integer(get(meta, "length"), 1, capacity-addr, "memory length");
                auto result = backend.call("read_mem", {{"length", length}, {"mem_sel", memory}, {"addr", addr}});
                send_frame(conn->fd, response(RESPONSE, frame.request_id, Json::object(), std::move(result.arrays)));
            } else {
                backend.call("reset_gens"); loaded_id.clear();
                send_frame(conn->fd, response(RESPONSE, frame.request_id, {{"reset", true}}));
            }
        } else throw Error("ProtocolError", "Unknown command");
    }

    void claim_board() {
        if (stop) throw Error("CancelledError", "Server is shutting down");
        if (!board.try_lock()) throw Error("BusyError", "Board is acquiring");
        try {
            if (stop) throw Error("CancelledError", "Server is shutting down");
            std::lock_guard<std::mutex> lock(state);
            if (takeover_pending) throw Error("BusyError", "A new experimental session is taking over");
            if (recover_until > now_seconds()) throw Error("BusyError", "ADC window recovering (" + std::to_string(recover_until-now_seconds()) + " seconds remaining)");
        } catch (...) { board.unlock(); throw; }
    }

    void takeover(const std::shared_ptr<Connection>& conn) {
        auto deadline = Clock::now() + std::chrono::seconds(30);
        std::unique_lock<std::timed_mutex> takeover_guard(takeover_lock, std::defer_lock);
        if (!takeover_guard.try_lock_until(deadline)) throw Error("BusyError", "Previous session takeover did not finish");
        std::unique_lock<BoardGate> lease(board, std::defer_lock);
        try {
            {
                std::lock_guard<std::mutex> lock(state);
                takeover_pending = true;
                if (experiment && experiment != conn) experiment->disconnect();
                experiment.reset();
                if (active_capture) active_capture->cancel.store(true);
                if (active_connection && active_connection != conn) active_connection->disconnect();
            }
            while (true) {
                if (stop) throw Error("CancelledError", "Server is shutting down");
                if (Clock::now() >= deadline) throw Error("BusyError", "Board cleanup or ADC recovery exceeded the takeover deadline");
                if (readable(conn->fd, 0)) {
                    if (peer_closed(conn->fd)) throw Error("EOFError", "New experimental session disconnected");
                    throw Error("ProtocolError", "Commands cannot overlap a session takeover");
                }
                bool recovered;
                {
                    std::lock_guard<std::mutex> lock(state);
                    if (active_capture) active_capture->cancel.store(true);
                    recovered = now_seconds() >= recover_until;
                }
                if (recovered && lease.try_lock()) {
                    { std::lock_guard<std::mutex> lock(state); recovered = now_seconds() >= recover_until; }
                    if (recovered) break;
                    lease.unlock();
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            loaded_id.clear();
            backend.call("start_src", {{"src", "internal"}}); backend.call("stop_tproc");
            for (std::size_t ch = 0; ch < cfg.at("readouts").size(); ++ch)
                backend.call("enable_buf", {{"ch", ch}, {"enable_avg", false}, {"enable_buf", false}});
            backend.call("reset_gens");
            while (!promote(conn)) {
                if (Clock::now() >= deadline) throw Error("BusyError", "No normal session slot became available for takeover");
                if (stop) throw Error("CancelledError", "Server is shutting down");
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            { std::lock_guard<std::mutex> lock(state); experiment = conn; takeover_pending = false; }
        } catch (...) {
            std::lock_guard<std::mutex> lock(state); takeover_pending = false; throw;
        }
    }

    std::pair<std::string, bool> upload(Json plan, const Arrays& arrays) {
        if (!plan.is_object() || get(plan, "schema") != 1) fail("Expected Mercator plan schema 1");
        if (get(plan, "firmware_id") != firmware_id) fail("Program was compiled for a different board configuration");
        validate_plan(plan, arrays);
        std::vector<std::vector<std::uint8_t>> parts{bytes(canonical_json(plan))};
        std::size_t size = 0;
        for (const auto& item : arrays) {
            parts.push_back(bytes(canonical_json(Json::array({item.first, item.second.dtype, item.second.shape}))));
            parts.push_back(item.second.bytes); size += item.second.bytes.size();
        }
        if (size > cache_limit) fail("Program exceeds cache capacity");
        auto id = sha256_parts(parts);
        std::lock_guard<std::mutex> lock(state);
        auto found = std::find_if(cache.begin(), cache.end(), [&](const auto& item) { return item.first == id; });
        if (found != cache.end()) { cache.splice(cache.end(), cache, found); return {id, true}; }
        while (!cache.empty() && (cache_size+size > cache_limit || cache.size() >= 8)) {
            cache_size -= cache.front().second->size; cache.pop_front();
        }
        auto entry = std::make_shared<Program>(); entry->plan = std::move(plan); entry->arrays = arrays; entry->size = size;
        cache.emplace_back(id, std::move(entry)); cache_size += size;
        return {id, false};
    }

    void validate_plan(Json& plan, const Arrays& arrays) {
        static const std::set<std::string> allowed{"schema", "firmware_id", "loop_dims", "counter_addr", "generators", "readouts", "pfbs", "envelopes", "weights"};
        for (auto it = plan.begin(); it != plan.end(); ++it) if (!allowed.count(it.key())) fail("Unknown plan fields");
        const auto loops = get(plan, "loop_dims");
        if (!loops.is_array() || loops.empty() || loops.size() > 16) fail("Invalid loop dimensions");
        std::int64_t total = 1;
        for (const auto& dim : loops) {
            auto n = integer(dim, 1, INT32_MAX, "loop dimension");
            if (n > INT32_MAX/total) fail("Invalid total shots");
            total *= n;
        }
        integer(get(plan, "counter_addr"), 1, 2, "counter register");
        std::set<std::string> expected;
        for (const auto& name : {"pmem", "wmem", "dmem"}) {
            auto it = arrays.find(name);
            if (it == arrays.end()) { if (std::string(name) == "pmem") fail("Program memory is required"); continue; }
            const auto& array = it->second; bool data = std::string(name) == "dmem";
            if (array.dtype != "<i4" || array.shape.size() != (data ? 1u : 2u)) fail(std::string("Invalid ")+name+" format");
            if (!data && array.shape[1] != 8) fail("Memory rows must contain eight words");
            if (!array.rows() || array.rows() > cfg.at("tprocs").at(0).at(std::string(name)+"_size").get<std::size_t>()) fail("Memory capacity exceeded");
            expected.insert(name);
        }
        for (const auto& pair : {std::pair<std::string, std::string>{"generators", "gens"}, {"readouts", "readouts"}}) {
            auto values = get(plan, pair.first, Json::array());
            if (!values.is_array() || values.size() > cfg.at(pair.second).size()) fail("Invalid channel declarations");
            std::set<std::int64_t> channels;
            for (auto& item : values) {
                if (!item.is_object()) fail("Channel declarations must be objects");
                auto ch = integer(get(item, "ch"), 0, static_cast<std::int64_t>(cfg.at(pair.second).size())-1, "channel");
                if (!channels.insert(ch).second) fail("Duplicate channel declaration");
                const auto& channel_cfg = cfg.at(pair.second).at(static_cast<std::size_t>(ch));
                if (pair.first == "generators") {
                    if (!item.contains("nqz")) item["nqz"] = 1;
                    integer(item["nqz"], 1, 2, "Nyquist zone");
                    if (item.contains("mixer_freq") && !finite_number(item["mixer_freq"])) fail("Invalid mixer frequency");
                    if (item.contains("mux_tones")) {
                        const auto& tones = item["mux_tones"];
                        if (!tones.is_array() || tones.size() > get(channel_cfg, "n_tones", 0).get<std::size_t>()) fail("Invalid mux tones");
                        for (const auto& tone : tones) if (!tone.is_object() || !tone.contains("freq_int")) fail("Invalid mux tones");
                    }
                } else {
                    integer(get(item, "length"), 1, INT32_MAX, "readout length");
                    integer(get(item, "trigs"), 0, INT32_MAX, "readout triggers");
                    if (!channel_cfg.contains("tproc_ctrl") && !channel_cfg.contains("pfb_port") && !get(item, "ro_config").is_object()) fail("Static readout configuration is required");
                }
            }
            if (plan.contains(pair.first)) plan[pair.first] = std::move(values);
        }
        std::set<std::string> pfb_paths;
        for (const auto& ro : cfg.at("readouts")) if (ro.contains("pfb_port")) pfb_paths.insert(string_value(ro, "ro_fullpath"));
        auto pfbs = get(plan, "pfbs", Json::array());
        if (!pfbs.is_array()) fail("PFB declarations must be a list");
        for (const auto& pfb : pfbs) {
            if (!pfb.is_object()) fail("PFB declarations must be objects");
            if (!get(pfb, "path").is_string() || !pfb_paths.count(pfb["path"].get<std::string>()) || !get(pfb, "regs").is_array()) fail("Invalid PFB readout");
        }
        for (const auto& name : {"envelopes", "weights"}) {
            bool envelope = std::string(name) == "envelopes";
            auto values = get(plan, name, Json::array());
            auto cfgkey = envelope ? "gens" : "readouts";
            auto capacity_key = envelope ? "maxlen" : "wgt_maxlen";
            if (!values.is_array()) fail("Envelope/weight declarations must be lists");
            for (auto& item : values) {
                if (!item.is_object()) fail("Envelope/weight declarations must be objects");
                auto ch = integer(get(item, "ch"), 0, static_cast<std::int64_t>(cfg.at(cfgkey).size())-1, "channel");
                const auto& channel_cfg = cfg.at(cfgkey).at(static_cast<std::size_t>(ch));
                auto key = string_value(item, "array"); auto found = arrays.find(key);
                if (found == arrays.end() || expected.count(key)) fail("Missing or duplicate envelope/weight array");
                const auto& array = found->second;
                if (array.dtype != "<i2" || array.shape.size() != 2 || array.shape[1] != 2) fail("Envelope/weight array must be int16 IQ");
                auto capacity = get(channel_cfg, capacity_key, 0).get<std::int64_t>();
                auto addr = integer(get(item, "addr", 0), 0, capacity, "envelope address");
                if (!item.contains("addr")) item["addr"] = 0;
                if (!array.rows() || array.rows() > static_cast<std::size_t>(capacity-addr)) fail("Envelope/weight memory capacity exceeded");
                if (envelope && array.rows() % channel_cfg.at("samps_per_clk").get<std::size_t>()) fail("Envelope length is not a whole fabric cycle");
                expected.insert(key);
            }
            if (plan.contains(name)) plan[name] = std::move(values);
        }
        if (arrays.size() != expected.size()) fail("Unexpected program arrays");
    }

    void check(const std::shared_ptr<Capture>& capture, double deadline = 0) {
        if (capture->cancel || stop) throw Error("CancelledError", "Acquisition cancelled");
        if (deadline > 0 && now_seconds() >= deadline) throw Error("TimeoutError", "Acquisition timed out");
    }
    void emit(const std::shared_ptr<Capture>& capture, Frame frame) {
        check(capture);
        std::unique_lock<std::mutex> lock(capture->mutex);
        if (!capture->changed.wait_for(lock, std::chrono::milliseconds(10), [&] { return capture->output.size() < queue_depth || capture->cancel.load() || stop.load(); }))
            throw Error("OverflowError", "TCP output queue filled; acquisition stopped without dropping data");
        check(capture);
        capture->output.push_back(std::move(frame)); capture->changed.notify_all();
    }

    void stream_run(const std::shared_ptr<Connection>& conn, std::uint32_t request_id, const Json& meta) {
        auto mode = string_value(meta, "mode");
        if (mode != "accumulated" && mode != "decimated" && mode != "run") fail("Invalid acquisition mode");
        auto rounds = integer(get(meta, "rounds", 1), 1, 1000000, "round count");
        auto source = string_value(meta, "start_src", "internal");
        if (source != "internal" && source != "external") fail("Invalid start source");
        auto timeout = get(meta, "timeout");
        if (!timeout.is_null() && (!finite_number(timeout) || timeout.get<double>() <= 0)) fail("Timeout must be positive seconds");
        boolean(meta, "load_envelopes", true);
        auto id = string_value(meta, "program_id");
        std::shared_ptr<Program> entry;
        {
            std::lock_guard<std::mutex> lock(state);
            for (const auto& item : cache) if (item.first == id) { entry = item.second; break; }
        }
        if (!entry) throw Error("CacheMissError", "Program is not cached; upload it again");
        auto total = total_shots(entry->plan);
        auto readouts = get(entry->plan, "readouts", Json::array());
        if (mode != "run" && readouts.empty()) fail("Acquisition needs triggered readouts");
        for (const auto& ro : readouts) {
            auto triggers = ro.at("trigs").get<std::int64_t>();
            if (mode != "run" && triggers <= 0) fail("Acquisition needs triggered readouts");
            const auto& c = cfg.at("readouts").at(ro.at("ch").get<std::size_t>());
            auto average_capacity = c.at("avg_maxlen").get<std::int64_t>();
            if (mode == "accumulated" && triggers > average_capacity-2) fail("One shot exceeds the accumulated buffer");
            if (mode == "decimated") {
                auto length = ro.at("length").get<std::int64_t>(), capacity = c.at("buf_maxlen").get<std::int64_t>();
                // Division avoids signed overflow on hostile dimensions/trigger lengths.
                if (length >= capacity || total > (average_capacity-2)/triggers || total > (capacity/length)/triggers)
                    fail("Decimated acquisition exceeds a capture buffer");
            }
        }
        auto capture = std::make_shared<Capture>();
        claim_board();
        { std::lock_guard<std::mutex> lock(state); active_connection = conn; active_capture = capture; }
        std::thread worker;
        try { worker = std::thread([this, meta, entry, capture] { capture_run(meta, entry, capture); }); }
        catch (...) {
            { std::lock_guard<std::mutex> lock(state); active_connection.reset(); active_capture.reset(); }
            board.unlock(); throw;
        }
        auto finish = [&] { capture->cancel.store(true); capture->changed.notify_all(); if (worker.joinable()) worker.join(); };
        try {
            double last_sent = now_seconds();
            while (true) {
                Frame output; bool have_output = false, done = false; std::exception_ptr error;
                {
                    std::unique_lock<std::mutex> lock(capture->mutex);
                    capture->changed.wait_for(lock, std::chrono::duration<double>(std::min(0.05, heartbeat_interval)), [&] { return capture->done || (!capture->error && !capture->output.empty()); });
                    done = capture->done; error = capture->error;
                    if (!error && !capture->output.empty()) {
                        output = std::move(capture->output.front()); capture->output.pop_front(); have_output = true; capture->changed.notify_all();
                    }
                }
                if (error && done) std::rethrow_exception(error);
                if (have_output) {
                    output.request_id = request_id; send_frame(conn->fd, output); last_sent = now_seconds(); continue;
                }
                if (done && !error) break;
                if (!error && readable(conn->fd, 0)) {
                    if (peer_closed(conn->fd)) throw Error("EOFError", "Acquisition client disconnected");
                    throw Error("ProtocolError", "Commands cannot overlap an acquisition stream");
                }
                if (now_seconds()-last_sent >= heartbeat_interval) {
                    send_frame(conn->fd, response(DATA, request_id, {{"kind", "heartbeat"}})); last_sent = now_seconds();
                }
            }
            send_frame(conn->fd, response(DONE, request_id, {{"event", "run"}, {"rounds", rounds}}));
            finish();
        } catch (...) { finish(); throw; }
    }

    void configure(const std::string& id, const Program& entry, bool load_envelopes) {
        backend.call("start_src", {{"src", "internal"}}); backend.call("stop_tproc", {{"lazy", true}});
        if (loaded_id == id && (loaded_envelopes || loaded_envelopes == load_envelopes)) return;
        const auto& plan = entry.plan;
        if (load_envelopes) {
            for (const auto& name : {"envelopes", "weights"}) for (const auto& item : get(plan, name, Json::array())) {
                backend.call(std::string(name) == "envelopes" ? "load_envelope" : "load_weights",
                    {{"ch", item.at("ch")}, {"addr", get(item, "addr", 0)}}, {{"data", entry.arrays.at(item.at("array").get<std::string>())}});
            }
        }
        for (const auto& gen : get(plan, "generators", Json::array())) {
            backend.call("set_nyquist", {{"ch", gen.at("ch")}, {"nqz", gen.at("nqz")}});
            if (gen.contains("mixer_freq")) backend.call("set_mixer_freq", {{"ch", gen.at("ch")}, {"f", gen.at("mixer_freq")}});
            if (gen.contains("mux_tones")) backend.call("config_mux_gen", {{"ch", gen.at("ch")}, {"tones", gen.at("mux_tones")}});
        }
        for (const auto& ro : get(plan, "readouts", Json::array())) {
            const auto& c = cfg.at("readouts").at(ro.at("ch").get<std::size_t>());
            if (!c.contains("tproc_ctrl") && !c.contains("pfb_port")) backend.call("configure_readout", {{"ch", ro.at("ch")}, {"ro_regs", ro.at("ro_config")}});
        }
        for (const auto& pfb : get(plan, "pfbs", Json::array()))
            backend.call("config_mux_readout", {{"pfbpath", pfb.at("path")}, {"cfgs", pfb.at("regs")}, {"sel", get(pfb, "sel")}});
        Arrays memories;
        for (const auto& name : {"pmem", "wmem", "dmem"}) { auto it = entry.arrays.find(name); if (it != entry.arrays.end()) memories.emplace(*it); }
        backend.call("load_bin_program", {{"load_mem", false}}, memories);
        loaded_id = id; loaded_envelopes = load_envelopes;
    }

    double recovery_delay(const Json& readouts) {
        double result = 0;
        for (const auto& ro : readouts) {
            const auto& c = cfg.at("readouts").at(ro.at("ch").get<std::size_t>());
            auto output = get(c, "f_output", 1.0), fabric = get(c, "f_fabric", output);
            double f_output = finite_number(output) && output.get<double>() > 0 ? output.get<double>() : 1.0;
            double f_fabric = finite_number(fabric) && fabric.get<double>() > 0 ? fabric.get<double>() : 1.0;
            result = std::max(result, (ro.at("length").get<double>()/f_output+4/f_fabric)*1e-6);
        }
        return result;
    }

    void capture_run(const Json& meta, const std::shared_ptr<Program>& entry, const std::shared_ptr<Capture>& capture) noexcept {
        const auto& plan = entry->plan;
        auto readouts = get(plan, "readouts", Json::array());
        bool prepared = false;
        try {
            check(capture);
            configure(meta.at("program_id").get<std::string>(), *entry, boolean(meta, "load_envelopes", true));
            auto rounds = get(meta, "rounds", 1).get<std::int64_t>(), total = total_shots(plan);
            auto mode = meta.at("mode").get<std::string>();
            for (std::int64_t round = 0; round < rounds; ++round) {
                check(capture);
                for (const auto& ro : readouts) if (mode != "run") {
                    backend.call("config_avg", {{"ch", ro.at("ch")}, {"length", ro.at("length")}, {"edge_counting", get(ro, "edge_counting", false)},
                        {"high_threshold", get(ro, "high_threshold", 0)}, {"low_threshold", get(ro, "low_threshold", 0)}});
                    if (mode == "decimated") backend.call("config_buf", {{"ch", ro.at("ch")}, {"length", ro.at("length")}});
                    backend.call("enable_buf", {{"ch", ro.at("ch")}, {"enable_avg", true}, {"enable_buf", mode == "decimated"}});
                }
                backend.call("reload_mem"); backend.call("clear_tproc_counter", {{"addr", plan.at("counter_addr")}});
                prepared = true; backend.call("prepare_round"); backend.call("start_src", {{"src", get(meta, "start_src", "internal")}});
                double deadline = get(meta, "timeout").is_null() ? 0 : now_seconds()+meta.at("timeout").get<double>();
                backend.call("start_tproc");
                if (mode == "accumulated") accumulated(plan, round, total, capture, deadline);
                else {
                    wait_counter(plan, total, capture, deadline); backend.call("start_src", {{"src", "internal"}});
                    if (mode == "decimated") {
                        Arrays raw; Json channels = Json::array();
                        for (const auto& ro : readouts) {
                            auto ch = ro.at("ch").get<std::int64_t>();
                            auto n = total*ro.at("trigs").get<std::int64_t>()*ro.at("length").get<std::int64_t>();
                            auto result = backend.call("get_decimated", {{"ch", ch}, {"address", 0}, {"length", n}});
                            Arrays data; data.emplace(std::to_string(ch), std::move(result.arrays.at("data")));
                            emit(capture, response(DATA, 0, {{"kind", "decimated"}, {"round", round}, {"offset", 0}, {"count", n}, {"channels", Json::array({ch})}}, std::move(data)));
                        }
                        for (const auto& ro : readouts) {
                            auto ch = ro.at("ch").get<std::int64_t>(); channels.push_back(ch);
                            auto result = backend.call("get_accumulated", {{"ch", ch}, {"address", 0}, {"length", total*ro.at("trigs").get<std::int64_t>()}});
                            raw.emplace(std::to_string(ch), std::move(result.arrays.at("data")));
                        }
                        emit(capture, response(DATA, 0, {{"kind", "raw"}, {"round", round}, {"offset", 0}, {"count", total}, {"channels", channels}}, std::move(raw)));
                    }
                }
                backend.call("start_src", {{"src", "internal"}}); prepared = false; backend.call("cleanup_round");
                emit(capture, response(DONE, 0, {{"event", "round"}, {"round", round}, {"shots", total}}));
            }
        } catch (...) {
            { std::lock_guard<std::mutex> lock(capture->mutex); capture->error = std::current_exception(); capture->changed.notify_all(); }
            loaded_id.clear();
            for (const auto& ro : readouts) {
                try { backend.call("enable_buf", {{"ch", ro.at("ch")}, {"enable_avg", false}, {"enable_buf", false}}); }
                catch (const std::exception& e) { std::cerr << "Buffer cleanup: " << e.what() << '\n'; }
            }
            {
                std::lock_guard<std::mutex> lock(state);
                recover_until = std::max(recover_until, now_seconds()+recovery_delay(readouts));
            }
            for (const auto& op : {"start_src", "stop_tproc", "reset_gens"}) {
                try { backend.call(op, std::string(op) == "start_src" ? Json{{"src", "internal"}} : Json::object()); }
                catch (const std::exception& e) { std::cerr << "Acquisition cleanup " << op << ": " << e.what() << '\n'; }
            }
        }
        if (prepared) {
            try { backend.call("cleanup_round"); }
            catch (const std::exception& e) { std::cerr << "Round cleanup: " << e.what() << '\n'; }
        }
        { std::lock_guard<std::mutex> lock(state); active_connection.reset(); active_capture.reset(); }
        board.unlock();
        { std::lock_guard<std::mutex> lock(capture->mutex); capture->done = true; capture->changed.notify_all(); }
    }

    std::int64_t counter(const Json& plan) {
        auto result = backend.call("get_tproc_counter", {{"addr", plan.at("counter_addr")}});
        try { return integer(get(result.meta, "value"), 0, INT64_MAX, "shot counter"); }
        catch (const Error&) { throw Error("RuntimeError", "Invalid or reset shot counter"); }
    }
    void wait_counter(const Json& plan, std::int64_t total, const std::shared_ptr<Capture>& capture, double deadline) {
        std::int64_t previous = 0;
        while (true) {
            check(capture, deadline); auto count = counter(plan);
            if (count < previous || count > total) throw Error("RuntimeError", "Invalid or reset shot counter");
            if (count == total) return;
            previous = count; std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    }
    void accumulated(const Json& plan, std::int64_t round, std::int64_t total, const std::shared_ptr<Capture>& capture, double deadline) {
        const auto& readouts = plan.at("readouts");
        std::int64_t maxshots = INT64_MAX, bytes_per_shot = 0, offset = 0;
        Json channels = Json::array();
        for (const auto& ro : readouts) {
            auto ch = ro.at("ch").get<std::size_t>(), triggers = ro.at("trigs").get<std::size_t>(); channels.push_back(ch);
            auto capacity = cfg.at("readouts").at(ch).at("avg_maxlen").get<std::int64_t>();
            maxshots = std::min(maxshots, (capacity-2)/static_cast<std::int64_t>(triggers));
            bytes_per_shot += static_cast<std::int64_t>(triggers)*8;
        }
        std::int64_t stride = std::max<std::int64_t>(1, std::min<std::int64_t>(1024, maxshots/8));
        stride = std::min(stride, std::max<std::int64_t>(1, MAX_PAYLOAD/bytes_per_shot));
        while (offset < total) {
            check(capture, deadline); auto count = counter(plan);
            if (count < offset || count > total) throw Error("RuntimeError", "Invalid or reset shot counter");
            auto available = count-offset;
            if (available > maxshots) throw Error("OverflowError", "Accumulated buffer overrun; increase the repetition delay");
            if (available < stride && count != total) { std::this_thread::sleep_for(std::chrono::microseconds(100)); continue; }
            auto n = std::min(available, stride);
            if (!n) continue;
            Arrays raw;
            for (const auto& ro : readouts) {
                auto ch = ro.at("ch").get<std::size_t>(), reads = ro.at("trigs").get<std::size_t>();
                auto capacity = cfg.at("readouts").at(ch).at("avg_maxlen").get<std::int64_t>();
                auto result = backend.call("get_accumulated", {{"ch", ch}, {"address", (offset*reads)%capacity}, {"length", n*reads}});
                auto& array = result.arrays.at("data");
                if (array.shape != std::vector<std::size_t>{static_cast<std::size_t>(n)*reads, 2}) throw Error("RuntimeError", "Hardware returned an invalid IQ block");
                raw.emplace(std::to_string(ch), std::move(array));
            }
            if (counter(plan)-offset > maxshots) throw Error("OverflowError", "Accumulated buffer overwritten during DMA transfer");
            emit(capture, response(DATA, 0, {{"kind", "accumulated"}, {"round", round}, {"offset", offset}, {"count", n}, {"channels", channels}}, std::move(raw)));
            offset += n;
        }
    }
};

Server::Server(Backend& backend, const Json& settings) : impl_(std::make_unique<Impl>(backend, settings)) {}
Server::~Server() { try { impl_->shutdown(); } catch (...) {} }
void Server::run() { impl_->run(); }
void Server::request_stop() noexcept { impl_->request_stop(); }
void Server::shutdown() { impl_->shutdown(); }
unsigned short Server::port() const { return impl_->bound_port; }
}
