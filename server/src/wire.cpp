#include "quick/wire.hpp"
#include <cerrno>
#include <cmath>
#include <cstring>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <set>
#include <sys/socket.h>
#include <unistd.h>

namespace quick {
namespace {
[[noreturn]] void socket_error() { throw Error("OSError", std::strerror(errno)); }
void receive(int fd, std::uint8_t *data, std::size_t size) {
    while (size) {
        auto n = ::recv(fd, data, size, 0);
        if (n == 0)
            throw Error("EOFError", "Peer closed the connection");
        if (n < 0) {
            if (errno == EINTR)
                continue;
            socket_error();
        }
        data += n;
        size -= static_cast<std::size_t>(n);
    }
}
void send(int fd, const std::uint8_t *data, std::size_t size) {
    while (size) {
        auto n = ::send(fd, data, size, MSG_NOSIGNAL);
        if (n <= 0) {
            if (n < 0 && errno == EINTR)
                continue;
            socket_error();
        }
        data += n;
        size -= static_cast<std::size_t>(n);
    }
}
std::uint64_t get(const std::uint8_t *p, int n) {
    std::uint64_t v = 0;
    for (int i = 0; i < n; ++i)
        v = (v << 8) | p[i];
    return v;
}
void put(std::uint8_t *p, std::uint64_t v, int n) {
    for (int i = n - 1; i >= 0; --i) {
        p[i] = static_cast<std::uint8_t>(v);
        v >>= 8;
    }
}
bool nonnegative(const Json &v) {
    return v.is_number_unsigned() || (v.is_number_integer() && v.get<std::int64_t>() >= 0);
}
std::size_t name_length(const std::string &s) {
    std::size_t n = 0;
    for (unsigned char c : s)
        if ((c & 0xc0) != 0x80)
            ++n;
    return n;
}
std::size_t extent(const Array &a) {
    bool zero = false;
    for (auto n : a.shape)
        zero |= (n == 0);
    if (zero)
        return 0;
    std::size_t size = a.itemsize();
    for (auto n : a.shape) {
        if (n > MAX_PAYLOAD / size)
            throw Error("ProtocolError", "Array extends beyond payload");
        size *= n;
    }
    return size;
}
} // namespace
void configure_socket(int fd) {
    int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    setsockopt(fd, SOL_SOCKET, SO_KEEPALIVE, &one, sizeof(one));
    int idle = 10, interval = 2, count = 3;
    setsockopt(fd, IPPROTO_TCP, TCP_KEEPIDLE, &idle, sizeof(idle));
    setsockopt(fd, IPPROTO_TCP, TCP_KEEPINTVL, &interval, sizeof(interval));
    setsockopt(fd, IPPROTO_TCP, TCP_KEEPCNT, &count, sizeof(count));
#ifdef TCP_USER_TIMEOUT
    int timeout = 10000;
    setsockopt(fd, IPPROTO_TCP, TCP_USER_TIMEOUT, &timeout, sizeof(timeout));
#endif
}
void socket_timeout(int fd, double seconds) {
    timeval t{};
    t.tv_sec = static_cast<time_t>(seconds);
    t.tv_usec = static_cast<suseconds_t>((seconds - t.tv_sec) * 1e6);
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &t, sizeof(t));
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &t, sizeof(t));
}
bool readable(int fd, double seconds) {
    pollfd p{fd, POLLIN, 0};
    int r;
    do {
        r = ::poll(&p, 1, static_cast<int>(std::ceil(seconds * 1000)));
    } while (r < 0 && errno == EINTR);
    if (r < 0)
        socket_error();
    return r > 0;
}
bool peer_closed(int fd) {
    char c;
    auto n = ::recv(fd, &c, 1, MSG_PEEK | MSG_DONTWAIT);
    if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK))
        return false;
    if (n < 0)
        socket_error();
    return n == 0;
}
void close_socket(int fd) {
    if (fd >= 0) {
        ::shutdown(fd, SHUT_RDWR);
        ::close(fd);
    }
}

Frame recv_frame(int fd) {
    std::uint8_t h[24];
    receive(fd, h, sizeof(h));
    if (std::memcmp(h, "QK01", 4) || h[4] != 1 || h[5] < 1 || h[5] > 5 || get(h + 6, 2))
        throw Error("ProtocolError", "Unsupported frame header");
    auto msize = get(h + 12, 4), psize = get(h + 16, 8);
    if (msize > MAX_METADATA || psize > MAX_PAYLOAD)
        throw Error("ProtocolError", "Frame exceeds size limit");
    std::string text(static_cast<std::size_t>(msize), '\0');
    receive(fd, reinterpret_cast<std::uint8_t *>(text.data()), text.size());
    Frame f;
    f.kind = h[5];
    f.request_id = static_cast<std::uint32_t>(get(h + 8, 4));
    f.meta = parse_json(text);
    if (!f.meta.is_object())
        throw Error("ProtocolError", "Frame metadata must be an object");
    auto descriptors = f.meta.value("_arrays", Json::array());
    f.meta.erase("_arrays");
    if (!descriptors.is_array() || descriptors.size() > MAX_ARRAYS)
        throw Error("ProtocolError", "Invalid array descriptors");
    std::size_t offset = 0;
    std::vector<std::string> order;
    for (const auto &d : descriptors) {
        if (!d.is_object() || d.size() != 5 || !d.contains("name") || !d.contains("dtype") ||
            !d.contains("shape") || !d.contains("offset") || !d.contains("nbytes"))
            throw Error("ProtocolError", "Invalid array descriptor");
        if (!d["name"].is_string() || !d["dtype"].is_string())
            throw Error("ProtocolError", "Invalid array name or dtype");
        auto name = d["name"].get<std::string>();
        if (name.empty() || name_length(name) > 128 || f.arrays.count(name))
            throw Error("ProtocolError", "Invalid or duplicate array name");
        Array a;
        a.dtype = d["dtype"].get<std::string>();
        a.itemsize();
        if (!d["shape"].is_array() || d["shape"].size() > 8)
            throw Error("ProtocolError", "Invalid array shape");
        for (const auto &n : d["shape"]) {
            if (!nonnegative(n))
                throw Error("ProtocolError", "Invalid array shape");
            a.shape.push_back(n.get<std::size_t>());
        }
        auto size = extent(a);
        if (!nonnegative(d["offset"]) || !nonnegative(d["nbytes"]) ||
            d["offset"].get<std::uint64_t>() != offset || d["nbytes"].get<std::uint64_t>() != size)
            throw Error("ProtocolError", "Invalid array extent");
        offset += size;
        if (offset > psize)
            throw Error("ProtocolError", "Array extends beyond payload");
        a.bytes.resize(size);
        order.push_back(name);
        f.arrays.emplace(name, std::move(a));
    }
    if (offset != psize)
        throw Error("ProtocolError", "Undescribed payload bytes");
    for (auto &name : order) {
        auto &bytes = f.arrays.at(name).bytes;
        receive(fd, bytes.data(), bytes.size());
    }
    return f;
}
void send_frame(int fd, const Frame &f) {
    if (f.kind < 1 || f.kind > 5 || !f.meta.is_object() || f.meta.contains("_arrays"))
        throw Error("ProtocolError", "Invalid frame metadata or kind");
    if (f.arrays.size() > MAX_ARRAYS)
        throw Error("ProtocolError", "Too many arrays");
    Json meta = f.meta;
    meta["_arrays"] = Json::array();
    std::size_t size = 0;
    for (const auto &item : f.arrays) {
        const auto &name = item.first;
        const auto &a = item.second;
        if (name.empty() || name_length(name) > 128 || a.shape.size() > 8 ||
            extent(a) != a.bytes.size())
            throw Error("ProtocolError", "Invalid outgoing array");
        meta["_arrays"].push_back({{"name", name},
                                   {"dtype", a.dtype},
                                   {"shape", a.shape},
                                   {"offset", size},
                                   {"nbytes", a.bytes.size()}});
        size += a.bytes.size();
        if (size > MAX_PAYLOAD)
            throw Error("ProtocolError", "Frame payload exceeds limit");
    }
    auto text = canonical_json(meta);
    if (text.size() > MAX_METADATA)
        throw Error("ProtocolError", "Frame metadata exceeds limit");
    std::uint8_t h[24]{};
    std::memcpy(h, "QK01", 4);
    h[4] = 1;
    h[5] = f.kind;
    put(h + 8, f.request_id, 4);
    put(h + 12, text.size(), 4);
    put(h + 16, size, 8);
    send(fd, h, sizeof(h));
    send(fd, reinterpret_cast<const std::uint8_t *>(text.data()), text.size());
    for (const auto &item : f.arrays)
        send(fd, item.second.bytes.data(), item.second.bytes.size());
}
} // namespace quick
