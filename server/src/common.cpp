#include "quick/common.hpp"
#include "ryu/ryu.h"
#include <array>
#include <charconv>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <set>

namespace quick {
std::size_t Array::itemsize() const {
    static const std::map<std::string, std::size_t> sizes = {
        {"|i1", 1}, {"|u1", 1}, {"<i2", 2}, {"<u2", 2}, {"<i4", 4},
        {"<u4", 4}, {"<i8", 8}, {"<u8", 8}, {"<f4", 4}, {"<f8", 8}};
    auto it = sizes.find(dtype);
    if (it == sizes.end())
        throw Error("ProtocolError", "Unsupported array dtype");
    return it->second;
}

namespace {
std::string python_float(double v) {
    if (!std::isfinite(v))
        throw Error("ValueError", "Nonfinite JSON number");
    if (v == 0)
        return std::signbit(v) ? "-0.0" : "0.0";
    // Ryu chooses the shortest correctly rounded decimal, as Python repr does.
    // Grisu2 can choose a different final digit and break Mercator's cfg hash.
    char digits[32];
    std::string s(digits, static_cast<std::size_t>(d2s_buffered_n(v, digits)));
    bool neg = s[0] == '-';
    if (neg)
        s.erase(0, 1);
    auto epos = s.find_first_of("eE");
    int exponent = epos == std::string::npos ? 0 : std::stoi(s.substr(epos + 1));
    if (epos != std::string::npos)
        s.resize(epos);
    auto dot = s.find('.');
    int before = dot == std::string::npos ? static_cast<int>(s.size()) : static_cast<int>(dot);
    if (dot != std::string::npos)
        s.erase(dot, 1);
    int decimal = before + exponent;
    while (s.size() > 1 && s.front() == '0') {
        s.erase(0, 1);
        --decimal;
    }
    while (s.size() > 1 && s.back() == '0')
        s.pop_back();
    int sci = decimal - 1;
    std::string out = neg ? "-" : "";
    if (sci < -4 || sci >= 16) {
        out += s.front();
        if (s.size() > 1)
            out += "." + s.substr(1);
        out += sci < 0 ? "e-" : "e+";
        std::string exp = std::to_string(std::abs(sci));
        if (exp.size() < 2)
            out += '0';
        out += exp;
    } else if (decimal <= 0) {
        out += "0." + std::string(static_cast<std::size_t>(-decimal), '0') + s;
    } else if (decimal >= static_cast<int>(s.size())) {
        out += s + std::string(static_cast<std::size_t>(decimal) - s.size(), '0') + ".0";
    } else {
        out += s.substr(0, decimal) + "." + s.substr(decimal);
    }
    return out;
}

void append_json(std::string &out, const Json &v) {
    if (v.is_number_float())
        out += python_float(v.get<double>());
    else if (v.is_array()) {
        out += '[';
        bool first = true;
        for (const auto &x : v) {
            if (!first)
                out += ',';
            first = false;
            append_json(out, x);
        }
        out += ']';
    } else if (v.is_object()) {
        out += '{';
        bool first = true;
        for (auto it = v.begin(); it != v.end(); ++it) {
            if (!first)
                out += ',';
            first = false;
            out += Json(it.key()).dump(-1, ' ', true);
            out += ':';
            append_json(out, it.value());
        }
        out += '}';
    } else
        out += v.dump(-1, ' ', true);
}

class Sha256 {
    std::array<std::uint32_t, 8> h_{{0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f,
                                     0x9b05688c, 0x1f83d9ab, 0x5be0cd19}};
    std::array<std::uint8_t, 64> buffer_{};
    std::uint64_t size_ = 0;
    std::size_t used_ = 0;
    static std::uint32_t rr(std::uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
    void block(const std::uint8_t *p) {
        static constexpr std::uint32_t k[] = {
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
            0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
            0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
            0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
            0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
            0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
            0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
            0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
            0xc67178f2};
        std::uint32_t w[64];
        for (int i = 0; i < 16; ++i)
            w[i] = (std::uint32_t(p[4 * i]) << 24) | (std::uint32_t(p[4 * i + 1]) << 16) |
                   (std::uint32_t(p[4 * i + 2]) << 8) | p[4 * i + 3];
        for (int i = 16; i < 64; ++i) {
            auto x = w[i - 15], y = w[i - 2];
            w[i] = w[i - 16] + (rr(x, 7) ^ rr(x, 18) ^ (x >> 3)) + w[i - 7] +
                   (rr(y, 17) ^ rr(y, 19) ^ (y >> 10));
        }
        auto a = h_[0], b = h_[1], c = h_[2], d = h_[3], e = h_[4], f = h_[5], g = h_[6], h = h_[7];
        for (int i = 0; i < 64; ++i) {
            auto t1 = h + (rr(e, 6) ^ rr(e, 11) ^ rr(e, 25)) + ((e & f) ^ ((~e) & g)) + k[i] + w[i];
            auto t2 = (rr(a, 2) ^ rr(a, 13) ^ rr(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
            h = g;
            g = f;
            f = e;
            e = d + t1;
            d = c;
            c = b;
            b = a;
            a = t1 + t2;
        }
        h_[0] += a;
        h_[1] += b;
        h_[2] += c;
        h_[3] += d;
        h_[4] += e;
        h_[5] += f;
        h_[6] += g;
        h_[7] += h;
    }

  public:
    void update(const std::uint8_t *data, std::size_t size) {
        size_ += size;
        while (size) {
            auto n = std::min(size, 64 - used_);
            std::memcpy(buffer_.data() + used_, data, n);
            used_ += n;
            data += n;
            size -= n;
            if (used_ == 64) {
                block(buffer_.data());
                used_ = 0;
            }
        }
    }
    std::string finish() {
        auto bits = size_ * 8;
        std::uint8_t pad[128]{};
        pad[0] = 0x80;
        update(pad, used_ < 56 ? 56 - used_ : 120 - used_);
        std::uint8_t len[8];
        for (int i = 0; i < 8; ++i)
            len[7 - i] = static_cast<std::uint8_t>(bits >> (i * 8));
        update(len, 8);
        std::string out;
        const char *hex = "0123456789abcdef";
        for (auto word : h_)
            for (int j = 7; j >= 0; --j)
                out += hex[(word >> (j * 4)) & 15];
        return out;
    }
};
} // namespace

std::string canonical_json(const Json &value) {
    std::string out;
    append_json(out, value);
    return out;
}
Json parse_json(const std::string &text) {
    std::vector<std::set<std::string>> keys;
    try {
        return Json::parse(text, [&keys](int depth, Json::parse_event_t event, Json &parsed) {
            if (depth > 128)
                throw Error("ProtocolError", "JSON nesting limit exceeded");
            if (event == Json::parse_event_t::object_start)
                keys.emplace_back();
            if (event == Json::parse_event_t::key &&
                !keys.back().insert(parsed.get<std::string>()).second)
                throw Error("ProtocolError", "Duplicate JSON key");
            if (event == Json::parse_event_t::object_end)
                keys.pop_back();
            return true;
        });
    } catch (const Error &) {
        throw;
    } catch (const std::exception &) {
        throw Error("ProtocolError", "Invalid frame metadata");
    }
}
std::string sha256(const std::string &bytes) {
    Sha256 h;
    h.update(reinterpret_cast<const std::uint8_t *>(bytes.data()), bytes.size());
    return h.finish();
}
std::string sha256(const std::vector<std::uint8_t> &bytes) {
    Sha256 h;
    h.update(bytes.data(), bytes.size());
    return h.finish();
}
std::string sha256_parts(const std::vector<std::vector<std::uint8_t>> &parts) {
    Sha256 h;
    for (auto &p : parts)
        h.update(p.data(), p.size());
    return h.finish();
}
std::string sha256_file(const std::string &path) {
    std::ifstream file(path, std::ios::binary);
    if (!file)
        throw Error("RuntimeError", "Cannot open " + path);
    Sha256 h;
    std::array<char, 65536> buf;
    while (file) {
        file.read(buf.data(), buf.size());
        h.update(reinterpret_cast<const std::uint8_t *>(buf.data()),
                 static_cast<std::size_t>(file.gcount()));
    }
    if (!file.eof())
        throw Error("RuntimeError", "Cannot read " + path);
    return h.finish();
}
} // namespace quick
