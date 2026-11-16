// Native register/DMA backend for ZCU216, PYNQ 3.0.1.
// Register layouts/workarounds derived from QICK 0.2.418, Open Quantum Hardware
// (MIT). Native PYNQ compatibility follows Xilinx PYNQ v3.0.1 (BSD-3-Clause).
#include "quick/hardware.hpp"
#include "quick/detail/clock_presets.hpp"
#include "quick/metadata.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <regex>
#include <set>
#include <sstream>
#include <thread>

#ifdef __linux__
#include <cerrno>
#include <dlfcn.h>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <unistd.h>
extern "C" {
#include "xrfdc_abi.h"
}
namespace quick {
namespace {
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
using namespace std::chrono_literals;
[[noreturn]] void fail(const std::string &message) { throw Error("RuntimeError", message); }
void require(bool valid, const std::string &message) {
    if (!valid)
        throw Error("ValueError", message);
}
std::string s(const Json &j, const std::string &key) { return j.at(key).get<std::string>(); }
std::uint64_t param(const Json &p, const std::string &key) {
    auto v = s(p, key);
    if (v == "true")
        return 1;
    if (v == "false")
        return 0;
    return std::stoull(v, nullptr, 0);
}
double realparam(const Json &p, const std::string &key) { return std::stod(s(p, key)); }
std::uint32_t regvalue(const Json &v) {
    if (v.is_boolean())
        return v.get<bool>() ? 1 : 0;
    if (v.is_number_unsigned())
        return std::uint32_t(v.get<std::uint64_t>());
    return std::uint32_t(v.get<std::int64_t>());
}
std::vector<std::uint8_t> read_file(const fs::path &p) {
    std::ifstream f(p, std::ios::binary);
    if (!f)
        fail("Cannot read " + p.string());
    return {std::istreambuf_iterator<char>(f), {}};
}
void write_file(const fs::path &p, const void *data, std::size_t size) {
    std::ofstream f(p, std::ios::binary);
    if (!f)
        fail("Cannot write " + p.string());
    f.write(static_cast<const char *>(data), std::streamsize(size));
    f.close();
    if (!f)
        fail("Write failed: " + p.string());
}
void write_text(const fs::path &p, const std::string &text) {
    write_file(p, text.data(), text.size());
}
std::string text_file(const fs::path &p) {
    auto data = read_file(p);
    return {data.begin(), data.end()};
}
fs::path package_file(const std::string &package, const std::string &relative) {
    std::vector<fs::path> bases = {"/usr/local/share/pynq-venv/lib", "/usr/local/lib", "/usr/lib",
                                   "/opt/pynq-venv/lib"};
    if (auto v = std::getenv("VIRTUAL_ENV"))
        bases.insert(bases.begin(), fs::path(v) / "lib");
    for (const auto &base : bases) {
        std::error_code ec;
        if (!fs::is_directory(base, ec))
            continue;
        for (const auto &py : fs::directory_iterator(base, ec)) {
            if (py.path().filename().string().rfind("python", 0) != 0)
                continue;
            for (const char *site : {"site-packages", "dist-packages"}) {
                auto p = py.path() / site / package / relative;
                if (fs::exists(p, ec))
                    return p;
            }
        }
    }
    fail("Cannot locate installed " + package + "/" + relative +
         "; supply its native path in the server configuration");
}
class Library {
    void *handle_ = nullptr;

  public:
    explicit Library(const std::string &path) {
        handle_ = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!handle_)
            fail("Cannot load " + path + ": " + dlerror());
    }
    ~Library() {
        if (handle_)
            dlclose(handle_);
    }
    // A faulted DMA can retain native resources until process exit.
    void retain() { handle_ = nullptr; }
    Library(const Library &) = delete;
    Library &operator=(const Library &) = delete;
    template <class T> T get(const char *name) const {
        dlerror();
        auto p = dlsym(handle_, name);
        if (auto e = dlerror())
            fail("Missing native symbol " + std::string(name) + ": " + e);
        return reinterpret_cast<T>(p);
    }
};
class Mmio {
    void *mapping_ = MAP_FAILED;
    std::size_t mapped_size_ = 0, size_ = 0;
    volatile std::uint32_t *data_ = nullptr;

  public:
    Mmio(std::uint64_t address, std::size_t size) : size_(size) {
        auto pages = std::uint64_t(sysconf(_SC_PAGESIZE)), base = address & ~(pages - 1),
             offset = address - base;
        mapped_size_ = size + offset;
        int fd = open("/dev/mem", O_RDWR | O_SYNC | O_CLOEXEC);
        if (fd < 0)
            fail("Cannot open /dev/mem: " + std::string(strerror(errno)));
        mapping_ = mmap(nullptr, mapped_size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd, off_t(base));
        int saved = errno;
        ::close(fd);
        if (mapping_ == MAP_FAILED)
            fail("Cannot map hardware registers: " + std::string(strerror(saved)));
        data_ = reinterpret_cast<volatile std::uint32_t *>(static_cast<char *>(mapping_) + offset);
    }
    ~Mmio() {
        if (mapping_ != MAP_FAILED)
            munmap(mapping_, mapped_size_);
    }
    Mmio(const Mmio &) = delete;
    Mmio &operator=(const Mmio &) = delete;
    std::uint32_t read(std::size_t reg) const {
        if (reg >= size_ / 4)
            fail("MMIO register exceeds mapped region");
        __sync_synchronize();
        auto v = data_[reg];
        __sync_synchronize();
        return v;
    }
    void write(std::size_t reg, std::uint32_t value) {
        if (reg >= size_ / 4)
            fail("MMIO register exceeds mapped region");
        __sync_synchronize();
        data_[reg] = value;
        __sync_synchronize();
    }
    std::uintptr_t virtual_address() const { return reinterpret_cast<std::uintptr_t>(data_); }
};
struct BoProperties {
    unsigned handle, flags;
    std::uint64_t size, paddr;
    unsigned domain;
};
class Xrt {
  public:
    Library library;
    void *handle = nullptr;
    bool unsafe = false;
    unsigned (*allocate)(void *, std::size_t, int, unsigned);
    void *(*map)(void *, unsigned, bool);
    int (*properties)(void *, unsigned, BoProperties *);
    int (*sync)(void *, unsigned, int, std::size_t, std::size_t);
    void (*free)(void *, unsigned);
    void (*close)(void *);
    explicit Xrt(const Json &settings)
        : library(settings.value("xrt_library", std::getenv("XILINX_XRT")
                                                    ? std::string(std::getenv("XILINX_XRT")) +
                                                          "/lib/libxrt_core.so"
                                                    : "libxrt_core.so")) {
        auto open = library.get<void *(*)(unsigned, const char *, int)>("xclOpen");
        close = library.get<decltype(close)>("xclClose");
        allocate = library.get<decltype(allocate)>("xclAllocBO");
        map = library.get<decltype(map)>("xclMapBO");
        properties = library.get<decltype(properties)>("xclGetBOProperties");
        sync = library.get<decltype(sync)>("xclSyncBO");
        free = library.get<decltype(free)>("xclFreeBO");
        handle = open(0, nullptr, 0);
        if (!handle)
            fail("Cannot open PYNQ XRT device 0");
    }
    ~Xrt() {
        if (unsafe)
            library.retain();
        else if (handle)
            close(handle);
    }
    void load(const fs::path &p) {
        auto data = read_file(p);
        require(data.size() >= 64 && std::memcmp(data.data(), "xclbin2", 7) == 0,
                "Invalid XRT xclbin metadata");
        auto lock = library.get<int (*)(void *)>("xclLockDevice"),
             unlock = library.get<int (*)(void *)>("xclUnlockDevice");
        auto load = library.get<int (*)(void *, const void *)>("xclLoadXclBin");
        if (lock(handle))
            fail("Cannot lock XRT device for metadata registration");
        int result = load(handle, data.data());
        unlock(handle);
        if (result)
            fail("XRT metadata registration failed: " + std::to_string(result));
    }
};
class Buffer {
    Xrt &xrt_;
    unsigned bo_ = ~0u;

  public:
    std::size_t size = 0;
    std::uint64_t physical = 0;
    void *data = MAP_FAILED;
    Buffer(Xrt &xrt, std::size_t bytes, unsigned bank) : xrt_(xrt), size(bytes) {
        bo_ = xrt_.allocate(xrt_.handle, bytes, 0, bank);
        if (bo_ >= 0x80000000u)
            fail("XRT DMA buffer allocation failed; check loaded memory topology");
        data = xrt_.map(xrt_.handle, bo_, true);
        BoProperties props{};
        if (data == MAP_FAILED || !data || xrt_.properties(xrt_.handle, bo_, &props)) {
            if (data && data != MAP_FAILED)
                munmap(data, bytes);
            xrt_.free(xrt_.handle, bo_);
            bo_ = ~0u;
            fail("XRT DMA buffer mapping failed");
        }
        physical = props.paddr;
        std::memset(data, 0, bytes);
    }
    ~Buffer() {
        if (bo_ != ~0u && !xrt_.unsafe) {
            if (data && data != MAP_FAILED)
                munmap(data, size);
            xrt_.free(xrt_.handle, bo_);
        }
    }
    void sync(bool from_device, std::size_t bytes) {
        if (bytes > size)
            fail("DMA transfer exceeds allocated buffer");
        if (xrt_.sync(xrt_.handle, bo_, from_device ? 1 : 0, bytes, 0))
            fail("XRT DMA cache synchronization failed");
    }
};
class Rfdc {
    Library library_;
    XRFdc instance_{};
    XRFdc_Config config_{};

  public:
    Rfdc(const Json &settings, const Json &p, Mmio &mmio)
        : library_(settings.contains("rfdc_library")
                       ? s(settings, "rfdc_library")
                       : package_file("xrfdc", "libxrfdc.so").string()) {
        static_assert(sizeof(unsigned long) == 8, "PYNQ 3.0.1 ZCU216 RFDC ABI requires LP64");
        config_.BaseAddr = mmio.virtual_address();
        config_.ADCType = param(p, "C_High_Speed_ADC");
        config_.ADCSysRefSource = param(p, "C_Sysref_Source");
        config_.DACSysRefSource = config_.ADCSysRefSource;
        config_.IPType = param(p, "C_IP_Type");
        config_.SiRevision = param(p, "C_Silicon_Revision");
        for (int t = 0; t < 4; ++t) {
            auto ts = std::to_string(t);
            populate_tile(config_.DACTile_Config[t], p, "C_DAC" + ts + "_");
            populate_tile(config_.ADCTile_Config[t], p, "C_ADC" + ts + "_");
            for (int b = 0; b < 4; ++b) {
                auto key = ts + std::to_string(b);
                auto &da = config_.DACTile_Config[t].DACBlock_Analog_Config[b];
                auto &dd = config_.DACTile_Config[t].DACBlock_Digital_Config[b];
                auto &aa = config_.ADCTile_Config[t].ADCBlock_Analog_Config[b];
                auto &ad = config_.ADCTile_Config[t].ADCBlock_Digital_Config[b];
                da.BlockAvailable = param(p, "C_DAC_Slice" + key + "_Enable");
                da.InvSyncEnable = param(p, "C_DAC_Invsinc_Ctrl" + key);
                da.MixMode = param(p, "C_DAC_Mixer_Mode" + key);
                da.DecoderMode = param(p, "C_DAC_Decoder_Mode" + key);
                dd.MixerInputDataType = param(p, "C_DAC_Data_Type" + key);
                dd.DataWidth = param(p, "C_DAC_Data_Width" + key);
                dd.InterpolationMode = param(p, "C_DAC_Interpolation_Mode" + key);
                dd.MixerType = param(p, "C_DAC_Mixer_Type" + key);
                aa.BlockAvailable = param(p, "C_ADC_Slice" + key + "_Enable");
                aa.MixMode = param(p, "C_ADC_Mixer_Mode" + key);
                ad.MixerInputDataType = param(p, "C_ADC_Data_Type" + key);
                ad.DataWidth = param(p, "C_ADC_Data_Width" + key);
                ad.DecimationMode = param(p, "C_ADC_Decimation_Mode" + key);
                ad.MixerType = param(p, "C_ADC_Mixer_Type" + key);
            }
        }
        call<decltype(&XRFdc_CfgInitialize)>("XRFdc_CfgInitialize", &config_);
    }
    template <class T> void populate_tile(T &t, const Json &p, const std::string &prefix) {
        t.Enable = param(p, prefix + "Enable");
        t.PLLEnable = param(p, prefix + "PLL_Enable");
        t.SamplingRate = realparam(p, prefix + "Sampling_Rate");
        t.RefClkFreq = realparam(p, prefix + "Refclk_Freq");
        t.FabClkFreq = realparam(p, prefix + "Fabric_Freq");
        t.FeedbackDiv = param(p, prefix + "FBDIV");
        t.OutputDiv = param(p, prefix + "OutDiv");
        t.RefClkDiv = param(p, prefix + "Refclk_Div");
        t.MultibandConfig = param(p, prefix + "Band");
        t.MaxSampleRate = realparam(p, prefix + "Fs_Max");
        t.NumSlices = param(p, prefix + "Slices");
    }
    template <class Fn, class... Args> void call(const char *name, Args... args) {
        auto rc = library_.get<Fn>(name)(&instance_, args...);
        if (rc)
            fail(std::string(name) + " failed (" + std::to_string(rc) + ")");
    }
    bool locked(const Json &p) {
        for (unsigned kind = 0; kind < 2; ++kind)
            for (unsigned t = 0; t < 4; ++t)
                if (p.value("C_" + std::string(kind ? "DAC" : "ADC") + std::to_string(t) +
                                "_Enable",
                            "") == "1") {
                    u32 status = 0;
                    call<decltype(&XRFdc_GetPLLLockStatus)>("XRFdc_GetPLLLockStatus", kind, t,
                                                            &status);
                    if (status != 2)
                        return false;
                }
        return true;
    }
    Json actual(const Json &p) {
        Json data = {{"dac", Json::object()}, {"adc", Json::object()}};
        for (unsigned kind = 0; kind < 2; ++kind)
            for (unsigned t = 0; t < 4; ++t) {
                auto upper = std::string(kind ? "DAC" : "ADC"),
                     lower = std::string(kind ? "dac" : "adc"), ts = std::to_string(t);
                if (p.value("C_" + upper + ts + "_Enable", "") != "1")
                    continue;
                XRFdc_PLL_Settings pll{};
                call<decltype(&XRFdc_GetPLLConfig)>("XRFdc_GetPLLConfig", kind, t, &pll);
                Json values = {{"ref", pll.RefClkFreq},
                               {"mult", pll.FeedbackDivider},
                               {"div", std::uint64_t(pll.RefClkDivider) * pll.OutputDivider},
                               {"factors", Json::object()}};
                for (unsigned b = 0; b < 4; ++b) {
                    if (p.value("C_" + upper + "_Slice" + ts + std::to_string(b) + "_Enable", "") !=
                        "true")
                        continue;
                    u32 factor = 0;
                    if (kind)
                        call<decltype(&XRFdc_GetInterpolationFactor)>(
                            "XRFdc_GetInterpolationFactor", t, b, &factor);
                    else
                        call<decltype(&XRFdc_GetDecimationFactor)>("XRFdc_GetDecimationFactor", t,
                                                                   b, &factor);
                    values["factors"][std::to_string(b)] = factor;
                }
                data[lower][ts] = values;
            }
        return data;
    }
};
void configure_clocks(double ref, const Json &settings) {
    struct Device {
        fs::path path;
        std::string chip;
        unsigned width;
    };
    std::vector<Device> lmk, lmx;
    for (const auto &entry : fs::directory_iterator("/sys/bus/spi/devices")) {
        auto dev = entry.path(), compat = dev / "of_node/compatible";
        if (!fs::exists(compat))
            continue;
        auto chip = text_file(compat);
        if (chip.rfind("ti,", 0) == 0)
            chip.erase(0, 3);
        if (auto end = chip.find('\0'); end != std::string::npos)
            chip.resize(end);
        if (chip.rfind("lmk", 0) != 0 && chip.rfind("lmx", 0) != 0)
            continue;
        if (fs::exists(dev / "driver"))
            write_text(dev / "driver/unbind", dev.filename().string());
        write_text(dev / "driver_override", "spidev");
        write_text("/sys/bus/spi/drivers/spidev/bind", dev.filename().string());
        fs::path spidev;
        for (const auto &p : fs::directory_iterator(dev / "spidev")) {
            spidev = fs::path("/dev") / p.path().filename();
            break;
        }
        if (spidev.empty())
            fail("SPI clock device did not bind");
        unsigned width = 3;
        if (chip.rfind("lmk", 0) == 0) {
            auto d = read_file(dev / "of_node/num_bytes");
            require(d.size() == 4, "Invalid clock SPI width");
            width = (unsigned(d[0]) << 24) | (unsigned(d[1]) << 16) | (unsigned(d[2]) << 8) | d[3];
            require(width == 3 || width == 4, "Unsupported clock SPI width");
            lmk.push_back({spidev, chip, width});
        } else
            lmx.push_back({spidev, chip, 3});
    }
    if (lmk.empty() || lmx.empty())
        fail("LMK/LMX devices were not found in the board device tree");
    auto dir = settings.contains("clock_config_dir") ? fs::path(s(settings, "clock_config_dir"))
                                                     : package_file("xrfclk", "");
    auto program = [&](const Device &device, double frequency, bool synth) {
        std::vector<std::uint32_t> values;
        for (const auto &file : fs::directory_iterator(dir)) {
            auto name = file.path().filename().string();
            if (!detail::clock_preset_matches(name, device.chip, frequency))
                continue;
            std::istringstream input(text_file(file.path()));
            std::string line;
            while (std::getline(input, line)) {
                std::smatch hex;
                if (std::regex_search(line, hex, std::regex("0[xX][0-9A-Fa-f]+")))
                    values.push_back(std::stoul(hex.str(), nullptr, 16));
            }
            break;
        }
        if (values.empty())
            fail("No installed clock preset for " + device.chip + " at " +
                 std::to_string(frequency) + " MHz");
        int fd = open(device.path.c_str(), O_RDWR | O_CLOEXEC);
        if (fd < 0)
            fail("Cannot open clock SPI device");
        auto send = [&](std::uint32_t v) {
            unsigned char bytes[4] = {std::uint8_t(v >> 24), std::uint8_t(v >> 16),
                                      std::uint8_t(v >> 8), std::uint8_t(v)};
            if (::write(fd, bytes + 4 - device.width, device.width) != device.width)
                fail("Clock SPI write failed");
        };
        try {
            if (synth) {
                require(values.size() >= 113, "Incomplete LMX clock preset");
                send(0x020000);
                send(0);
            }
            for (auto v : values)
                send(v);
            if (synth)
                send(values[112]);
        } catch (...) {
            ::close(fd);
            throw;
        }
        ::close(fd);
    };
    for (const auto &d : lmk)
        program(d, ref, false);
    for (const auto &d : lmx)
        program(d, ref * 2, true);
}
void update_field(std::uint64_t addr, unsigned shift, unsigned width, std::uint32_t value) {
    Mmio block(addr, 4);
    auto mask = ((1u << width) - 1) << shift;
    block.write(0, (block.read(0) & ~mask) | ((value << shift) & mask));
}
void configure_pl_clocks(const Firmware &fw) {
    for (const auto &m : fw.modules) {
        if (m["type"] != "zynq_ultra_ps_e")
            continue;
        const auto &p = m["params"];
        Mmio crl(0xFF5E0000, 0x100);
        for (unsigned i = 0; i < 4; ++i) {
            auto prefix = "PSU__CRL_APB__PL" + std::to_string(i) + "_REF_CTRL__";
            auto enable = "PSU__FPGA_PL" + std::to_string(i) + "_ENABLE";
            if (!p.contains(enable))
                continue;
            unsigned div0 = 0, div1 = 0;
            if (param(p, enable)) {
                div0 = unsigned(param(p, prefix + "DIVISOR0"));
                div1 = unsigned(param(p, prefix + "DIVISOR1"));
            } else {
                // Overlay.download sets otherwise unused outputs to 100 MHz.
                auto pll = crl.read(0x20 / 4);
                require(((pll >> 20) & 7) == 0, "Invalid PS IOPLL reference source");
                double source = 33.333 * ((pll >> 8) & 127) / ((pll & (1u << 16)) ? 2 : 1);
                double best = std::numeric_limits<double>::infinity();
                for (unsigned a = 1; a < 64; ++a)
                    for (unsigned b = 1; b < 64; ++b) {
                        auto distance = std::abs(double(a * b) - source / 100.0);
                        if (distance <= best) {
                            best = distance;
                            div0 = a;
                            div1 = b;
                        }
                    }
            }
            require(div0 > 0 && div0 < 64 && div1 > 0 && div1 < 64,
                    "PS PL clock divider is out of range");
            auto reg = (0xc0 + i * 4) / 4;
            crl.write(reg, (crl.read(reg) & ~(0x01000000u | 0x003f0000u | 0x00003f00u | 7u)) |
                               (1u << 24) | (div1 << 16) | (div0 << 8));
        }
    }
}
void configure_ps(const Firmware &fw) {
    for (const auto &m : fw.modules) {
        if (m["type"] != "zynq_ultra_ps_e")
            continue;
        const auto &p = m["params"];
        for (unsigned i = 0; i < 3; ++i) {
            auto key = "C_MAXIGP" + std::to_string(i) + "_DATA_WIDTH";
            if (!p.contains(key))
                continue;
            auto w = param(p, key);
            require(w == 32 || w == 64 || w == 128, "Unsupported PS AXI width");
            update_field(i == 2 ? 0xFF419000 : 0xFD615000, i == 1 ? 10 : 8, 2,
                         w == 32   ? 0
                         : w == 64 ? 1
                                   : 2);
        }
        for (unsigned i = 0; i < 7; ++i) {
            auto key = "C_SAXIGP" + std::to_string(i) + "_DATA_WIDTH";
            if (!p.contains(key))
                continue;
            auto w = param(p, key);
            require(w == 32 || w == 64 || w == 128, "Unsupported PS AXI width");
            std::uint64_t base = i == 6 ? 0xFF9B0000 : 0xFD360000 + i * 0x10000;
            for (unsigned off : {0u, 0x14u})
                update_field(base + off, 0, 2, w == 32 ? 2 : w == 64 ? 1 : 0);
        }
    }
}
void download_bitstream(const Firmware &fw, Xrt &xrt, const Json &settings) {
    auto input = read_file(fw.bitfile);
    std::size_t pos = 0;
    auto big = [&](unsigned n) {
        if (pos + n > input.size())
            fail("Truncated bitstream header");
        std::uint32_t v = 0;
        while (n--)
            v = (v << 8) | input[pos++];
        return v;
    };
    auto first = big(2);
    pos += first;
    big(2);
    std::vector<std::uint8_t> bin;
    for (;;) {
        auto field = big(1);
        if (field == 0x65) {
            auto length = big(4);
            require(length % 4 == 0 && pos + length == input.size(),
                    "Invalid bitstream payload length");
            bin.assign(input.begin() + pos, input.end());
            break;
        }
        require(field >= 0x61 && field <= 0x64, "Unknown bitstream header field");
        auto length = big(2);
        pos += length;
    }
    for (std::size_t i = 0; i < bin.size(); i += 4) {
        std::swap(bin[i], bin[i + 3]);
        std::swap(bin[i + 1], bin[i + 2]);
    }
    // Use a unique, fixed basename rather than interpreting paths in sysfs.
    auto filename = "quick-native-" + fw.fingerprint.substr(0, 16) + ".bin";
    write_file(fs::path("/lib/firmware") / filename, bin.data(), bin.size());
    configure_pl_clocks(fw);
    write_text("/sys/class/fpga_manager/fpga0/flags", "0");
    write_text("/sys/class/fpga_manager/fpga0/firmware", filename);
    configure_ps(fw);
    auto xclbin = settings.contains("xclbin") ? fs::path(s(settings, "xclbin"))
                                              : package_file("pynq", "pl_server/default.xclbin");
    xrt.load(xclbin);
}
class BoardHardware final : public Backend {
    Firmware firmware_;
    Json settings_, cfg_, routes_;
    double timeout_;
    bool failed_ = false, reset_ = false, closed_ = false;
    int lock_ = -1;
    std::unique_ptr<Xrt> xrt_;
    std::map<std::string, std::unique_ptr<Mmio>> mmios_;
    std::unique_ptr<Rfdc> rf_;
    std::unique_ptr<Buffer> mem_write_, mem_read_, env_write_, avg_read_, buf_read_;
    Arrays program_;
    std::map<std::string, unsigned> nyquist_;
    std::map<std::string, double> mixers_;
    std::set<std::pair<std::string, bool>> dma_used_;
    Mmio &block(const std::string &name) {
        auto &ptr = mmios_[name];
        if (!ptr) {
            const auto &p = firmware_.modules.at(name)["params"];
            auto base = param(p, "C_BASEADDR"), last = param(p, "C_HIGHADDR");
            require(last >= base, "Invalid MMIO range");
            ptr = std::make_unique<Mmio>(base, last - base + 1);
        }
        return *ptr;
    }
    Mmio &tproc() { return block(firmware_.tproc); }
    const Json &gen(const Json &a) const {
        auto ch = a.at("ch").get<int>();
        require(ch >= 0 && std::size_t(ch) < cfg_["gens"].size(), "Invalid generator channel");
        return cfg_["gens"][ch];
    }
    const Json &readout(const Json &a) const {
        auto ch = a.at("ch").get<int>();
        require(ch >= 0 && std::size_t(ch) < cfg_["readouts"].size(), "Invalid readout channel");
        return cfg_["readouts"][ch];
    }
    void select(const Json &route, bool backward) {
        if (route[1].is_null())
            return;
        auto name = route[1].get<std::string>();
        const auto &p = firmware_.modules.at(name)["params"];
        auto master = backward ? route[2].get<unsigned>() : 0,
             slave = backward ? 0 : route[2].get<unsigned>();
        require(master < param(p, "NUM_MI") && slave < param(p, "NUM_SI"),
                "Invalid AXIS switch route");
        auto &b = block(name);
        b.write(0, 0);
        for (unsigned i = 0; i < param(p, "NUM_MI"); ++i)
            b.write(16 + i, 0x80000000);
        b.write(16 + master, slave);
        b.write(0, 2);
    }
    void reset_dma() {
        try {
            std::set<std::string> names = {s(routes_, "tproc")};
            for (const auto &r : routes_["gens"])
                names.insert(r[0].get<std::string>());
            for (const auto &r : routes_["readouts"])
                for (const auto &v : r)
                    names.insert(v[0].get<std::string>());
            for (const auto &n : names) {
                const auto &p = firmware_.modules.at(n)["params"];
                block(n).write(param(p, "C_INCLUDE_MM2S") ? 0 : 12, 4);
            }
            auto deadline = Clock::now() + 500ms;
            while (Clock::now() < deadline) {
                bool busy = false;
                for (const auto &n : names) {
                    const auto &p = firmware_.modules.at(n)["params"];
                    busy |= bool(block(n).read(param(p, "C_INCLUDE_MM2S") ? 0 : 12) & 4);
                }
                if (!busy) {
                    reset_ = true;
                    xrt_->unsafe = false;
                    return;
                }
                std::this_thread::sleep_for(1ms);
            }
        } catch (...) {
        }
    }
    std::size_t dma(const std::string &name, bool receive, Buffer &buffer, std::size_t bytes,
                    const std::function<void()> &start = {}) {
        if (failed_)
            fail("DMA is faulted; restart the server");
        try {
            const auto &p = firmware_.modules.at(name)["params"];
            require(param(p, "C_INCLUDE_SG") == 0,
                    "Scatter-gather DMA is not supported by this firmware backend");
            auto include = receive ? "C_INCLUDE_S2MM" : "C_INCLUDE_MM2S";
            require(param(p, include) != 0, "DMA direction is unavailable");
            auto lengthwidth = p.contains("C_SG_LENGTH_WIDTH") ? param(p, "C_SG_LENGTH_WIDTH") : 26;
            require(lengthwidth > 0 && lengthwidth <= 32, "Invalid DMA length width");
            require(bytes > 0 && bytes <= buffer.size && bytes < (std::uint64_t(1) << lengthwidth),
                    "Invalid DMA transfer length");
            auto &b = block(name);
            unsigned off = receive ? 12 : 0;
            auto status = b.read(off + 1);
            if (status & 0x70)
                fail("DMA reported error status " + std::to_string(status));
            if (status & 1) {
                b.write(off, 1);
                auto deadline = Clock::now() + std::chrono::duration<double>(timeout_);
                while (b.read(off + 1) & 1) {
                    if (Clock::now() >= deadline)
                        throw Error("TimeoutError", "DMA channel did not start");
                    std::this_thread::sleep_for(50us);
                }
            }
            if (dma_used_.count({name, receive}) && !(b.read(off + 1) & 2))
                fail("DMA channel is not idle");
            auto width =
                param(p, receive ? "C_M_AXI_S2MM_DATA_WIDTH" : "C_M_AXI_MM2S_DATA_WIDTH") / 8;
            auto dre = param(p, receive ? "C_INCLUDE_S2MM_DRE" : "C_INCLUDE_MM2S_DRE");
            require(width > 0 && (dre || buffer.physical % width == 0),
                    "DMA buffer address is not aligned");
            auto address_width =
                param(p, receive ? "C_M_AXI_S2MM_ADDR_WIDTH" : "C_M_AXI_MM2S_ADDR_WIDTH");
            require(address_width > 0 && address_width <= 64, "Invalid DMA address width");
            require(address_width == 64 ||
                        (buffer.physical < (std::uint64_t(1) << address_width) &&
                         bytes - 1 < ((std::uint64_t(1) << address_width) - buffer.physical)),
                    "DMA buffer exceeds address width");
            if (!receive)
                buffer.sync(false, bytes);
            b.write(off + 6, std::uint32_t(buffer.physical));
            b.write(off + 7, std::uint32_t(buffer.physical >> 32));
            b.write(off + 10, std::uint32_t(bytes));
            dma_used_.insert({name, receive});
            if (start)
                start();
            auto deadline = Clock::now() + std::chrono::duration<double>(timeout_);
            for (;;) {
                status = b.read(off + 1);
                if (status & 0x70)
                    fail("DMA reported error status " + std::to_string(status));
                if (status & 2)
                    break;
                if (Clock::now() >= deadline)
                    throw Error("TimeoutError",
                                "FPGA DMA did not complete; restart the server before reusing it");
                std::this_thread::sleep_for(50us);
            }
            if (receive)
                buffer.sync(true, bytes);
            return b.read(off + 10);
        } catch (...) {
            failed_ = true;
            xrt_->unsafe = true;
            reset_dma();
            throw;
        }
    }
    void memory(const std::string &kind, std::size_t length, std::int64_t addr) {
        require(kind == "pmem" || kind == "wmem" || kind == "dmem", "Invalid tProc memory");
        auto max = cfg_["tprocs"][0].at(kind + "_size").get<std::size_t>();
        require(length > 0 && addr >= 0 && std::size_t(addr) <= max &&
                    length <= max - std::size_t(addr),
                "tProc memory transfer exceeds capacity");
        tproc().write(2, std::uint32_t(addr));
        tproc().write(3, std::uint32_t(length));
    }
    void check_memory(const Array &a, const std::string &kind) {
        require(a.dtype == "<i4" || a.dtype == "=i4" || a.dtype == "i4",
                "tProc memory must be int32");
        require(!a.shape.empty() && ((kind == "dmem" && a.shape.size() == 1) ||
                                     (kind != "dmem" && a.shape.size() == 2 && a.shape[1] == 8)),
                "Invalid tProc memory array shape");
        auto n = a.rows();
        std::size_t width = kind == "dmem" ? 4 : 32;
        require(n > 0 && n <= cfg_["tprocs"][0].at(kind + "_size").get<std::size_t>() &&
                    a.bytes.size() == n * width,
                "Invalid " + kind + " length");
    }
    void load_memory(const Array &a, const std::string &kind, std::int64_t addr) {
        check_memory(a, kind);
        memory(kind, a.rows(), addr);
        std::memset(mem_write_->data, 0, a.rows() * 32);
        if (kind == "dmem")
            for (std::size_t i = 0; i < a.rows(); ++i)
                std::memcpy(static_cast<char *>(mem_write_->data) + i * 32, a.bytes.data() + i * 4,
                            4);
        else
            std::memcpy(mem_write_->data, a.bytes.data(), a.bytes.size());
        auto config = tproc().read(1) & ~63u;
        tproc().write(1, config | (kind == "pmem" ? 7 : kind == "dmem" ? 11 : 15));
        try {
            dma(s(routes_, "tproc"), false, *mem_write_, a.rows() * 32);
        } catch (...) {
            tproc().write(1, config);
            throw;
        }
        tproc().write(1, config);
    }
    void reload() {
        if (program_.empty())
            fail("No program loaded");
        for (const char *kind : {"wmem", "dmem"})
            if (program_.count(kind))
                load_memory(program_.at(kind), kind, 0);
    }
    void load_program(const Arrays &arrays, bool load) {
        require(arrays.count("pmem"), "Program memory is required");
        for (const char *kind : {"pmem", "wmem", "dmem"})
            if (arrays.count(kind))
                check_memory(arrays.at(kind), kind);
        load_memory(arrays.at("pmem"), "pmem", 0);
        program_.clear();
        for (const char *kind : {"pmem", "wmem", "dmem"})
            if (arrays.count(kind))
                program_[kind] = arrays.at(kind);
        if (load)
            reload();
    }
    Array capture(const Json &r, std::size_t address, std::size_t length, const std::string &kind) {
        auto maximum = r.at(kind + "_maxlen").get<std::size_t>();
        require(length > 0 && length <= maximum && address < maximum,
                "Capture transfer exceeds buffer capacity");
        auto padded = length + length % 2;
        if (padded >= maximum) {
            auto first = maximum - 2;
            require(first > 0, "Capture buffer is too small");
            auto a = capture(r, address, first, kind),
                 b = capture(r, (address + first) % maximum, length - first, kind);
            a.shape[0] += b.shape[0];
            a.bytes.insert(a.bytes.end(), b.bytes.begin(), b.bytes.end());
            return a;
        }
        auto name = s(r, "avgbuf_fullpath");
        auto route = routes_["readouts"][name][kind];
        select(route, false);
        auto &blockref = block(name);
        unsigned reg = kind == "avg" ? 3 : 9;
        auto &buffer = kind == "avg" ? *avg_read_ : *buf_read_;
        std::size_t width = kind == "avg" ? 8 : 4;
        blockref.write(reg + 1, std::uint32_t(address));
        blockref.write(reg + 2, std::uint32_t(padded));
        std::size_t transferred = 0;
        try {
            transferred = dma(route[0].get<std::string>(), true, buffer, padded * width,
                              [&]() { blockref.write(reg, 1); });
        } catch (...) {
            blockref.write(reg, 0);
            throw;
        }
        blockref.write(reg, 0);
        if (transferred != padded * width)
            fail("Incomplete readout DMA transfer");
        Array out;
        out.dtype = kind == "avg" ? "<i4" : "<i2";
        out.shape = {length, 2};
        out.bytes.resize(length * width);
        std::memcpy(out.bytes.data(), buffer.data, out.bytes.size());
        return out;
    }
    void stop() { tproc().write(0, 8); }
    void enable(const Json &r, bool avg, bool buf) {
        auto &b = block(s(r, "avgbuf_fullpath"));
        b.write(0, avg);
        b.write(6, buf);
    }
    void reset_generators() {
        Array pmem, wmem;
        pmem.dtype = wmem.dtype = "<i4";
        auto n = cfg_["gens"].size();
        pmem.shape = {n + 2, 8};
        wmem.shape = {n, 8};
        pmem.bytes.resize((n + 2) * 32);
        wmem.bytes.resize(n * 32);
        auto put = [](Array &a, std::size_t row, std::size_t col, std::uint32_t value) {
            std::memcpy(a.bytes.data() + row * 32 + col * 4, &value, 4);
        };
        for (std::size_t i = 0; i < n; ++i) {
            const auto &g = cfg_["gens"][i];
            unsigned conf = (g.contains("n_tones") ? 0 : 9) | (g.value("tmux_ch", 0) << 8);
            put(wmem, i, 4, 3);
            put(wmem, i, 5, conf);
            std::uint64_t address = (i << 6) | g["tproc_ch"].get<unsigned>();
            std::uint64_t lo =
                (std::uint64_t(1) << 63) | (std::uint64_t(96) << 56) | (address << 39);
            put(pmem, i + 1, 0, std::uint32_t(lo));
            put(pmem, i + 1, 1, std::uint32_t(lo >> 32));
            put(pmem, i + 1, 2, 0xdc);
        }
        auto jump = std::uint64_t(n + 1) << 45;
        put(pmem, n + 1, 0, std::uint32_t(jump));
        put(pmem, n + 1, 1, std::uint32_t(jump >> 32));
        put(pmem, n + 1, 2, 0x3c);
        stop();
        tproc().write(1, tproc().read(1) & ~(1u << 10));
        tproc().write(0, 64);
        load_program({{"pmem", pmem}, {"wmem", wmem}}, true);
        tproc().write(0, 4);
        std::this_thread::sleep_for(1ms);
        stop();
    }

  public:
    explicit BoardHardware(const Json &settings)
        : firmware_(s(settings, "bitfile")), settings_(settings),
          timeout_(settings.value("dma_timeout", 5.0)) {
        if (auto board = std::getenv("BOARD"))
            require(std::string(board) == "ZCU216", "This backend supports ZCU216 only");
        require(std::isfinite(timeout_) && timeout_ > 0, "dma_timeout must be positive");
        lock_ = open("/run/quick-native-hardware.lock", O_RDWR | O_CREAT | O_CLOEXEC, 0600);
        if (lock_ < 0 || flock(lock_, LOCK_EX | LOCK_NB)) {
            if (lock_ >= 0)
                ::close(lock_);
            lock_ = -1;
            fail("Another native server owns the hardware, or /run is not writable");
        }
        try {
            xrt_ = std::make_unique<Xrt>(settings);
            bool download = settings.value("download", true);
            auto nominal = firmware_.configuration().first;
            if (download)
                download_bitstream(firmware_, *xrt_, settings);
            const auto &p = firmware_.modules.at(firmware_.rfpath)["params"];
            rf_ = std::make_unique<Rfdc>(settings, p, block(firmware_.rfpath));
            if (settings.value("force_init_clks", false) || !rf_->locked(p)) {
                if (!download)
                    fail("RFDC clocks are unlocked; initialize with download=True");
                rf_.reset();
                configure_clocks(nominal["refclk_freq"], settings);
                std::this_thread::sleep_for(1s);
                download_bitstream(firmware_, *xrt_, settings);
                rf_ = std::make_unique<Rfdc>(settings, p, block(firmware_.rfpath));
            }
            if (!rf_->locked(p))
                fail("RFDC PLLs are not locked");
            for (const std::string kind : {"dac", "adc"}) {
                auto key = kind + "_sample_rates";
                if (!settings.contains(key) || settings[key].is_null())
                    continue;
                require(settings[key].is_object(), "Sample rates must map tile IDs to MHz");
                for (auto i = settings[key].begin(); i != settings[key].end(); ++i) {
                    int tile = std::stoi(i.key());
                    require(tile >= 0 && tile < 4, "Requested RFDC tile is not enabled");
                    auto prefix = "C_" + std::string(kind == "dac" ? "DAC" : "ADC") +
                                  std::to_string(tile) + "_";
                    require(p.value(prefix + "Enable", "") == "1",
                            "Requested RFDC tile is not enabled");
                    double frequency = i.value(),
                           maximum = realparam(p, prefix + "Sampling_Rate") * 1000;
                    require((kind == "dac" && tile == 0) || std::abs(frequency - maximum) <= 1e-6,
                            "Sample-rate override is supported only for independent DAC tile 0");
                    require(std::isfinite(frequency) && frequency >= 500 &&
                                frequency <= maximum + 1e-6,
                            "Sample rate exceeds the firmware clock design");
                    rf_->call<decltype(&XRFdc_DynamicPLLConfig)>(
                        "XRFdc_DynamicPLLConfig", kind == "dac" ? 1u : 0u, unsigned(tile), u8(1),
                        realparam(p, prefix + "Refclk_Freq"), frequency);
                }
            }
            auto configuration = firmware_.configuration(rf_->actual(p));
            cfg_ = configuration.first;
            routes_ = configuration.second;
            auto tc = cfg_["tprocs"][0];
            auto maxmem =
                std::max({tc["pmem_size"].get<std::size_t>(), tc["wmem_size"].get<std::size_t>(),
                          tc["dmem_size"].get<std::size_t>()});
            std::size_t maxenv = 0, maxavg = 0, maxbuf = 0;
            for (const auto &g : cfg_["gens"])
                maxenv = std::max(maxenv, g.value("maxlen", std::size_t(0)));
            for (const auto &r : cfg_["readouts"]) {
                maxavg = std::max(maxavg, r["avg_maxlen"].get<std::size_t>());
                maxbuf = std::max(maxbuf, r["buf_maxlen"].get<std::size_t>());
            }
            auto bank = settings.value("xrt_memory_bank", 0u);
            mem_write_ = std::make_unique<Buffer>(*xrt_, maxmem * 32, bank);
            mem_read_ = std::make_unique<Buffer>(*xrt_, maxmem * 32, bank);
            env_write_ =
                std::make_unique<Buffer>(*xrt_, std::max(std::size_t(4), maxenv * 4), bank);
            avg_read_ = std::make_unique<Buffer>(*xrt_, maxavg * 8, bank);
            buf_read_ = std::make_unique<Buffer>(*xrt_, maxbuf * 4, bank);
            stop();
            for (unsigned reg = 0; reg < 9; ++reg)
                tproc().write(reg, 0);
            for (const auto &g : cfg_["gens"]) {
                auto &b = block(s(g, "fullpath"));
                if (g.contains("maxlen")) {
                    b.write(0, 0);
                    b.write(1, 0);
                    if (g["type"] == "axis_signal_gen_v6")
                        b.write(2, 10);
                } else {
                    for (unsigned reg = 0; reg < 24; ++reg)
                        b.write(reg, 0);
                    b.write(24, 1);
                    b.write(24, 0);
                }
            }
            for (const auto &r : cfg_["readouts"]) {
                auto &b = block(s(r, "avgbuf_fullpath"));
                for (unsigned reg : {0u, 3u, 6u, 9u, 12u})
                    b.write(reg, 0);
            }
        } catch (...) {
            ::close(lock_);
            lock_ = -1;
            throw;
        }
    }
    ~BoardHardware() override {
        try {
            close();
        } catch (...) {
        }
        if (lock_ >= 0)
            ::close(lock_);
    }
    void close() {
        if (closed_)
            return;
        closed_ = true;
        stop();
        for (const auto &r : cfg_["readouts"])
            enable(r, false, false);
    }
    Json get_cfg() const override { return cfg_; }
    Frame call(const std::string &op, const Json &a, const Arrays &arrays) override {
        Frame result;
        if (op == "get_cfg") {
            result.meta = {{"value", cfg_}};
            return result;
        }
        if (op == "close") {
            close();
            return result;
        }
        if (closed_)
            fail("Hardware is closed");
        if (op == "prepare_round" || op == "cleanup_round")
            return result;
        if (op == "stop_tproc") {
            stop();
            return result;
        }
        if (op == "start_tproc") {
            if (!(tproc().read(1) & (1u << 10)))
                tproc().write(0, 4);
            return result;
        }
        if (op == "start_src") {
            auto src = s(a, "src");
            require(src == "internal" || src == "external", "Invalid start source");
            stop();
            tproc().write(1, (tproc().read(1) & ~(1u << 10)) | ((src == "external") << 10));
            return result;
        }
        if (op == "clear_tproc_counter" || op == "get_tproc_counter") {
            auto addr = a.value("addr", 1);
            require(addr == 1 || addr == 2, "Invalid tProc counter");
            if (op == "clear_tproc_counter")
                tproc().write(0, 64);
            else {
                tproc().write(8, 1);
                result.meta = {{"value", tproc().read(10 + addr)}};
            }
            return result;
        }
        if (op == "reset_gens") {
            reset_generators();
            return result;
        }
        if (op == "reload_mem") {
            reload();
            return result;
        }
        if (op == "load_bin_program") {
            load_program(arrays, a.value("load_mem", true));
            return result;
        }
        if (op == "load_mem") {
            load_memory(arrays.at("data"), a.value("mem_sel", std::string("dmem")),
                        a.value("addr", std::int64_t(0)));
            return result;
        }
        if (op == "read_mem") {
            auto kind = a.value("mem_sel", std::string("dmem"));
            auto length = a.at("length").get<std::size_t>();
            memory(kind, length, a.value("addr", std::int64_t(0)));
            auto config = tproc().read(1) & ~63u;
            tproc().write(1, config | (kind == "pmem" ? 5 : kind == "dmem" ? 9 : 13));
            try {
                dma(s(routes_, "tproc"), true, *mem_read_, length * 32);
            } catch (...) {
                tproc().write(1, config);
                throw;
            }
            tproc().write(1, config);
            std::size_t columns = kind == "pmem" ? 3 : kind == "wmem" ? 6 : 1;
            Array out;
            out.dtype = "<i4";
            out.shape = kind == "dmem" ? std::vector<std::size_t>{length}
                                       : std::vector<std::size_t>{length, columns};
            out.bytes.resize(length * columns * 4);
            for (std::size_t i = 0; i < length; ++i)
                std::memcpy(out.bytes.data() + i * columns * 4,
                            static_cast<char *>(mem_read_->data) + i * 32, columns * 4);
            result.arrays["data"] = std::move(out);
            return result;
        }
        if (op == "load_weights")
            throw Error("ValueError", "This firmware has no weighted readout buffers");
        if (op == "configure_readout")
            throw Error("ValueError", "This firmware uses tProc-controlled or PFB readouts");
        if (op == "set_nyquist" || op == "set_mixer_freq") {
            const auto &g = gen(a);
            auto name = s(g, "dac");
            unsigned tile = name.at(0) - '0', b = name.at(1) - '0';
            if (op == "set_nyquist") {
                auto zone = a.at("nqz").get<unsigned>();
                require(zone == 1 || zone == 2, "Nyquist zone must be 1 or 2");
                if (!nyquist_.count(name)) {
                    u32 current = 0;
                    rf_->call<decltype(&XRFdc_GetNyquistZone)>("XRFdc_GetNyquistZone", 1u, tile, b,
                                                               &current);
                    nyquist_[name] = current;
                }
                if (a.value("force", false) || nyquist_[name] != zone) {
                    rf_->call<decltype(&XRFdc_SetNyquistZone)>("XRFdc_SetNyquistZone", 1u, tile, b,
                                                               zone);
                    nyquist_[name] = zone;
                }
            } else {
                require(!a.contains("ro_ch") || a["ro_ch"].is_null(),
                        "Send a pre-rounded mixer frequency");
                require(g["has_mixer"].get<bool>(), "Generator has no RF mixer");
                double f = a.at("f");
                require(std::isfinite(f), "Mixer frequency must be finite");
                XRFdc_Mixer_Settings mixer{};
                rf_->call<decltype(&XRFdc_GetMixerSettings)>("XRFdc_GetMixerSettings", 1u, tile, b,
                                                             &mixer);
                if (!mixers_.count(name))
                    mixers_[name] = mixer.Freq;
                if (mixers_[name] != f) {
                    if (mixer.MixerType != 2)
                        fail("RFDC fine mixer is disabled");
                    mixer.EventSource = 0;
                    mixer.Freq = f;
                    mixer.MixerType = 2;
                    mixer.PhaseOffset = 0;
                    rf_->call<decltype(&XRFdc_SetMixerSettings)>("XRFdc_SetMixerSettings", 1u, tile,
                                                                 b, &mixer);
                    rf_->call<decltype(&XRFdc_UpdateEvent)>("XRFdc_UpdateEvent", 1u, tile, b, 1u);
                    if (a.value("phase_reset", true))
                        rf_->call<decltype(&XRFdc_ResetNCOPhase)>("XRFdc_ResetNCOPhase", 1u, tile,
                                                                  b);
                    mixers_[name] = f;
                }
            }
            return result;
        }
        if (op == "config_mux_gen") {
            const auto &g = gen(a);
            const auto &tones = a.at("tones");
            require(g.value("n_tones", 0) == 8 && tones.is_array() && tones.size() <= 8,
                    "Invalid mux tone count");
            auto &b = block(s(g, "fullpath"));
            for (unsigned i = 0; i < 8; ++i)
                if (i < tones.size()) {
                    b.write(i, regvalue(tones[i].at("freq_int")));
                    b.write(i + 8, regvalue(tones[i].at("phase_int")));
                    b.write(i + 16, regvalue(tones[i].at("gain_int")));
                } else
                    b.write(i + 16, 0);
            b.write(24, 1);
            b.write(24, 0);
            return result;
        }
        if (op == "config_mux_readout") {
            auto path = s(a, "pfbpath");
            require(firmware_.modules.contains(path) &&
                        firmware_.modules[path]["type"] == "axis_pfb_readout_v4" &&
                        (!a.contains("sel") || a["sel"].is_null()),
                    "Unsupported PFB readout configuration");
            auto &b = block(path);
            auto nch = param(firmware_.modules[path]["params"], "N");
            for (const auto &c : a.at("cfgs")) {
                int ch = c.at("pfb_ch"), port = c.at("pfb_port");
                require(ch >= 0 && std::uint64_t(ch) < nch && port >= 0 && port < 8,
                        "Invalid PFB channel or output");
                b.write(port, ((ch % 8) << 8) + ch / 8);
                b.write(8 + 2 * port, regvalue(c.at("f_int")));
                b.write(9 + 2 * port, regvalue(c.at("phase_int")));
            }
            return result;
        }
        if (op == "load_envelope") {
            const auto &g = gen(a);
            const auto &data = arrays.at("data");
            auto addr = a.value("addr", std::int64_t(0));
            require(data.dtype == "<i2" && data.shape.size() == 2 && data.shape[1] == 2 &&
                        data.rows() > 0 && data.bytes.size() == data.rows() * 4,
                    "Envelope must be nonempty (N,2) int16");
            auto max = g.value("maxlen", std::size_t(0));
            require(addr >= 0 && std::size_t(addr) <= max && data.rows() <= max - std::size_t(addr),
                    "Envelope memory exceeded");
            require(data.rows() % g["samps_per_clk"].get<std::size_t>() == 0,
                    "Envelope must contain whole fabric cycles");
            auto name = s(g, "fullpath");
            auto route = routes_["gens"][name];
            select(route, true);
            std::memcpy(env_write_->data, data.bytes.data(), data.bytes.size());
            auto &b = block(name);
            b.write(0, std::uint32_t(addr));
            b.write(1, 1);
            try {
                dma(route[0].get<std::string>(), false, *env_write_, data.bytes.size());
            } catch (...) {
                b.write(1, 0);
                throw;
            }
            b.write(1, 0);
            return result;
        }
        if (op == "enable_buf") {
            enable(readout(a), a.value("enable_avg", true), a.value("enable_buf", true));
            return result;
        }
        if (op == "config_avg" || op == "config_buf") {
            const auto &r = readout(a);
            auto address = a.value("address", std::int64_t(0)),
                 length = a.value("length", std::int64_t(1));
            auto &b = block(s(r, "avgbuf_fullpath"));
            if (op == "config_avg") {
                require(length > 0 && std::uint64_t(length) < (std::uint64_t(1) << 32),
                        "Invalid integration length");
                require(address >= 0 &&
                            std::uint64_t(address) < r["avg_maxlen"].get<std::uint64_t>(),
                        "Invalid accumulated buffer address");
                enable(r, false, false);
                b.write(1, std::uint32_t(address));
                b.write(2, std::uint32_t(length));
                b.write(12, a.value("edge_counting", false));
                b.write(13, regvalue(a.value("high_threshold", Json(0))));
                b.write(14, regvalue(a.value("low_threshold", Json(0))));
            } else {
                require(length > 0 &&
                            std::uint64_t(length) < r["buf_maxlen"].get<std::uint64_t>() &&
                            address >= 0 &&
                            std::uint64_t(address) < r["buf_maxlen"].get<std::uint64_t>(),
                        "Invalid decimated buffer window");
                enable(r, false, false);
                b.write(7, std::uint32_t(address));
                b.write(8, std::uint32_t(length));
            }
            return result;
        }
        if (op == "get_accumulated" || op == "get_decimated") {
            const auto &r = readout(a);
            auto kind = std::string(op == "get_accumulated" ? "avg" : "buf");
            auto address = a.value("address", std::int64_t(0));
            auto length = a.contains("length") && !a["length"].is_null()
                              ? a["length"].get<std::int64_t>()
                              : r[kind + "_maxlen"].get<std::int64_t>() - 2;
            require(address >= 0 && length > 0, "Capture transfer exceeds buffer capacity");
            result.arrays["data"] = capture(r, std::size_t(address), std::size_t(length), kind);
            return result;
        }
        if (op == "clocks_locked") {
            result.meta = {
                {"value", rf_->locked(firmware_.modules.at(firmware_.rfpath)["params"])}};
            return result;
        }
        throw Error("ValueError", "Unknown hardware operation: " + op);
    }
};
} // namespace
std::unique_ptr<Backend> make_hardware(const Json &settings) {
    return std::make_unique<BoardHardware>(settings);
}
} // namespace quick
#else
namespace quick {
std::unique_ptr<Backend> make_hardware(const Json &) {
    throw Error("RuntimeError", "Native FPGA hardware access requires Linux on the ZCU216; use "
                                "--inspect-firmware for offline metadata inspection");
}
} // namespace quick
#endif
