#include "quick/hardware.hpp"
#include "quick/metadata.hpp"
#include "quick/server.hpp"
#include <csignal>
#include <fstream>
#include <iostream>
#include <string>

namespace {
quick::Server *active_server = nullptr;
void stop(int) {
    if (active_server)
        active_server->request_stop();
}
std::string read_file(const std::string &path) {
    std::ifstream f(path);
    if (!f)
        throw quick::Error("ValueError", "Cannot open configuration: " + path);
    return std::string((std::istreambuf_iterator<char>(f)), {});
}
void help() {
    std::cout << "QuICK native C++ server for ZCU216 / PYNQ 3.0.1\n"
                 "Usage: quick-server --config FILE [overrides]\n"
                 "       quick-server --bitfile FILE [--attach] [overrides]\n"
                 "       quick-server --inspect-firmware FILE\n\n"
                 "--config FILE             JSON server/hardware settings\n"
                 "--bitfile FILE            Supported tProc v2 .bit with adjacent .hwh\n"
                 "--attach                  Use already configured FPGA (no download)\n"
                 "--host ADDRESS            Bind IPv4 address (default 0.0.0.0)\n"
                 "--port NUMBER             TCP port (default 9999)\n"
                 "--queue-depth NUMBER      Bounded outgoing acquisition queue (default 8)\n"
                 "--cache-bytes NUMBER      Program cache byte budget\n"
                 "--max-clients NUMBER      Normal TCP session limit (default 8)\n"
                 "--heartbeat-interval SEC  Stream heartbeat interval (default 1)\n"
                 "--dma-timeout SEC         Bounded DMA wait (default 5)\n"
                 "--dac-sample-rates JSON   Mapping of DAC tile to MHz\n"
                 "--adc-sample-rates JSON   Mapping of ADC tile to MHz\n"
                 "--force-init-clks         Reprogram the RF clock chips\n"
                 "--inspect-firmware FILE  Print nominal metadata without touching hardware\n";
}
} // namespace
int main(int argc, char **argv) {
    try {
        quick::Json settings = quick::Json::object(), overrides = quick::Json::object();
        std::string inspect;
        for (int i = 1; i < argc; ++i) {
            std::string opt = argv[i];
            if (opt == "--help" || opt == "-h") {
                help();
                return 0;
            }
            if (opt == "--version") {
                std::cout << "quick-server 1.0 C++17\n";
                return 0;
            }
            if (opt == "--attach") {
                overrides["download"] = false;
                continue;
            }
            if (opt == "--force-init-clks") {
                overrides["force_init_clks"] = true;
                continue;
            }
            if (i + 1 >= argc)
                throw quick::Error("ValueError", "Missing value for " + opt);
            std::string value = argv[++i];
            if (opt == "--config") {
                settings = quick::parse_json(read_file(value));
                if (!settings.is_object())
                    throw quick::Error("ValueError", "config must be a JSON object");
            } else if (opt == "--inspect-firmware")
                inspect = value;
            else if (opt == "--bitfile")
                overrides["bitfile"] = value;
            else if (opt == "--host")
                overrides["host"] = value;
            else if (opt == "--dac-sample-rates")
                overrides["dac_sample_rates"] = quick::parse_json(value);
            else if (opt == "--adc-sample-rates")
                overrides["adc_sample_rates"] = quick::parse_json(value);
            else if (opt == "--port" || opt == "--queue-depth" || opt == "--cache-bytes" ||
                     opt == "--max-clients" || opt == "--heartbeat-interval" ||
                     opt == "--dma-timeout") {
                auto key = opt.substr(2);
                for (auto &c : key)
                    if (c == '-')
                        c = '_';
                overrides[key] = quick::parse_json(value);
            } else
                throw quick::Error("ValueError", "Unknown option: " + opt);
        }
        if (!inspect.empty()) {
            std::cout << quick::canonical_json(quick::inspect_firmware(inspect)) << '\n';
            return 0;
        }
        settings.update(overrides);
        if (!settings.contains("bitfile") || !settings["bitfile"].is_string() ||
            settings["bitfile"].get<std::string>().empty()) {
            help();
            throw quick::Error("ValueError", "--bitfile or config.bitfile is required");
        }
        if (!settings.contains("host"))
            settings["host"] = "0.0.0.0";
        auto hardware = quick::make_hardware(settings);
        quick::Server server(*hardware, settings);
        active_server = &server;
        std::signal(SIGTERM, stop);
        std::signal(SIGINT, stop);
        std::signal(SIGPIPE, SIG_IGN);
        std::cerr << "QuICK native TCP listening on " << settings["host"].get<std::string>() << ':'
                  << server.port() << '\n';
        server.run();
        server.shutdown();
        active_server = nullptr;
        return 0;
    } catch (const quick::Error &e) {
        std::cerr << e.type << ": " << e.what() << '\n';
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << '\n';
    }
    return 1;
}
