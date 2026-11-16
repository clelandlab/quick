#pragma once
#include "quick/common.hpp"
#include <set>
#include <utility>

namespace quick {
class Firmware {
  public:
    explicit Firmware(const std::string &bitfile);
    std::string bitfile, hwhfile, tproc, rfpath, fingerprint, timestamp;
    Json modules = Json::object();
    using Endpoint = std::pair<std::string, std::string>;
    std::map<std::string, std::vector<Endpoint>> buses, signals;
    std::pair<Json, Json> configuration(const Json &actual = Json::object()) const;

  private:
    std::string single(const std::string &kind) const;
    Endpoint peer(const std::string &name, const std::string &port) const;
    Endpoint trace(std::string name, std::string port, const std::set<std::string> &kinds,
                   bool backward = true) const;
    Json controller(const std::string &name, const std::string &port) const;
    Json dma_route(const std::string &name, const std::string &port, bool backward) const;
    double clock(const std::string &name, const std::string &port, const Json &channels) const;
};
Json inspect_firmware(const std::string &bitfile);
} // namespace quick
