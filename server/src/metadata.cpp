// Native port of quickserver/metadata.py. Copyright Open Quantum Hardware,
// MIT.
#include "quick/metadata.hpp"
#include "pugixml.hpp"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <limits>
#include <regex>

namespace quick {
namespace {
std::string str(const Json &j, const std::string &k) { return j.at(k).get<std::string>(); }
std::int64_t integer(const Json &j, const std::string &k) {
    return std::stoll(str(j, k), nullptr, 0);
}
double real(const Json &j, const std::string &k) { return std::stod(str(j, k)); }
int number(const std::string &s) {
    std::smatch m;
    if (!std::regex_search(s, m, std::regex("[0-9]+")))
        throw Error("ValueError", "Missing port number: " + s);
    return std::stoi(m.str());
}
std::int64_t power(const Json &j, const std::string &k) {
    auto n = integer(j, k);
    if (n < 0 || n > 30)
        throw Error("ValueError", "Invalid firmware memory width");
    return std::int64_t(1) << n;
}
std::pair<std::int64_t, std::int64_t> fraction(double x, std::int64_t maxden) {
    if (!std::isfinite(x) || x <= 0)
        throw Error("ValueError", "Invalid firmware frequency ratio");
    std::int64_t p0 = 0, q0 = 1, p1 = 1, q1 = 0;
    double y = x;
    for (int i = 0; i < 64; ++i) {
        auto a = std::int64_t(std::floor(y));
        if (a > maxden || (q1 > 0 && a > (maxden - q0) / q1))
            break;
        auto p2 = p0 + a * p1, q2 = q0 + a * q1;
        p0 = p1;
        q0 = q1;
        p1 = p2;
        q1 = q2;
        auto rem = y - a;
        if (rem < 1e-14)
            return {p1, q1};
        y = 1 / rem;
    }
    if (q1 == 0)
        return {std::int64_t(std::llround(x)), 1};
    auto k = (maxden - q0) / q1;
    auto a = p0 + k * p1, b = q0 + k * q1;
    return std::abs(double(p1) / q1 - x) <= std::abs(double(a) / b - x) ? std::make_pair(p1, q1)
                                                                        : std::make_pair(a, b);
}
} // namespace
Firmware::Firmware(const std::string &path) {
    bitfile = std::filesystem::absolute(path).lexically_normal().string();
    auto hwh = std::filesystem::path(bitfile);
    hwh.replace_extension(".hwh");
    hwhfile = hwh.string();
    pugi::xml_document doc;
    auto result = doc.load_file(hwhfile.c_str());
    if (!result)
        throw Error("ValueError", "Cannot parse HWH: " + std::string(result.description()));
    auto root = doc.document_element();
    timestamp = root.attribute("TIMESTAMP").value();
    for (auto node : root.child("MODULES").children("MODULE")) {
        std::string name = node.attribute("FULLNAME").value();
        if (!name.empty() && name[0] == '/')
            name.erase(0, 1);
        Json m = {{"type", node.attribute("MODTYPE").value()},
                  {"version", node.attribute("HWVERSION").value()},
                  {"revision", node.attribute("COREREVISION").as_int(0)},
                  {"params", Json::object()},
                  {"ports", Json::object()},
                  {"buses", Json::object()}};
        for (auto p : node.child("PARAMETERS").children("PARAMETER"))
            m["params"][p.attribute("NAME").value()] = p.attribute("VALUE").value();
        for (auto p : node.child("PORTS").children("PORT")) {
            std::string port = p.attribute("NAME").value();
            Json attrs = Json::object();
            for (auto a : p.attributes())
                attrs[a.name()] = a.value();
            m["ports"][port] = attrs;
            signals[attrs.value("SIGNAME", "")].push_back({name, port});
        }
        for (auto p : node.child("BUSINTERFACES").children("BUSINTERFACE")) {
            std::string port = p.attribute("NAME").value(), bus = p.attribute("BUSNAME").value();
            m["buses"][port] = bus;
            if (bus != "__NOC__")
                buses[bus].push_back({name, port});
        }
        modules[name] = m;
    }
    for (auto p : root.child("EXTERNALPORTS").children("PORT"))
        signals[p.attribute("SIGNAME").value()].push_back({"", p.attribute("NAME").value()});
    tproc = single("qick_processor");
    rfpath = single("usp_rf_data_converter");
    int rev = modules[tproc]["revision"];
    if (rev < 21 || rev > 27)
        throw Error("ValueError", "Unsupported tProc v2 revision");
    fingerprint = sha256_file(bitfile) + ":" + sha256_file(hwhfile);
}
std::string Firmware::single(const std::string &kind) const {
    std::vector<std::string> names;
    for (auto i = modules.begin(); i != modules.end(); ++i)
        if (i.value()["type"] == kind)
            names.push_back(i.key());
    if (names.size() != 1)
        throw Error("ValueError", "Expected one " + kind);
    return names.front();
}
Firmware::Endpoint Firmware::peer(const std::string &name, const std::string &port) const {
    auto bus = str(modules.at(name).at("buses"), port);
    std::vector<Endpoint> others;
    auto found = buses.find(bus);
    if (found != buses.end())
        for (auto x : found->second)
            if (x != Endpoint{name, port})
                others.push_back(x);
    if (others.size() != 1)
        throw Error("ValueError", "Ambiguous AXIS connection " + name + "/" + port);
    return others.front();
}
Firmware::Endpoint Firmware::trace(std::string name, std::string port,
                                   const std::set<std::string> &kinds, bool backward) const {
    std::set<Endpoint> seen;
    while (seen.emplace(name, port).second) {
        auto endpoint = peer(name, port);
        name = endpoint.first;
        port = endpoint.second;
        const auto &m = modules.at(name);
        auto type = str(m, "type");
        if (kinds.count(type))
            return {name, port};
        if (type == "axis_cdcsync_v1")
            port = (backward ? "s" : "m") + port.substr(1);
        else {
            std::vector<std::string> choices;
            for (auto i = m["buses"].begin(); i != m["buses"].end(); ++i) {
                auto p = i.key();
                std::transform(p.begin(), p.end(), p.begin(),
                               [](unsigned char c) { return char(std::tolower(c)); });
                if (p[0] == (backward ? 's' : 'm') && p.find("axis") != std::string::npos)
                    choices.push_back(i.key());
            }
            if (choices.size() != 1)
                throw Error("ValueError", "Unsupported AXIS route through " + name);
            port = choices.front();
        }
    }
    throw Error("ValueError", "Cycle in AXIS graph");
}
Json Firmware::controller(const std::string &name, const std::string &port) const {
    auto e = trace(name, port, {"qick_processor", "axis_tmux_v1"});
    Json c = Json::object();
    if (modules.at(e.first)["type"] == "axis_tmux_v1") {
        c["tmux_ch"] = number(e.second);
        e = trace(e.first, "s_axis", {"qick_processor"});
    }
    c["tproc_ch"] = number(e.second);
    return c;
}
Json Firmware::dma_route(const std::string &name, const std::string &port, bool backward) const {
    auto e = trace(name, port, {"axi_dma", "axis_switch"}, backward);
    Json sw = nullptr, ch = nullptr;
    if (modules.at(e.first)["type"] == "axis_switch") {
        sw = e.first;
        ch = number(e.second);
        e = trace(e.first, backward ? "S00_AXIS" : "M00_AXIS", {"axi_dma"}, backward);
    }
    return Json::array({e.first, sw, ch});
}
double Firmware::clock(const std::string &name, const std::string &port,
                       const Json &channels) const {
    const auto &desc = modules.at(name).at("ports").at(port);
    double nominal = real(desc, "CLKFREQUENCY") / 1e6;
    auto signal = str(desc, "SIGNAME");
    for (int depth = 0; depth < 10; ++depth) {
        std::vector<Endpoint> drivers;
        auto f = signals.find(signal);
        if (f == signals.end())
            break;
        for (auto e : f->second)
            if (!e.first.empty() && modules.at(e.first)["ports"][e.second].value("DIR", "") == "O")
                drivers.push_back(e);
        if (drivers.size() != 1)
            break;
        auto e = drivers.front();
        std::smatch match;
        if (e.first == rfpath &&
            std::regex_match(e.second, match, std::regex("clk_(adc|dac)([0-9])"))) {
            auto kind = match[1].str(), tile = match[2].str();
            for (auto i = channels[kind].begin(); i != channels[kind].end(); ++i)
                if (i.key().substr(0, 1) == tile) {
                    auto upper = kind;
                    std::transform(upper.begin(), upper.end(), upper.begin(), ::toupper);
                    double base =
                        real(modules.at(rfpath)["params"], "C_" + upper + tile + "_Sampling_Rate") *
                        1000;
                    auto r = fraction(i.value()["fs"].get<double>() / base, 100000);
                    return nominal * (double(r.first) / r.second);
                }
        }
        std::vector<Json> inputs;
        for (auto p : modules.at(e.first)["ports"])
            if (p.value("DIR", "") == "I" && p.value("SIGIS", "") == "clk")
                inputs.push_back(p);
        if (inputs.size() != 1)
            break;
        signal = str(inputs.front(), "SIGNAME");
    }
    return nominal;
}
std::pair<Json, Json> Firmware::configuration(const Json &actual) const {
    const auto &pars = modules.at(rfpath)["params"];
    Json channels = {{"dac", Json::object()}, {"adc", Json::object()}};
    std::set<double> refs;
    for (const std::string kind : {"dac", "adc"})
        for (int tile = 0; tile < 4; ++tile) {
            auto upper = kind;
            std::transform(upper.begin(), upper.end(), upper.begin(), ::toupper);
            auto ts = std::to_string(tile), prefix = "C_" + upper + ts + "_";
            if (pars.value(prefix + "Enable", "") != "1")
                continue;
            double ref = real(pars, prefix + "Refclk_Freq"),
                   nominal = real(pars, prefix + "Sampling_Rate") * 1000;
            auto fabric_div = std::llround(nominal / real(pars, prefix + "Fabric_Freq"));
            auto ratio = fraction(nominal / ref, 10000);
            auto mult = ratio.first, div = ratio.second;
            Json hw = Json::object();
            if (actual.contains(kind) && actual[kind].contains(ts)) {
                hw = actual[kind][ts];
                ref = hw.at("ref");
                mult = hw.at("mult");
                div = hw.at("div");
            }
            double fs = ref * double(mult) / double(div);
            refs.insert(ref);
            for (int block = 0; block < 4; ++block) {
                auto key = ts + std::to_string(block);
                if (pars.value("C_" + upper + "_Slice" + key + "_Enable", "") != "true")
                    continue;
                auto factor = integer(
                    pars, "C_" + upper + "_" +
                              (kind == "dac" ? "Interpolation_Mode" : "Decimation_Mode") + key);
                if (hw.contains("factors"))
                    factor = hw["factors"].at(std::to_string(block));
                channels[kind][key] = {{"fs", fs},
                                       {"fs_mult", mult},
                                       {"fs_div", div},
                                       {"f_fabric", fs / fabric_div},
                                       {kind == "dac" ? "interpolation" : "decimation", factor}};
            }
        }
    if (refs.size() != 1)
        throw Error("ValueError", "ZCU216 needs one common RF reference frequency");
    const auto &tm = modules.at(tproc);
    const auto &tp = tm["params"];
    Json tc = {{"type", "qick_processor"},
               {"fullpath", tproc},
               {"revision", tm["revision"]},
               {"version", tm["version"]},
               {"f_time", clock(tproc, "t_clk_i", channels)},
               {"f_core", clock(tproc, "c_clk_i", channels)},
               {"output_pins", Json::array()},
               {"start_pin", nullptr},
               {"stop_pin", nullptr}};
    for (const std::string k : {"pmem", "dmem", "wmem"}) {
        auto up = k;
        std::transform(up.begin(), up.end(), up.begin(), ::toupper);
        tc[k + "_size"] = power(tp, up + "_AW");
    }
    tc["dreg_qty"] = power(tp, "REG_AW");
    tc["fifo_depth"] = power(tp, "FIFO_DEPTH");
    for (auto i = tm["ports"].begin(); i != tm["ports"].end(); ++i) {
        auto f = signals.find(i.value().value("SIGNAME", ""));
        if (f == signals.end())
            continue;
        std::vector<std::string> external;
        for (auto e : f->second)
            if (e.first.empty())
                external.push_back(e.second);
        if (external.empty())
            continue;
        if (std::regex_match(i.key(), std::regex("trig_[0-9]+_o")))
            tc["output_pins"].push_back(
                Json::array({"trig", number(i.key()), 0, external.front()}));
        if (i.key() == "proc_start_i")
            tc["start_pin"] = external.front();
        if (i.key() == "proc_stop_i")
            tc["stop_pin"] = external.front();
    }
    std::sort(tc["output_pins"].begin(), tc["output_pins"].end(),
              [](const Json &a, const Json &b) { return a[1].get<int>() < b[1].get<int>(); });
    Json gens = Json::array(), ros = Json::array(),
         routes = {{"gens", Json::object()}, {"readouts", Json::object()}};
    const std::set<std::string> generator_types = {"axis_signal_gen_v6", "axis_sg_int4_v2",
                                                   "axis_sg_mixmux8_v1"};
    for (auto it = modules.begin(); it != modules.end(); ++it) {
        auto name = it.key();
        const auto &m = it.value();
        auto type = str(m, "type");
        if (!generator_types.count(type))
            continue;
        auto e = trace(name, "m_axis", {"usp_rf_data_converter"}, false);
        auto dac = e.second.substr(1, 2);
        Json g = {{"type", type},
                  {"fullpath", name},
                  {"dac", dac},
                  {"revision", m["revision"]},
                  {"version", m["version"]},
                  {"maxv", 32766},
                  {"has_dds", true},
                  {"b_dds", 32},
                  {"b_phase", 32},
                  {"has_mixer", type != "axis_signal_gen_v6"}};
        g.update(channels["dac"].at(dac));
        g.update(controller(name, type == "axis_sg_mixmux8_v1" ? "s_axis" : "s1_axis"));
        g["f_dds"] = g["fs"].get<double>() / g["interpolation"].get<double>();
        g["fdds_div"] = g["fs_div"].get<std::int64_t>() * g["interpolation"].get<std::int64_t>();
        if (type == "axis_sg_mixmux8_v1") {
            g["n_tones"] = 8;
            g["has_gain"] = true;
            g["has_phase"] = true;
        } else {
            int samps = type == "axis_signal_gen_v6" ? 16 : 1;
            g["maxlen"] = power(m["params"], "N") * samps;
            g["samps_per_clk"] = samps;
            g["complex_env"] = true;
            g["maxv_scale"] = samps == 16 ? 1.0 : 0.9;
            routes["gens"][name] = dma_route(name, "s0_axis", true);
        }
        gens.push_back(g);
    }
    std::sort(gens.begin(), gens.end(), [](const Json &a, const Json &b) {
        return std::make_pair(a["tproc_ch"].get<int>(), a.value("tmux_ch", -1)) <
               std::make_pair(b["tproc_ch"].get<int>(), b.value("tmux_ch", -1));
    });
    for (auto it = modules.begin(); it != modules.end(); ++it) {
        auto name = it.key();
        const auto &m = it.value();
        if (m["type"] != "axis_avg_buffer")
            continue;
        if (m["version"] != "1.2")
            throw Error("ValueError", "Only axis_avg_buffer 1.2 is supported");
        auto ro = trace(name, "s_axis", {"axis_dyn_readout_v1", "axis_pfb_readout_v4"});
        const auto &rm = modules.at(ro.first);
        bool dyn = rm["type"] == "axis_dyn_readout_v1";
        auto adc = trace(ro.first, dyn ? "s1_axis" : "s_axis", {"usp_rf_data_converter"})
                       .second.substr(1, 2);
        Json r = {{"adc", adc},
                  {"avg_maxlen", power(m["params"], "N_AVG")},
                  {"buf_maxlen", power(m["params"], "N_BUF")},
                  {"has_edge_counter", true},
                  {"has_weights", false},
                  {"b_phase", 32},
                  {"b_dds", 32},
                  {"has_outsel", dyn},
                  {"iq_offset", dyn ? 0.0 : -0.5},
                  {"avgbuf_type", m["type"]},
                  {"avgbuf_fullpath", name},
                  {"avgbuf_revision", m["revision"]},
                  {"avgbuf_version", m["version"]},
                  {"ro_type", rm["type"]},
                  {"ro_fullpath", ro.first},
                  {"ro_revision", rm["revision"]},
                  {"ro_version", rm["version"]}};
        r.update(channels["adc"].at(adc));
        auto trigger = str(m["ports"]["trigger"], "SIGNAME");
        std::vector<Endpoint> peers;
        auto sig = signals.find(trigger);
        if (sig != signals.end())
            for (auto e : sig->second)
                if (e.first == tproc)
                    peers.push_back(e);
        if (peers.size() != 1 || peers.front().second.rfind("trig_", 0) != 0)
            throw Error("ValueError", "Unsupported readout trigger wiring");
        r["trigger_type"] = "tport";
        r["trigger_port"] = number(peers.front().second);
        r["trigger_bit"] = 0;
        r["tproc_ch"] = number(trace(name, "m2_axis", {"qick_processor"}, false).second);
        int down = dyn ? 8 : int(integer(rm["params"], "N") / 2);
        if (dyn) {
            auto c = controller(ro.first, "s0_axis");
            r["tproc_ctrl"] = c["tproc_ch"];
            c.erase("tproc_ch");
            r.update(c);
        } else {
            r["pfb_port"] = number(ro.second);
            r["pfb_nch"] = down * 2;
            r["pfb_nout"] = 8;
            r["pfb_ch_offset"] = down;
            r["pfb_dds_on_output"] = true;
        }
        r["f_dds"] = r["fs"].get<double>() / r["decimation"].get<double>() / (dyn ? 1 : down);
        r["fdds_div"] = r["fs_div"].get<std::int64_t>() * r["decimation"].get<std::int64_t>() *
                        (dyn ? 1 : down);
        r["f_output"] = r["fs"].get<double>() / (r["decimation"].get<double>() * down);
        auto avg = dma_route(name, "m0_axis", false), buf = dma_route(name, "m1_axis", false);
        if (avg[2] != buf[2])
            throw Error("ValueError", "Readout switches disagree on channel numbering");
        routes["readouts"][name] = {{"avg", avg}, {"buf", buf}};
        ros.push_back(r);
    }
    std::sort(ros.begin(), ros.end(), [&](const Json &a, const Json &b) {
        return routes["readouts"][str(a, "avgbuf_fullpath")]["avg"][2] <
               routes["readouts"][str(b, "avgbuf_fullpath")]["avg"][2];
    });
    routes["tproc"] = dma_route(tproc, "s_dma_axis_i", true)[0];
    Json cfg = {{"board", "ZCU216"},
                {"sw_version", "quick-server-1"},
                {"firmware_id", fingerprint},
                {"fw_timestamp", timestamp},
                {"refclk_freq", *refs.begin()},
                {"tprocs", Json::array({tc})},
                {"gens", gens},
                {"readouts", ros},
                {"iqs", Json::array()},
                {"time_taggers", Json::array()},
                {"extra_description", Json::array()}};
    if (gens.empty() || ros.empty())
        throw Error("ValueError", "Firmware has no supported generators/readouts");
    return {cfg, routes};
}
Json inspect_firmware(const std::string &bitfile) {
    Firmware fw(bitfile);
    auto result = fw.configuration();
    return {{"cfg", result.first}, {"routes", result.second}};
}
} // namespace quick
