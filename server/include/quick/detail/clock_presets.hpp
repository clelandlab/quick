#pragma once
#include <algorithm>
#include <cctype>
#include <cmath>
#include <regex>
#include <string>

namespace quick::detail {
// PYNQ normalizes the chip name parsed from TICS preset filenames to lowercase.
// Board packages commonly ship uppercase LMK04828_245.76.txt/LMX2594_491.52.txt.
inline bool clock_preset_matches(const std::string &filename, std::string chip, double frequency) {
    static const std::regex pattern("([A-Za-z0-9]+)_([0-9]+(?:\\.[0-9]+)?)\\.txt");
    std::smatch match;
    if (!std::isfinite(frequency) || !std::regex_match(filename, match, pattern))
        return false;
    auto preset_chip = match[1].str();
    auto lower = [](unsigned char c) { return static_cast<char>(std::tolower(c)); };
    std::transform(preset_chip.begin(), preset_chip.end(), preset_chip.begin(), lower);
    std::transform(chip.begin(), chip.end(), chip.begin(), lower);
    if (preset_chip != chip)
        return false;
    try {
        return std::abs(std::stod(match[2].str()) - frequency) <= 1e-8;
    } catch (const std::exception &) {
        return false;
    }
}
} // namespace quick::detail
