#pragma once
#include "quick/common.hpp"
namespace quick {
Frame recv_frame(int fd);
void send_frame(int fd, const Frame &frame);
void configure_socket(int fd);
void socket_timeout(int fd, double seconds);
bool readable(int fd, double seconds);
bool peer_closed(int fd);
void close_socket(int fd);
} // namespace quick
