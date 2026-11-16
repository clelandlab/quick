# Native QuICK board server

`quick-server` is a C++17 server for ZCU216 tProc v2 firmware on PYNQ 3.0.1.
It implements the existing QK01 protocol and works with the unchanged Python
PC client and compiler. The server does not start or embed Python.

## Build

Run from the repository root on Linux with GCC 9 or later and CMake 3.16 or later:

```sh
cmake -S server -B server/build -DCMAKE_BUILD_TYPE=Release
cmake --build server/build --parallel 2
server/build/quick-server --help
```

Dependencies are vendored for offline compilation. Building does not require
Python or access FPGA hardware. Windows users can build through WSL.

For an AArch64 cross-build using the verified Arm GNU 10.2-2020.11 toolchain:

```sh
cmake -S server -B server/build-arm \
  -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64-linux.cmake \
  -DQUICK_CROSS_PREFIX=/opt/gcc-arm-10.2-2020.11-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu- \
  -DCMAKE_BUILD_TYPE=Release -DQUICK_STATIC_CXX_RUNTIME=ON
cmake --build server/build-arm --parallel 2
```

Use a toolchain whose glibc baseline is no newer than the board's. The verified
one uses glibc 2.31. `QUICK_STATIC_CXX_RUNTIME` links libstdc++ and libgcc
statically; glibc and PYNQ's native libraries remain dynamically linked.

## Run

The server uses PYNQ's installed `libxrfdc.so`, `libxrt_core.so`, xclbin and
LMK/LMX clock presets. Root access is needed for FPGA programming, register access,
DMA and SPI. Only one server may own the FPGA at a time.

```sh
sudo server/build/quick-server --config /home/xilinx/quick_server.json
```

Example configuration:

```json
{
  "bitfile": "/home/xilinx/jupyter_notebooks/quick_server/firmware/qick_216.bit",
  "host": "0.0.0.0",
  "port": 9999,
  "download": true,
  "dac_sample_rates": {"0": 6881.28}
}
```

The matching `.hwh` must sit beside the bitstream. CLI options override JSON
settings; use `--help` for supported options. `--attach` skips FPGA programming
and requires matching firmware, usable XRT memory topology and locked RFDC clocks.

Firmware metadata can be inspected without hardware access:

```sh
server/build/quick-server --inspect-firmware /path/to/qick_216.bit
```

`quick_server.service` provides the systemd unit. The separately prepared
`board_setup_next` bundle contains `dependency.tar.gz` and `install.sh` for
deployment; it is outside this source repository.
