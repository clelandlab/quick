# Board server

`quickserver` runs the supplied ZCU216 tProc v2 firmware through PYNQ, RFDC registers and DMA. It receives compiled programs over TCP port 9999 and streams accumulated IQ. Decimated capture uses finite firmware buffers. QICK and Pyro are not runtime dependencies.

The board needs its existing Python 3.8+, NumPy, PYNQ, `xrfdc` and `xrfclk` environment. The server source is in [quickserver](quickserver/); PC transport is in [client.py](../quick/client.py), with identical framing implementations in [quick/wire.py](../quick/wire.py) and [quickserver/wire.py](quickserver/wire.py). [quick_server.service](quick_server.service) supplies automatic startup.

## Board installation

The separate `board_setup_next` folder contains `install.sh`, `uninstall.sh`, `dependency.tar.gz` and its checksum. Copy that folder to the board and run:

```sh
sudo bash install.sh
```

The installer preserves clock settings and installs the service under `/home/xilinx/jupyter_notebooks/quick_server`, with configuration in `/home/xilinx/quick_server.json`. It supports `--yes` to retain the current hostname and `--no-reboot` to avoid rebooting. The old Pyro service is disabled; do not run both services against the FPGA together.

```sh
systemctl status quick_server
journalctl -u quick_server
```

The setup folder also contains the offline bundle builder. Firmware and derived code retain their license notices; see [FIRMWARE_LICENSE](FIRMWARE_LICENSE).

## Client use

Install the PC package from the repository root, then connect:

```python
import quick
soccfg, soc = quick.connect("10.0.1.91")
```

A new default connection cancels the previous run, waits for ADC recovery, and resets outputs. Use `takeover=False` for an observation connection. `soc.stop()` cancels the current client's run; `soc.reset_gens()` stops persistent outputs. Close the client when finished.

Network and acquisition timeouts are separate. Heartbeats keep long programs connected, but do not prove hardware progress. Interrupted acquisitions are not replayed automatically. A DMA fault requires service restart; missed or overlapping ADC triggers remain firmware limits. See the [connection API](../docs/API%20References/helper.md) and [Mercator tutorial](../docs/Tutorials/mercator.md).
