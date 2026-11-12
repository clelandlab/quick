# Board Setup

This checkout controls the supplied ZCU216 firmware through the QuICK TCP server on port **9999**. The board needs a compatible PYNQ image with Python 3.8+, NumPy, `pynq`, `xrfdc` and `xrfclk` already installed. The installer supplies the server and firmware; it does not install the operating system.

## Install the server

Connect the board and control PC to the same laboratory network. Find the board address using your router or existing network configuration. The following commands use `10.0.1.91`; substitute your board address.

Obtain the separate `board_setup_next` bundle. From the directory containing that folder, copy its installer, runtime archive and checksum to the board:

```sh
ssh xilinx@10.0.1.91 'mkdir -p ~/quick_setup'
scp board_setup_next/install.sh board_setup_next/dependency.tar.gz board_setup_next/dependency.sha256 xilinx@10.0.1.91:quick_setup/
ssh -t xilinx@10.0.1.91 'cd ~/quick_setup && sudo bash install.sh'
```

The installer checks the existing libraries, asks for a hostname, preserves clock settings, installs the firmware/server, and enables automatic startup. It reboots after successful installation. To keep the current hostname and defer reboot, run `sudo bash install.sh --yes --no-reboot` instead.

See the bundle's `README.md` for configuration, rollback and uninstall instructions. Runtime files are installed under `/home/xilinx/jupyter_notebooks/quick_server`; configuration is stored in `/home/xilinx/quick_server.json`.

## Connect from the PC

Install this source checkout with Python 3.13+ from the repository root:

```sh
python -m pip install -e .
```

```python
import quick

soccfg, soc = quick.connect("10.0.1.91")
try:
    print(soccfg)
finally:
    soc.close()
```

A connection starts a new experimental session by default: it cancels the previous board run and resets outputs. See the [connection API](../API%20References/helper.md) for timeout and takeover settings, then [Build an Experiment](build.md) for pulse protocols.

## Service control

On the board:

```sh
sudo systemctl status quick_server
sudo journalctl -u quick_server -n 50 --no-pager
sudo systemctl restart quick_server
```

Only one process may own the FPGA and DMA engines. Stop an active experiment before restarting the service. For an operating-system reboot, use `sudo reboot` over SSH.
