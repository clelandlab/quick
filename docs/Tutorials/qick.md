# QICK Setup 

This is a tutorial for setting up a new QICK board that is compatible for this package.

**This tutorial is only for ZCU216 hardware!**

## Hardware and Operating System

Assemble the hardware. Note that all the mode switches should be left, except the 0 should be right(on).

Use the [Win32 Disk Imager](https://sourceforge.net/projects/win32diskimager/) to write the pynq operating system into the SD card. The system image for ZCU216 can be downloaded [here](https://drive.google.com/file/d/10kDKrEqA4l0_S3ysTlWbbOHgTsV0Zpyq/view?usp=sharing), as provided by this [Github issue](https://github.com/sarafs1926/ZCU216-PYNQ/issues/1).

Plug in the SD card and power on the board.

## Network Setup

Connect the QICK board to a router via an Ethernet cable. Connect a computer to the same router.

Open the router's configuration page and find the IP address of the QICK board. By default, the QICK board uses DHCP to get an IP address automatically. This IP address can be found on the router configuration page under "Connected Devices", "DHCP Clients" or similar lists.

The IP address is labeled as `IP_ADDR` in the following steps.

## Install qick

Download the dependency and installation script from the [DropBox folder](https://www.dropbox.com/scl/fo/q5jk1mnduqls0lip6j0pf/ADj78VmSjqMefo2ei2uqL-Y?rlkey=vuk3ggd9mad78lnavzb1j28m7&st=i6ql1uwu&dl=0). Open a terminal in the folder to execute the following commands.

Transfer dependency and installation script to the QICK board:

```
scp install.sh dependency.tar.gz xilinx@IP_ADDR:~
```

Login to the QICK board via ssh:

```
ssh xilinx@IP_ADDR
```

On the QICK board, run the installation script:

```
sudo ./install.sh
```

Wait until the installation is complete. The script will ask you to set up a host name.

## Static IP (optional)

> SSH into the QICK board and run the following commands as root.
>
> Enter root user by command `sudo -s`.

To set up a static IP address, edit `/etc/network/interfaces.d/eth0` by the command

```
nano /etc/network/interfaces.d/eth0
```

and overwrite it with the following content

> Here, the desired IP address is `192.168.1.100`, the gateway (IP address of the router) is `192.168.1.1`, and the netmask is `255.255.255.0`.
>
> You should change these values into your desired configuration.

```
auto eth0
iface eth0 inet static
address 192.168.1.100
netmask 255.255.255.0
gateway 192.168.1.1
```

**Triple check before saving the file! You will need to redo everything in this tutorial if you made a typo here.**

Restart the QICK board to let the changes take effect.

