# QICK Setup 

This is a tutorial for setting up a new QICK board that is compatible for this package.

**This tutorial is only for ZCU208 and ZCU216 hardware!**

## Hardware and Operating System

Assemble the hardware. Note that all the mode switches should be left, except the 0 should be right(on).

Use the [Win32 Disk Imager](https://sourceforge.net/projects/win32diskimager/) to write the PYNQ operating system into the SD card. The system image can be downloaded [here](https://www.xilinx.com/bin/public/openDownload?filename=zcu208_v3.0.1.zip). Note that we are using [PYNQ](https://www.pynq.io/boards.html) 3.0.1 for ZCU208 here. If you are using ZCU216 hardware, you need to do an extra step below.

> There is no official support for PYNQ on ZCU216. You can also use the PYNQ 2.7.0 image [here](https://drive.google.com/file/d/10kDKrEqA4l0_S3ysTlWbbOHgTsV0Zpyq/view?usp=sharing), as provided by this [Github issue](https://github.com/sarafs1926/ZCU216-PYNQ/issues/1).

Plug in the SD card and power on the board.

## Network Setup

Connect the QICK board to a router via an Ethernet cable. Connect a computer to the same router.

Open the router's configuration page and find the IP address of the QICK board. By default, the QICK board uses DHCP to get an IP address automatically. This IP address can be found on the router configuration page under "Connected Devices", "DHCP Clients" or similar lists.

The IP address is labeled as `$IP` in the following steps. Now you can access the QICK board via SSH. (password is `xilinx`)

```
ssh xilinx@$IP
```

To enter the root user, use the command

```
sudo -s
```

### Static IP (optional)

Edit `/etc/network/interfaces.d/eth0` by the command (with root):

```
nano /etc/network/interfaces.d/eth0
```

Overwrite the whole file with the following content

> In the example here, the target IP address is `192.168.1.100`, the gateway (IP address of the router) is `192.168.1.1`, and the netmask is `255.255.255.0`.
>
> You should change these values to your desired configuration.

```
auto eth0
iface eth0 inet static
address 192.168.1.100
netmask 255.255.255.0
gateway 192.168.1.1
```

**Triple check before saving the file! You will need to redo everything in this tutorial if you made a typo here.**

Restart the QICK board to let the changes take effect.

## ZCU216 PYNQ Setup

If you are using the ZCU208 image on ZCU216, you need to edit `/etc/profile.d/boardname.sh` with the command (with root):

```
nano /etc/profile.d/boardname.sh
```

overwrite the whole file with the following content:

```
export BOARD=ZCU216
```

Restart the QICK board to let the changes take effect.

## Install qick

Download the dependency and installation script from the [DropBox folder](https://www.dropbox.com/scl/fo/q5jk1mnduqls0lip6j0pf/ADj78VmSjqMefo2ei2uqL-Y?rlkey=vuk3ggd9mad78lnavzb1j28m7&st=i6ql1uwu&dl=0). Open a terminal in the folder to execute the following commands.

Transfer dependency and installation script to the QICK board:

```
scp install.sh dependency.tar.gz xilinx@$IP:~
```

Login to the QICK board via ssh: (password is `xilinx`)

```
ssh xilinx@$IP
```

On the QICK board, run the installation script:

```
sudo chmod +x install.sh
sudo ./install.sh
```

Wait until the installation is complete. The script will ask you to set up a host name.

