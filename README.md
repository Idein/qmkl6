# QMKL6

QMKL6 (VideoCore VI QPU Math Kernel Library) is a
[BLAS](https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms) library
that runs on VideoCore VI QPU, the GPU (Graphic Processing Unit) of
[Raspberry Pi](https://www.raspberrypi.org/products) 4.
QMKL6 implements the same C interfaces (e.g. `cblas_sgemm`) as the other BLAS
libraries, so you can change your program to use QMKL6 only by modifying
compiler flags.


## Requirements

QMKL6 currently only supports Raspberry Pi 4 because it is the only Raspberry Pi
board equipped with VideoCore VI QPU.
If you need QPU-accelerated BLAS functions on older Raspberry Pi boards, use
[QMKL](https://github.com/Idein/qmkl) instead.

We recommend to use the official
[Raspberry Pi OS](https://www.raspberrypi.org/downloads) (formerly called
Raspbian) distribution.
However, QMKL6 does not work correctly with the current 5.4.y kernel because the
patchset for internal API changes is not merged yet (see
[raspberrypi/linux#3816](https://github.com/raspberrypi/linux/pull/3816)).
For that reason, you need to check your kernel version by running `uname -r`.
If it shows 5.4.y or higher, you need to downgrade the kernel to 4.19.y (no
workaround exists for aarch64):


```console
$ uname -r
5.4.51-v7l+
$ wget http://archive.raspberrypi.org/debian/pool/main/r/raspberrypi-firmware/raspberrypi-kernel_1.20200601-1_armhf.deb
$ sudo dpkg -i raspberrypi-kernel_1.20200601-1_armhf.deb
$ sudo reboot
```

To build and run QMKL6, you need to install
[py-videocore6](https://github.com/Idein/py-videocore6) and
[libdrm_v3d](https://github.com/Idein/libdrm_v3d) in advance.

QMKL6 communicates with the QPU through `/dev/dri/card0`, which is exposed by
the V3D DRM kernel driver.
To access the device, you need to belong to the `video` group by running
`sudo usermod --append --groups video $USER` (re-login to take effect).


## Running tests

```console
$ sudo apt update
$ sudo apt install build-essential git cmake
$ git clone https://github.com/Idein/qmkl6
$ cd qmkl6/
$ cmake .
$ make
$ ctest -V
```


## Installation

```console
$ make package
$ sudo dpkg -i qmkl6-0.0.0-Linux.deb
```

After that, you can obtain the compiler flags by running
`pkg-config --cflags --libs blas-qmkl6`.
