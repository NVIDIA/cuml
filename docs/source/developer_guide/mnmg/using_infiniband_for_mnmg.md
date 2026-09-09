> [!WARNING]
> Instructions on this page are deprecated and will not work with the latest version of cuML.

# Using Infiniband for Multi-Node Multi-GPU cuML

These instructions outline how to run multi-node multi-GPU cuML on devices with Infiniband. These instructions assume the necessary Infiniband hardware has already been installed and the relevant software has already been configured to enable communication over the Infiniband devices.

The steps in this wiki post have been largely adapted from the [Experiments in High Performance Networking with UCX and DGX](https://blog.dask.org/2019/06/09/ucx-dgx) blog by Matthew Rocklin and Rick Zamora.

## 1. Install UCX

### From Conda

Note: this package is experimental and will eventually be supported under the rapidsai channel. Currently, it requires CUDA9.2 but a CUDA10 package is also in the works.

`conda install -c conda-forge -c jakirkham/label/ucx cudatoolkit=9.2 ucx-proc=*=gpu ucx python=3.7`

### From Source

Install autogen if it's not already installed:
```bash
sudo apt-get install autogen autoconf libtool
```

Optionally install `gdrcopy` for faster GPU-Network card data transfer:

From the [ucx wiki](https://github.com/openucx/ucx/wiki/NVIDIA-GPU-Support), `gdrcopy` can be installed, and might be necessary, to enable faster GPU-Network card data transfer.

Here are the install instructions, taken from [gdrcopy github](https://github.com/NVIDIA/gdrcopy)
```bash
git clone https://github.com/NVIDIA/gdrcopy.git
cd gdrcopy
make -j PREFIX=$CONDA_INSTALL_PREFIX CUDA=/usr/local/cuda && make -j install
sudo ./insmod.sh
```


```bash
git clone https://github.com/cjnolet/ucx-py.git
cd ucx
git checkout fea-ext-expose_worker_and_ep
./autogen.sh
mkdir build && cd build
../configure --prefix=$CONDA_PREFIX --with-cuda=/usr/local/cuda --enable-mt --disable-cma CPPFLAGS="-I//usr/local/cuda/include"
make -j install
```

Note: If you have installed `gdrcopy`, you can add `--with-gdrcopy=/path/to/gdrcopy` to the options in `configure`

Verify with `ucx_info -d`. You should expect to see line(s) with the `rc` transport:

```
#   Transport: rc
#
#   Device: mlx5_0:1
#
#      capabilities:
#            bandwidth: 11794.23 MB/sec
#              latency: 600 nsec + 1 * N
#             overhead: 75 nsec
#            put_short: <= 124
#            put_bcopy: <= 8K
#            put_zcopy: <= 1G, up to 8 iov
#  put_opt_zcopy_align: <= 512
#        put_align_mtu: <= 4K
#            get_bcopy: <= 8K
#            get_zcopy: 65..1G, up to 8 iov
#  get_opt_zcopy_align: <= 512
#        get_align_mtu: <= 4K
#             am_short: <= 123
#             am_bcopy: <= 8191
#             am_zcopy: <= 8191, up to 7 iov
#   am_opt_zcopy_align: <= 512
#         am_align_mtu: <= 4K
#            am header: <= 127
#               domain: device
#           connection: to ep
#             priority: 30
#       device address: 3 bytes
#           ep address: 4 bytes
#       error handling: peer failure

```

You should also expect to see lines with `cuda_copy` and `cuda_ipc` transports:

```
#   Transport: cuda_copy
#
#   Device: cudacopy0
#
#      capabilities:
#            bandwidth: 6911.00 MB/sec
#              latency: 10000 nsec
#             overhead: 0 nsec
#            put_short: <= 4294967295
#            put_zcopy: unlimited, up to 1 iov
#  put_opt_zcopy_align: <= 1
#        put_align_mtu: <= 1
#            get_short: <= 4294967295
#            get_zcopy: unlimited, up to 1 iov
#  get_opt_zcopy_align: <= 1
#        get_align_mtu: <= 1
#           connection: to iface
#             priority: 0
#       device address: 0 bytes
#        iface address: 8 bytes
#       error handling: none
```

```
# Memory domain: cuda_ipc
#            component: cuda_ipc
#             register: <= 1G, cost: 0 nsec
#           remote key: 104 bytes
#
#   Transport: cuda_ipc
#
#   Device: cudaipc0
#
#      capabilities:
#            bandwidth: 24000.00 MB/sec
#              latency: 1 nsec
#             overhead: 0 nsec
#            put_zcopy: <= 1G, up to 1 iov
#  put_opt_zcopy_align: <= 1
#        put_align_mtu: <= 1
#            get_zcopy: <= 1G, up to 1 iov
#  get_opt_zcopy_align: <= 1
#        get_align_mtu: <= 1
#           connection: to iface
#             priority: 0
#       device address: 8 bytes
#        iface address: 4 bytes
#       error handling: none
#

```


If you configured UCX with the `gdrcopy` option, you should also expect to see transports in this list:

```bash
# Memory domain: gdr_copy
#            component: gdr_copy
#             register: unlimited, cost: 0 nsec
#           remote key: 32 bytes
#
#   Transport: gdr_copy
#
#   Device: gdrcopy0
#
#      capabilities:
#            bandwidth: 6911.00 MB/sec
#              latency: 1000 nsec
#             overhead: 0 nsec
#            put_short: <= 4294967295
#            get_short: <= 4294967295
#           connection: to iface
#             priority: 0
#       device address: 0 bytes
#        iface address: 8 bytes
#       error handling: none
```

To better understand the CUDA-based transports in UCX, refer to [this wiki](https://github.com/openucx/ucx/wiki/NVIDIA-GPU-Support) for more details.


## 2. Install ucx-py

### From Conda

Note: this package is experimental and will eventually be supported under the rapidsai channel. Currently, it requires CUDA9.2 but a CUDA10 package is also in the works.

`conda install -c conda-forge -c jakirkham/label/ucx cudatoolkit=9.2 ucx-py python=3.7`


### From Source

```bash
git clone git@github.com:rapidsai/ucx-py
cd ucx-py

export UCX_PATH=$CONDA_PREFIX
make -j install
```

## 3. Install NCCL

It's important that NCCL 2.4+ be installed and no previous versions of NCCL are conflicting on your library path. This will cause compile errors during the build of cuML.


```bash
conda install -c nvidia nccl
```

Create the file `.nccl.conf` in your home dir with the following:
```bash
NCCL_SOCKET_IFNAME=ib0
```

## 4. Enable IP over IB interface at ib0

Follow the instructions at [this link](https://docs.oracle.com/cd/E19436-01/820-3522-10/ch4-linux.html#50536461_82843) to create an IP interface for the IB devices.

From the link above, when the IP over IB kernel module has already been installed, mapping to an IP interface is simple:
```
sudo ifconfig ib0 10.0.0.50/24
```

You can verify the interface was created properly with `ifconfig ib0`

The output should look like this:

```
ib0       Link encap:UNSPEC  HWaddr 80-00-00-68-FE-80-00-00-00-00-00-00-00-00-00-00
          inet addr:10.0.0.50  Bcast:10.0.0.255  Mask:255.255.255.0
          inet6 addr: fe80::526b:4b03:f5:ce9c/64 Scope:Link
          UP BROADCAST RUNNING MULTICAST  MTU:65520  Metric:1
          RX packets:2655 errors:0 dropped:0 overruns:0 frame:0
          TX packets:2697 errors:0 dropped:10 overruns:0 carrier:0
          collisions:0 txqueuelen:256
          RX bytes:183152 (183.1 KB)  TX bytes:194696 (194.6 KB)

```

## 5.  Set UCX environment vars

Use `ibstatus` to see your open IB devices. Output will look like this:

```
Infiniband device 'mlx5_0' port 1 status:
	default gid:	 fe80:0000:0000:0000:506b:4b03:00f5:ce9c
	base lid:	 0xf
	sm lid:		 0x1
	state:		 4: ACTIVE
	phys state:	 5: LinkUp
	rate:		 100 Gb/sec (4X EDR)
	link_layer:	 InfiniBand

Infiniband device 'mlx5_1' port 1 status:
	default gid:	 fe80:0000:0000:0000:506b:4b03:0049:4236
	base lid:	 0x6
	sm lid:		 0x1
	state:		 4: ACTIVE
	phys state:	 5: LinkUp
	rate:		 100 Gb/sec (4X EDR)
	link_layer:	 InfiniBand

Infiniband device 'mlx5_2' port 1 status:
	default gid:	 fe80:0000:0000:0000:506b:4b03:00f5:cf04
	base lid:	 0x2
	sm lid:		 0x1
	state:		 4: ACTIVE
	phys state:	 5: LinkUp
	rate:		 100 Gb/sec (4X EDR)
	link_layer:	 InfiniBand

Infiniband device 'mlx5_3' port 1 status:
	default gid:	 fe80:0000:0000:0000:506b:4b03:0049:3eb2
	base lid:	 0x11
	sm lid:		 0x1
	state:		 4: ACTIVE
	phys state:	 5: LinkUp
	rate:		 100 Gb/sec (4X EDR)
	link_layer:	 InfiniBand

```

Put the devices and ports in a `UCX_NET_DEVICES` environment variable:


```bash
export UCX_NET_DEVICES=mlx5_0:1,mlx5_3:1,mlx5_2:1,mlx5_1:1
```

Set transports for UCX to use:
```bash
export UCX_TLS=rc,cuda_copy,cuda_ipc
```

Note: if `gdrcopy` was installed, add `gdr_copy` to the end of `UCX_TLS`

## 6. Start Dask cluster on ib0 interface:

Run this on the node designated for the scheduler:
```bash
dask-scheduler --protocol ucx --interface ib0
```

Then run this on each worker (for example, if the IP over IB device address running the scheduler is `10.0.0.50`):
```bash
dask-cuda-worker ucx://10.0.0.50:8786
```

## 7. Run cumlCommunicator test:

### First, create a Dask `Client` and cuML `Comms`:
```python
from dask.distributed import Client, wait
from cuml.raft.dask.common.comms import Comms
from cuml.dask.common import get_raft_comm_state
from cuml.dask.common import perform_test_comms_send_recv
from cuml.dask.common import perform_test_comms_allreduce

import random

c = Client("ucx://10.0.0.50:8786")
cb = Comms(comms_p2p=True)
cb.init()
```

### Test Point-to-Point Communications:
```python
n_trials = 2

def func_test_send_recv(sessionId, n_trials, r):
    handle = get_raft_comm_state(sessionId)["handle"]
    return perform_test_comms_send_recv(handle, n_trials)

p2p_dfs=[c.submit(func_test_send_recv, cb.sessionId, n_trials, random.random(), workers=[w]) for wid, w in zip(range(len(cb.worker_addresses)), cb.worker_addresses)]
wait(p2p_dfs)

p2p_result = list(map(lambda x: x.result(), p2p_dfs))
print(str(p2p_result))

assert all(p2p_result)
```

You should see the following output on your workers:
```

=========================
Trial 0
Rank 0 received: [1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 8, 9, 14, 15]
Rank 1 received: [0, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 8, 9, 14, 15]
Rank 2 received: [0, 1, 3, 4, 5, 6, 7, 10, 11, 12, 13, 8, 9, 14, 15]
Rank 3 received: [0, 1, 2, 4, 5, 6, 7, 10, 11, 12, 13, 8, 9, 14, 15]
Rank 4 received: [0, 1, 2, 3, 5, 6, 7, 10, 11, 12, 13, 8, 9, 14, 15]
Rank 5 received: [0, 1, 2, 3, 4, 6, 7, 10, 11, 12, 13, 8, 9, 14, 15]
Rank 6 received: [0, 1, 2, 3, 4, 5, 7, 10, 11, 12, 13, 8, 9, 14, 15]
Rank 7 received: [0, 1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 8, 9, 14, 15]
=========================
=========================
Trial 1
Rank 0 received: [11, 2, 13, 12, 9, 10, 15, 14, 1, 8, 5, 4, 3, 6, 7]
Rank 1 received: [2, 12, 11, 10, 9, 14, 13, 8, 15, 4, 5, 6, 3, 0, 7]
Rank 2 received: [12, 1, 11, 10, 9, 14, 13, 8, 15, 4, 5, 6, 3, 0, 7]
Rank 3 received: [2, 11, 12, 10, 9, 14, 13, 8, 15, 4, 1, 6, 5, 0, 7]
Rank 4 received: [2, 11, 12, 9, 13, 10, 15, 14, 1, 8, 3, 6, 5, 0, 7]
Rank 5 received: [2, 11, 12, 9, 10, 14, 13, 8, 15, 4, 1, 6, 3, 0, 7]
Rank 6 received: [2, 11, 12, 9, 10, 13, 15, 14, 1, 8, 5, 4, 3, 0, 7]
Rank 7 received: [2, 11, 12, 9, 10, 13, 14, 8, 15, 4, 1, 6, 5, 0, 3]
=========================

```

### Test collective communications:
```python
def func_test_allreduce(sessionId, r):
    handle = get_raft_comm_state(sessionId)["handle"]
    return perform_test_comms_allreduce(handle)

coll_dfs = [c.submit(func_test_allreduce, cb.sessionId, random.random(), workers=[w]) for wid, w in zip(range(len(cb.worker_addresses)), cb.worker_addresses)]
wait(coll_dfs)

coll_result = list(map(lambda x: x.result(), coll_dfs))

coll_result

assert all(coll_result)
```

You should see the following output on your workers:
```
Clique size: 16
Clique size: 16
Clique size: 16
Clique size: 16
Clique size: 16
Clique size: 16
final_size: 16
Clique size: 16
Clique size: 16
final_size: 16
final_size: 16
final_size: 16
final_size: 16
final_size: 16
final_size: 16
final_size: 16
```
