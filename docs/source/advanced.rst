Advanced Topics
===============

Here we cover a few assorted topics that may be of interest to more advanced
use cases.

CUDA Streams and Synchronization
--------------------------------

Functions and methods in cuML are written using a variety of technologies. As
such, while *most* methods run on the CUDA `per-thread default stream`_ (PTDS),
some methods might run on the `legacy default stream`_ (also known as the NULL
stream) instead.

cuML does not currently expose stream selection as part of its public API and
makes no guarantees on whether a particular method runs on the PTDS or legacy
default stream. Likewise there is no guarantee that the output of a cuML method
or function has been synchronized before returning.

For users, if you follow the following guideline you shouldn't have any
concurrency issues:

- Device memory input arrays should be either fully computed, or currently
  computing on the PTDS or legacy default stream.

- Device memory output arrays should be operated on using either the PTDS or
  legacy default stream, OR have the PTDS of the thread that ran the method
  synchronized before further access.

- Inputs and outputs using host memory have no restrictions and shouldn't be
  prone to concurrency issues.


Selecting the CUDA Device
-------------------------

All single-GPU cuML methods run on device 0 by default. Setting a device via
the :class:`cupy.cuda.Device` or :class:`cuda.core.Device` APIs is currently
not supported. To specify a device to run on, we recommend using the
``CUDA_VISIBLE_DEVICES`` (`doc
<https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html#cuda-visible-devices>`_)
environment variable. For example:

.. code-block:: shell

    CUDA_VISIBLE_DEVICES=2 python myscript.py

cuML does contain a few single-node multi-GPU implementations. When available,
these take a ``device_ids`` parameter to specify which devices to run on. See
the `cuml.manifold.UMAP` docs for an example.


Configuring the Memory Allocator
--------------------------------

Memory allocations in cuML are made using the `Rapids Memory Manager`_ (RMM).
We don't do any configuration of RMM on import; allocations are made using the
default memory resource (:class:`rmm.mr.CudaMemoryResource`).

Some applications may run better using an alternative memory resource. A few
common options:

- A good default to try is the :class:`rmm.mr.CudaAsyncMemoryResource`. This is
  a stream-ordered pooling resource, and may be faster for your application.

  .. code-block:: python

    import rmm

    rmm.mr.set_current_device_resource(rmm.mr.CudaAsyncMemoryResource())


- Users working with large data may want to enable cuML to use `CUDA Unified
  Memory`_ to enable GPU memory oversubscription. To do this, we recommend
  using :class:`rmm.mr.ManagedMemoryResource` wrapped in a
  :class:`rmm.mr.PrefetchResourceAdaptor` to minimize paging overhead.

  .. code-block:: python

    import rmm

    rmm.mr.set_current_device_resource(
        rmm.mr.PrefetchResourceAdaptor(rmm.mr.ManagedMemoryResource())
    )


For more details, see the `RMM documentation`_.


.. _per-thread default stream: https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html#per-thread-default-stream
.. _legacy default stream: https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html#legacy-default-stream
.. _Rapids Memory Manager:
.. Keep this version-pinned until stable subpath redirects are available:
   https://github.com/rapidsai/build-infra/issues/395
.. _RMM documentation: https://docs.nvidia.com/rmm/26.10/
.. _CUDA Unified Memory: https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html
