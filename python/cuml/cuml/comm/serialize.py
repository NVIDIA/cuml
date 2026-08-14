#
# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import cudf.comm.serialize  # noqa: F401

import cuml

try:
    from distributed.protocol import (
        dask_deserialize,
        dask_serialize,
        register_generic,
    )
    from distributed.protocol.cuda import cuda_deserialize, cuda_serialize
    from distributed.protocol.serialize import pickle_dumps, pickle_loads

    from cuml.ensemble import RandomForestClassifier, RandomForestRegressor

    # These classes require pickling instead of automatically traversing
    # __dict__. They contain non-serializable internal attributes, and their
    # representation is better serialized via pickle anyway.
    pickle_only_classes = (RandomForestRegressor, RandomForestClassifier)

    for name, dumps, loads in [
        ("cuda", cuda_serialize, cuda_deserialize),
        ("dask", dask_serialize, dask_deserialize),
    ]:
        # Generic implementation for `cuml.Base` estimators
        register_generic(cuml.Base, name, dumps, loads)

        # Overrides for pickle-only classes
        dumps.register(pickle_only_classes, pickle_dumps)
        loads.register(pickle_only_classes, pickle_loads)

except ImportError:
    pass
