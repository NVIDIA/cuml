#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

source rapids-init-pip

source ./ci/use_wheels_from_prs.sh
LIBCUML_WHEELHOUSE=$(rapids-download-from-github "$(rapids-artifact-name wheel_cpp libcuml cuml --cuda "$RAPIDS_CUDA_VERSION")")
CUML_WHEELHOUSE=$(rapids-download-from-github "$(rapids-artifact-name wheel_python cuml cuml --stable --cuda "$RAPIDS_CUDA_VERSION")")

EXITCODE=0
trap "EXITCODE=1" ERR
set +e

#
# BERTopic Integration Test
#
rapids-logger "===== Testing BERTopic Integration ====="

rapids-logger "Generating testing dependencies"
rapids-dependency-file-generator \
  --output requirements \
  --file-key test_integration_bertopic \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION};dependencies=${RAPIDS_DEPENDENCIES}" \
| tee ./requirements.txt

rapids-logger "Installing cuML, BERTopic, and dependencies"
rapids-pip-retry install \
  --prefer-binary \
  "${LIBCUML_WHEELHOUSE}"/libcuml*.whl \
  "${CUML_WHEELHOUSE}"/cuml*.whl \
  -r ./requirements.txt

# Test 1: Verify imports
rapids-logger "Testing imports"
python -c "
import cuml
import bertopic
print('✓ Import test passed')
"

# Test 2: Run minimal end-to-end example
rapids-logger "Running BERTopic end-to-end smoke test"
timeout -v 20m python -c "
import warnings
warnings.filterwarnings('ignore')

import random
from bertopic import BERTopic

from cuml.cluster import HDBSCAN
from cuml.manifold import UMAP

# Generate synthetic documents with topic-like word clusters
random.seed(42)
topics = [
    ['star', 'galaxy', 'planet', 'orbit', 'telescope', 'nasa', 'astronaut'],
    ['rocket', 'launch', 'satellite', 'mission', 'space', 'shuttle', 'station'],
    ['moon', 'mars', 'jupiter', 'asteroid', 'comet', 'meteor', 'crater'],
]

docs = []
for i in range(100):
    topic_words = topics[i % len(topics)]
    doc = ' '.join(random.choices(topic_words, k=random.randint(10, 30)))
    docs.append(doc)

hdbscan_model = HDBSCAN(min_samples=10, gen_min_span_tree=True, prediction_data=True)
umap_model = UMAP(n_components=5, n_neighbors=15, min_dist=0.0)

topic_model = BERTopic(verbose=False,
                       calculate_probabilities=False,
                       hdbscan_model=hdbscan_model,
                       umap_model=umap_model)

topics, probs = topic_model.fit_transform(docs)

print(f'✓ BERTopic smoke test passed - processed {len(docs)} documents, found {len(set(topics))} topics')
"

rapids-logger "===== BERTopic Integration Test Complete ====="

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
