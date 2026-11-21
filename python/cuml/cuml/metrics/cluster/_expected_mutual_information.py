#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import cupy as cp
import numpy as np
from cupyx.scipy.special import gammaln

from cuml.internals.validation import check_array

# Each GPU thread sums a segment of at most `_SEGMENT_LENGTH` terms. Longer
# segments are used if there would otherwise be more than `_MAX_SEGMENTS`.
_SEGMENT_LENGTH = 256
_MAX_SEGMENTS = 1 << 20

# Sum a segment of the terms of one pair of marginal counts (a, b)
_emi_kernel = cp.ElementwiseKernel(
    "int64 a, int64 b, int64 offset, int64 segment_length, int64 n_samples, "
    "raw float64 gln_table",
    "float64 emi",
    """
    const double N = (double)n_samples;
    const double log_N = log(N);
    const double log_a = log((double)a);
    const double log_b = log((double)b);
    const double gln_N = lgamma(N + 1.0);
    const double gln_ab = gln_table[a] + gln_table[b]
                          + lgamma(N - a + 1.0) + lgamma(N - b + 1.0);
    const long long first = a + b - n_samples > 1 ? a + b - n_samples : 1;
    const long long last = a < b ? a : b;
    const long long start = first + offset;
    const long long end = start + segment_length < last + 1
                          ? start + segment_length : last + 1;
    emi = 0.0;
    for (long long nij = start; nij < end; ++nij) {
        const double term1 = nij / N;
        const double term2 = log_N + log((double)nij) - log_a - log_b;
        // The probability of nij involves large factorials, use log space
        const double gln = gln_ab - (gln_table[nij] + gln_N)
                           - gln_table[a - nij]
                           - gln_table[b - nij]
                           - lgamma(N - a - b + nij + 1.0);
        emi += term1 * term2 * exp(gln);
    }
    """,
    "cuml_expected_mutual_information",
)


def expected_mutual_information(contingency, n_samples) -> float:
    """
    Calculate the expected mutual information for two labelings.

    This is the mutual information expected between two random labelings that
    have the same cluster sizes as the labelings described by ``contingency``.

    Parameters
    ----------
    contingency : array-like or sparse matrix (device or host) \
            of shape (n_classes_true, n_classes_pred)
        The contingency matrix of the two labelings, i.e. the number of
        samples for each pair of true and predicted labels. Only its row and
        column sums are used.
    n_samples : int
        The number of samples, i.e. the sum of all the elements of
        ``contingency``.

    Returns
    -------
    float
        The expected mutual information, in nats.
    """
    contingency = check_array(
        contingency,
        accept_sparse=True,
        # cupyx.scipy.sparse doesn't support integral dtypes
        dtype=[np.int64, np.float64],
        ensure_min_samples=0,
        ensure_min_features=0,
        input_name="contingency",
    )
    a = cp.ravel(contingency.sum(axis=1)).astype(cp.int64, copy=False)
    b = cp.ravel(contingency.sum(axis=0)).astype(cp.int64, copy=False)

    # Any labeling with zero entropy implies EMI = 0
    if a.size <= 1 or b.size <= 1:
        return 0.0

    # Number of terms (values of nij) for each pair of marginal counts
    n_terms = (
        cp.minimum(a[:, None], b[None, :])
        - cp.maximum(1, a[:, None] + b[None, :] - n_samples)
        + 1
    )
    n_terms = cp.maximum(n_terms, 0).ravel()

    # Split the terms of each pair into equal segments, one per GPU thread
    max_length = max(_SEGMENT_LENGTH, -(-int(n_terms.sum()) // _MAX_SEGMENTS))
    n_segments = (n_terms + max_length - 1) // max_length
    last_segment = cp.cumsum(n_segments)
    segment = cp.arange(int(n_segments.sum()))
    pair = cp.searchsorted(last_segment, segment, side="right")
    length = (n_terms[pair] + n_segments[pair] - 1) // n_segments[pair]
    offset = (segment - (last_segment - n_segments)[pair]) * length

    # Sort by length so that neighboring threads have similar workloads
    order = cp.argsort(length)
    pair, offset, length = pair[order], offset[order], length[order]

    # Lookup table, gln_table[k] = log(k!)
    gln_table = gammaln(
        cp.arange(1, max(int(a.max()), int(b.max())) + 2, dtype=cp.float64)
    )

    emi = _emi_kernel(
        a[pair // b.size],
        b[pair % b.size],
        offset,
        length,
        n_samples,
        gln_table,
    )
    return float(emi.sum())
