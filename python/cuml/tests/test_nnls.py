#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Tests for cuml.solvers.nnls / nnls_batched.

Both the single-problem ``nnls`` wrapper and the masked, shared-``A`` batched
``nnls_batched`` primitive are exercised against ``scipy.optimize.nnls`` on
three accuracy facets:

* the residual 2-norm is no worse than scipy's (within a small slack),
* the KKT residual ``max_j |min(x_j, g_j)|`` is small relative to the scale of
  ``A^T b``, and
* every coefficient is non-negative and masked-out coordinates are zero.

Only the Lawson-Hanson backend is currently available; the ``solver`` argument
is retained on the public API but ``"lawson"`` is the sole valid value.

Dimensions cover the sizes seen in Mutation Signature Analysis (the primary
consumer) as well as larger ``n`` and larger batch counts.  The heavier shapes
are guarded behind cuML's ``quality_param`` / ``stress_param`` tiers so the
default unit run stays fast.
"""

import cupy as cp
import numpy as np
import pytest
from scipy.optimize import nnls as scipy_nnls

from cuml.solvers import nnls as cuml_nnls
from cuml.solvers import nnls_batched as cuml_nnls_batched
from cuml.testing.utils import quality_param, stress_param, unit_param


def _lawson_kw(dtype):
    """Solver kwargs for the Lawson backend (max_iter=0 -> active-set cap)."""
    return dict(maxiter=0, tol=(1e-4 if dtype == np.float32 else 1e-8))


# ---------------------------------------------------------------------------
# Single-problem NNLS (cuml.solvers.nnls)
# ---------------------------------------------------------------------------


def _kkt_residual(A, x, b):
    """``max_j |min(x_j, g_j)|`` for the smooth NNLS objective."""
    g = A.T @ (A @ x - b)
    return float(np.max(np.abs(np.minimum(x, g))))


def _kkt_scale(A, b):
    return max(1.0, float(np.max(np.abs(A.T @ b))))


def _check_solution(A, x, b, *, dtype, residual_slack=1.5e-3, kkt_rel=1e-2):
    assert x.shape == (A.shape[1],)
    assert np.all(x >= -1e-6), "negative entries detected"
    x = np.maximum(x, 0.0)

    _, rnorm_ref = scipy_nnls(A.astype(np.float64), b.astype(np.float64))
    rnorm = float(np.linalg.norm(A @ x - b))

    # The residual should match scipy's to within float-precision slack, scaled
    # by ||b|| so the bound stays meaningful for under-determined problems where
    # rnorm_ref == 0 and the achievable residual is dominated by round-off.
    b_scale = max(1.0, float(np.linalg.norm(b)))
    abs_slack = (1e-3 if dtype == np.float32 else 1e-6) * b_scale
    assert rnorm <= rnorm_ref * (1.0 + residual_slack) + abs_slack, (
        f"residual {rnorm:.6g} > scipy residual {rnorm_ref:.6g} "
        f"(slack {residual_slack:.1%} + {abs_slack:.1e})"
    )

    kkt = _kkt_residual(
        A.astype(np.float64), x.astype(np.float64), b.astype(np.float64)
    )
    scale = _kkt_scale(A.astype(np.float64), b.astype(np.float64))
    assert kkt <= kkt_rel * scale, (
        f"KKT residual {kkt:.3e} > {kkt_rel:.1e} * scale {scale:.3e}"
    )


def _make_tall(n_rows, n_cols, *, seed, dtype, sparsity=0.5):
    """Random tall problem with a known non-negative ground truth."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n_rows, n_cols)).astype(dtype)
    x_true = rng.uniform(0.0, 2.0, n_cols).astype(dtype)
    mask = rng.random(n_cols) < sparsity
    x_true[mask] = 0.0
    noise = (0.01 if dtype == np.float64 else 0.05) * rng.standard_normal(
        n_rows
    ).astype(dtype)
    b = (A @ x_true + noise).astype(dtype)
    return A, b


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "n_rows,n_cols",
    [
        unit_param(200, 50),  # mildly tall
        unit_param(96, 40),  # MSA-ish
        quality_param(2000, 200),  # tall
        quality_param(200, 200),  # square
        stress_param(4000, 400),  # large n
    ],
)
def test_nnls_random_dense(dtype, n_rows, n_cols):
    A, b = _make_tall(n_rows, n_cols, seed=0, dtype=dtype)

    x, rnorm = cuml_nnls(A, b, **_lawson_kw(dtype))
    x = cp.asnumpy(x)

    _check_solution(A, x, b, dtype=dtype)
    assert rnorm == pytest.approx(
        float(np.linalg.norm(A @ x - b)), rel=1e-3, abs=1e-4
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_nnls_rank_deficient(dtype):
    """A near-duplicated column makes the Gram matrix ill-conditioned; the
    solver should still converge to a valid (possibly non-unique) solution."""
    rng = np.random.default_rng(7)
    n_rows, n_cols = 400, 30
    A = rng.standard_normal((n_rows, n_cols)).astype(dtype)
    A[:, 5] = A[:, 4] * dtype(0.5) + dtype(1e-6) * rng.standard_normal(
        n_rows
    ).astype(dtype)
    x_true = np.maximum(rng.standard_normal(n_cols), 0.0).astype(dtype)
    b = (A @ x_true + dtype(0.01) * rng.standard_normal(n_rows)).astype(dtype)

    x, _ = cuml_nnls(A, b, **_lawson_kw(dtype))
    x = cp.asnumpy(x)

    # Larger residual slack: rank-deficient problems have a continuum of optima.
    _check_solution(A, x, b, dtype=dtype, residual_slack=5e-3, kkt_rel=5e-2)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_nnls_zero_input(dtype):
    """An all-zero ``A`` is degenerate; the solver should return x == 0."""
    A = np.zeros((10, 4), dtype=dtype)
    b = np.ones(10, dtype=dtype)

    x, rnorm = cuml_nnls(A, b)
    x = cp.asnumpy(x)
    assert np.allclose(x, 0.0)
    assert rnorm == pytest.approx(float(np.linalg.norm(b)), rel=1e-5)


def test_nnls_compute_rnorm_false():
    """compute_rnorm=False skips the residual and returns None for it."""
    A, b = _make_tall(128, 20, seed=1, dtype=np.float64)
    x, rnorm = cuml_nnls(A, b, compute_rnorm=False)
    assert rnorm is None
    _check_solution(A, cp.asnumpy(x), b, dtype=np.float64)


def test_nnls_default_solver_is_lawson():
    """The solver argument defaults to (and only accepts) 'lawson'."""
    A, b = _make_tall(64, 12, seed=2, dtype=np.float64)
    x_default, _ = cuml_nnls(A, b)
    x_explicit, _ = cuml_nnls(A, b, solver="lawson")
    np.testing.assert_allclose(
        cp.asnumpy(x_default), cp.asnumpy(x_explicit), rtol=1e-9, atol=1e-9
    )


def test_nnls_unknown_solver():
    A = np.eye(3, dtype=np.float32)
    b = np.ones(3, dtype=np.float32)
    with pytest.raises(ValueError):
        cuml_nnls(A, b, solver="apg")


# ---------------------------------------------------------------------------
# Batched, masked, shared-A NNLS (cuml.solvers.nnls_batched)
# ---------------------------------------------------------------------------


def _msa_reference(A, B, masks, b_index):
    """Per-problem scipy reference matching MSA run_NNLS.nnls_batched."""
    n_cols, n_problems = masks.shape
    out_weights = np.zeros((n_cols, n_problems), dtype=np.float64)
    out_fitted = np.zeros((A.shape[0], n_problems), dtype=np.float64)
    Ad = A.astype(np.float64)
    Bd = B.astype(np.float64)
    for j in range(n_problems):
        cols = np.flatnonzero(masks[:, j])
        if cols.size == 0:
            continue
        w, _ = scipy_nnls(Ad[:, cols], Bd[:, b_index[j]])
        out_weights[cols, j] = w
        out_fitted[:, j] = Ad[:, cols] @ w
    return out_weights, out_fitted


def _check_batched(
    A,
    B,
    masks,
    b_index,
    out_weights,
    out_fitted,
    *,
    dtype,
    residual_slack=5e-3,
    kkt_rel=5e-2,
    ref_subset=None,
):
    """Validate a batched solve.  Non-negativity and masked-out zeros are
    checked on every problem; the (expensive) scipy residual/KKT comparison is
    limited to ``ref_subset`` problem indices when given, keeping large-batch
    runs bounded."""
    n_cols, P = masks.shape
    assert out_weights.shape == (n_cols, P)
    assert np.all(out_weights >= -1e-5), "negative weights detected"

    Ad = A.astype(np.float64)
    Bd = B.astype(np.float64)
    b_scale_all = max(
        1.0, float(np.max(np.linalg.norm(Bd[:, b_index], axis=0)))
    )
    abs_slack = (2e-3 if dtype == np.float32 else 1e-6) * b_scale_all

    if ref_subset is None:
        ref_subset = range(P)
    ref_subset = set(int(j) for j in ref_subset)

    for j in range(P):
        cols = np.flatnonzero(masks[:, j])
        off = np.ones(n_cols, dtype=bool)
        off[cols] = False
        assert np.allclose(out_weights[off, j], 0.0, atol=1e-5), (
            f"problem {j}: masked-out weights are nonzero"
        )
        if j not in ref_subset or cols.size == 0:
            continue

        b_j = Bd[:, b_index[j]]
        w_ref, _ = scipy_nnls(Ad[:, cols], b_j)
        w = out_weights[cols, j].astype(np.float64)
        r = float(np.linalg.norm(Ad[:, cols] @ w - b_j))
        r_ref = float(np.linalg.norm(Ad[:, cols] @ w_ref - b_j))
        assert r <= r_ref * (1.0 + residual_slack) + abs_slack, (
            f"problem {j}: residual {r:.6g} > scipy {r_ref:.6g}"
        )
        g = Ad[:, cols].T @ (Ad[:, cols] @ w - b_j)
        kkt = float(np.max(np.abs(np.minimum(w, g))))
        scale = max(1.0, float(np.max(np.abs(Ad[:, cols].T @ b_j))))
        assert kkt <= kkt_rel * scale, (
            f"problem {j}: KKT {kkt:.3e} > {kkt_rel:.1e} * {scale:.3e}"
        )

    assert out_fitted.shape == (A.shape[0], P)


def _run_batched(A, B, masks, **kw):
    """Call the device-native nnls_batched and mirror (X, fitted) to host."""
    X, fitted = cuml_nnls_batched(A, B, masks, **kw)
    X = cp.asnumpy(X)
    fitted = None if fitted is None else cp.asnumpy(fitted)
    return X, fitted


def _make_batched(
    m, n, P, *, seed, dtype, signature_like=False, mask_prob=None
):
    """Random batched problem with a known non-negative ground truth."""
    rng = np.random.default_rng(seed)
    if signature_like:
        A = np.abs(rng.standard_normal((m, n))).astype(dtype)
    else:
        A = rng.standard_normal((m, n)).astype(dtype)
    x_true = rng.uniform(0.0, 3.0, (n, P)).astype(dtype)
    x_true[rng.random((n, P)) < 0.5] = 0.0
    B = (A @ x_true + dtype(0.01) * rng.standard_normal((m, P))).astype(dtype)
    if mask_prob is None:
        masks = x_true > 0.0
    else:
        masks = rng.random((n, P)) < mask_prob
    # Guarantee every problem keeps at least a few active columns.
    for p in range(P):
        if masks[:, p].sum() < 3:
            masks[rng.choice(n, min(3, n), replace=False), p] = True
    b_index = np.arange(P)
    return A, B, masks, b_index


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("P", [1, 4, 64])
def test_batched_random_dense(dtype, P):
    A, B, masks, b_index = _make_batched(128, 24, P, seed=P, dtype=dtype)
    masks = np.ones((24, P), dtype=bool)  # unmasked full support

    out_weights, out_fitted = _run_batched(
        A, B, masks, b_index=b_index, **_lawson_kw(dtype)
    )
    _check_batched(A, B, masks, b_index, out_weights, out_fitted, dtype=dtype)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_batched_masked(dtype):
    A, B, masks, b_index = _make_batched(
        96, 20, 16, seed=123, dtype=dtype, mask_prob=0.6
    )

    out_weights, out_fitted = _run_batched(
        A, B, masks, b_index=b_index, **_lawson_kw(dtype)
    )
    _check_batched(A, B, masks, b_index, out_weights, out_fitted, dtype=dtype)


# MSA mutation-signature dimensions (m rows = mutation contexts, n = signatures).
_MSA_SHAPES = [
    unit_param(96, 65),  # SBS-96 default
    unit_param(78, 11),  # DBS
    unit_param(83, 17),  # ID
    quality_param(192, 54),  # SBS-192
    quality_param(288, 10),  # SBS-288
    quality_param(1536, 10),  # SBS-1536 (Gram setup dominates)
    stress_param(4608, 10),  # SBS-4608
]


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("m,n", _MSA_SHAPES)
def test_batched_msa_shapes(dtype, m, n):
    """Shapes mimicking a mutation-signature-attribution round."""
    P = 128
    A, B, masks, b_index = _make_batched(
        m, n, P, seed=(m * 131 + n), dtype=dtype, signature_like=True
    )

    out_weights, out_fitted = _run_batched(
        A, B, masks, b_index=b_index, **_lawson_kw(dtype)
    )
    _check_batched(A, B, masks, b_index, out_weights, out_fitted, dtype=dtype)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "n", [unit_param(64), quality_param(128), stress_param(256)]
)
def test_batched_larger_n(dtype, n):
    """Larger n exercises the multi-panel Cholesky and back-solve paths."""
    m, P = max(2 * n, 128), 32
    A, B, masks, b_index = _make_batched(
        m, n, P, seed=n, dtype=dtype, mask_prob=0.7
    )

    out_weights, out_fitted = _run_batched(
        A, B, masks, b_index=b_index, **_lawson_kw(dtype)
    )
    _check_batched(A, B, masks, b_index, out_weights, out_fitted, dtype=dtype)


def test_batched_b_index_gather():
    """Leave-one-out style: many problems share a few distinct B columns."""
    dtype = np.float64
    rng = np.random.default_rng(99)
    m, n = 96, 20
    n_targets, n_problems = 8, 64
    A = np.abs(rng.standard_normal((m, n))).astype(dtype)
    B = np.abs(rng.standard_normal((m, n_targets))).astype(dtype)
    masks = rng.random((n, n_problems)) < 0.7
    for p in range(n_problems):
        if masks[:, p].sum() < 3:
            masks[rng.choice(n, 3, replace=False), p] = True
    b_index = rng.integers(0, n_targets, size=n_problems)

    out_weights, out_fitted = _run_batched(
        A, B, masks, b_index=b_index, **_lawson_kw(dtype)
    )
    _check_batched(A, B, masks, b_index, out_weights, out_fitted, dtype=dtype)


def test_batched_b_index_out_of_range():
    A = np.abs(np.random.default_rng(0).standard_normal((16, 4)))
    B = np.abs(np.random.default_rng(1).standard_normal((16, 3)))
    masks = np.ones((4, 2), dtype=bool)
    with pytest.raises(ValueError):
        cuml_nnls_batched(A, B, masks, b_index=np.array([0, 5]))


def test_batched_b_index_length_mismatch():
    A = np.abs(np.random.default_rng(0).standard_normal((16, 4)))
    B = np.abs(np.random.default_rng(1).standard_normal((16, 3)))
    masks = np.ones((4, 2), dtype=bool)  # 2 problems
    with pytest.raises(ValueError):
        cuml_nnls_batched(A, B, masks, b_index=np.array([0, 1, 2]))


def test_batched_fitted_matches_AX():
    rng = np.random.default_rng(0)
    m, n, P = 64, 12, 8
    A = np.abs(rng.standard_normal((m, n))).astype(np.float64)
    B = np.abs(rng.standard_normal((m, P))).astype(np.float64)
    masks = np.ones((n, P), dtype=bool)

    out_weights, out_fitted = _run_batched(A, B, masks)
    np.testing.assert_allclose(
        out_fitted, A @ out_weights, rtol=1e-6, atol=1e-6
    )


def test_batched_returns_device_arrays():
    """nnls_batched is device-native: it returns cupy arrays, and fitted is
    None when compute_fitted=False (skipping the extra matmul)."""
    rng = np.random.default_rng(3)
    A = np.abs(rng.standard_normal((32, 6))).astype(np.float64)
    B = np.abs(rng.standard_normal((32, 5))).astype(np.float64)
    masks = np.ones((6, 5), dtype=bool)

    X, fitted = cuml_nnls_batched(A, B, masks)
    assert isinstance(X, cp.ndarray)
    assert isinstance(fitted, cp.ndarray)
    assert X.shape == (6, 5)
    assert fitted.shape == (32, 5)

    X2, fitted2 = cuml_nnls_batched(A, B, masks, compute_fitted=False)
    assert isinstance(X2, cp.ndarray)
    assert fitted2 is None
    np.testing.assert_allclose(
        cp.asnumpy(X2), cp.asnumpy(X), rtol=1e-6, atol=1e-6
    )


@pytest.mark.parametrize("order", ["C", "F"])
def test_batched_mask_order_parity(order):
    """The result must be identical whether the mask is C- or F-contiguous.
    F-contiguous is the zero-copy device path; C-contiguous is copied to F
    inside the Cython layer.  Both must produce the same solution."""
    rng = np.random.default_rng(order == "F")
    m, n, P = 96, 20, 32
    A = np.abs(rng.standard_normal((m, n))).astype(np.float64)
    x_true = rng.uniform(0.0, 3.0, (n, P)).astype(np.float64)
    x_true[rng.random((n, P)) < 0.5] = 0.0
    B = (A @ x_true).astype(np.float64)

    masks_bool = rng.random((n, P)) < 0.6
    for p in range(P):
        if masks_bool[:, p].sum() < 3:
            masks_bool[rng.choice(n, 3, replace=False), p] = True

    masks = np.asarray(masks_bool, order=order)
    assert masks.flags["%s_CONTIGUOUS" % order]
    b_index = np.arange(P)

    X, _ = cuml_nnls_batched(
        A, B, masks, b_index=b_index, **_lawson_kw(np.float64)
    )
    other = "C" if order == "F" else "F"
    X_other, _ = cuml_nnls_batched(
        A,
        B,
        np.asarray(masks_bool, order=other),
        b_index=b_index,
        **_lawson_kw(np.float64),
    )
    np.testing.assert_allclose(
        cp.asnumpy(X), cp.asnumpy(X_other), rtol=1e-6, atol=1e-6
    )


def test_batched_all_false_mask_column():
    """A problem with no active columns yields an all-zero solution column."""
    rng = np.random.default_rng(11)
    m, n, P = 48, 10, 6
    A = np.abs(rng.standard_normal((m, n))).astype(np.float64)
    B = np.abs(rng.standard_normal((m, P))).astype(np.float64)
    masks = np.ones((n, P), dtype=bool)
    masks[:, 2] = False  # empty support for problem 2

    X, fitted = _run_batched(A, B, masks)
    assert np.allclose(X[:, 2], 0.0)
    assert np.allclose(fitted[:, 2], 0.0)
    b_index = np.arange(P)
    _check_batched(A, B, masks, b_index, X, fitted, dtype=np.float64)


def test_batched_repeated_calls_resident():
    """Repeated calls that reuse the same device-resident A/B (MSA cache
    pattern, which also re-hits the per-n dispatch plan cache) are stable."""
    rng = np.random.default_rng(2024)
    m, n, P = 96, 24, 40
    A = cp.asarray(np.abs(rng.standard_normal((m, n))).astype(np.float64))
    B = cp.asfortranarray(
        cp.asarray(np.abs(rng.standard_normal((m, P))).astype(np.float64))
    )
    masks = cp.asarray(np.ones((n, P), dtype=np.uint8))

    X0, _ = cuml_nnls_batched(A, B, masks, **_lawson_kw(np.float64))
    for _ in range(3):
        Xi, _ = cuml_nnls_batched(A, B, masks, **_lawson_kw(np.float64))
        np.testing.assert_allclose(
            cp.asnumpy(Xi), cp.asnumpy(X0), rtol=1e-9, atol=1e-9
        )


def test_batched_on_device_l2_scoring_parity():
    """The GPU boundary (MSA batched_solve_and_score) reduces fitted to the
    L2_normalised_by_first similarity on the device.  Verify that the cupy
    reduction matches the numpy reference for the same fitted vectors."""
    rng = np.random.default_rng(2024)
    m, n = 96, 24
    n_targets, P = 10, 48
    A = np.abs(rng.standard_normal((m, n))).astype(np.float64)
    B = np.abs(rng.standard_normal((m, n_targets))).astype(np.float64)
    masks_bool = rng.random((n, P)) < 0.7
    for p in range(P):
        if masks_bool[:, p].sum() < 3:
            masks_bool[rng.choice(n, 3, replace=False), p] = True
    masks = np.asarray(masks_bool, order="F")
    b_index = rng.integers(0, n_targets, size=P)
    norm_obs = np.linalg.norm(B, axis=0)

    A_dev = cp.asarray(A)
    B_dev = cp.asfortranarray(cp.asarray(B))
    bi_dev = cp.asarray(b_index)
    _, fitted = cuml_nnls_batched(
        A_dev,
        B_dev,
        cp.asarray(masks),
        b_index=bi_dev,
        **_lawson_kw(np.float64),
    )
    resid = cp.linalg.norm(B_dev[:, bi_dev] - fitted, axis=0)
    sims_dev = cp.asnumpy(1.0 - resid / cp.asarray(norm_obs)[bi_dev])

    fitted_np = cp.asnumpy(fitted)
    sims_ref = (
        1.0
        - np.linalg.norm(B[:, b_index] - fitted_np, axis=0) / norm_obs[b_index]
    )
    np.testing.assert_allclose(sims_dev, sims_ref, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("P", [quality_param(4096), stress_param(65536)])
def test_batched_large_batch(P):
    """MSA-scale batch counts.  All GPU invariants are checked on every
    problem; the scipy residual/KKT comparison is limited to a deterministic
    subset so the run stays bounded."""
    dtype = np.float32
    m, n = 96, 65
    A, B, masks, b_index = _make_batched(
        m, n, P, seed=7, dtype=dtype, signature_like=True, mask_prob=0.6
    )

    out_weights, out_fitted = _run_batched(
        A, B, masks, b_index=b_index, **_lawson_kw(dtype)
    )

    subset = np.linspace(0, P - 1, 64, dtype=int)
    _check_batched(
        A,
        B,
        masks,
        b_index,
        out_weights,
        out_fitted,
        dtype=dtype,
        ref_subset=subset,
    )


def test_batched_unknown_solver():
    A = np.eye(4, dtype=np.float32)
    B = np.ones((4, 2), dtype=np.float32)
    masks = np.ones((4, 2), dtype=bool)
    with pytest.raises(ValueError):
        cuml_nnls_batched(A, B, masks, solver="apg")


def test_batched_bad_mask_shape():
    A = np.eye(4, dtype=np.float32)
    B = np.ones((4, 2), dtype=np.float32)
    with pytest.raises(ValueError):
        cuml_nnls_batched(A, B, np.ones((3, 4), dtype=bool))
