import numpy as np
import pytest


def _random_psd(rng, batch, n, rank=None):
    rank = rank or n
    G = rng.standard_normal(batch + (n, rank))
    return G @ np.swapaxes(G, -1, -2)


# ---------------------------------------------------------------- row_norms

@pytest.mark.parametrize("ord", [1, 2, np.inf])
def test_row_norms(sol, ord, no_loops):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((6, 4))
    out = sol.row_norms(X, ord)
    assert out.shape == (6,), f"expected one norm per row, shape (6,), got {out.shape}"
    expected = [np.linalg.norm(r, ord) for r in X]
    np.testing.assert_allclose(out, expected, err_msg=f"wrong row norms for ord={ord}")
    no_loops(sol.row_norms)


def test_row_norms_example(sol):
    X = np.array([[3.0, 4.0], [1.0, -1.0]])
    np.testing.assert_allclose(sol.row_norms(X, 2), [5.0, np.sqrt(2)])
    np.testing.assert_allclose(sol.row_norms(X, 1), [7.0, 2.0])
    np.testing.assert_allclose(sol.row_norms(X, np.inf), [4.0, 1.0])


# ---------------------------------------------------------------- low_rank_approx

def test_low_rank_approx_diag(sol, no_loops):
    out = sol.low_rank_approx(np.diag([3.0, 2.0, 1.0]), 2)
    np.testing.assert_allclose(out, np.diag([3.0, 2.0, 0.0]), atol=1e-12)
    no_loops(sol.low_rank_approx)


def test_low_rank_approx_properties(sol):
    rng = np.random.default_rng(0)
    A = rng.standard_normal((8, 5))
    out = sol.low_rank_approx(A, 2)
    assert out.shape == A.shape
    assert np.linalg.matrix_rank(out) == 2, "the result should have rank k"
    s = np.linalg.svd(A, compute_uv=False)
    err = np.linalg.norm(A - out)
    assert err == pytest.approx(np.sqrt((s[2:] ** 2).sum())), (
        "the Frobenius error of the best rank-k approximation is sqrt(sum of the dropped s**2)"
    )
    np.testing.assert_allclose(sol.low_rank_approx(A, 5), A, atol=1e-12, err_msg="k = min(m, n) should give A back")


def test_low_rank_approx_batched(sol):
    rng = np.random.default_rng(1)
    A = rng.standard_normal((4, 6, 3))
    out = sol.low_rank_approx(A, 1)
    assert out.shape == (4, 6, 3), f"batch shape should be kept, got {out.shape}"
    for b in range(4):
        U, s, Vh = np.linalg.svd(A[b], full_matrices=False)
        np.testing.assert_allclose(out[b], s[0] * np.outer(U[:, 0], Vh[0]), atol=1e-12,
                                   err_msg=f"matrix {b} of the batch is wrong")


# ---------------------------------------------------------------- top_eigenpair

def test_top_eigenpair_example(sol, no_loops):
    vals, vecs = sol.top_eigenpair(np.array([[[2.0, 0.0], [0.0, 5.0]]]))
    np.testing.assert_allclose(vals, [5.0], err_msg="eigh sorts ascending: the largest is the LAST one")
    np.testing.assert_allclose(np.abs(vecs), [[0.0, 1.0]], atol=1e-12)
    no_loops(sol.top_eigenpair)


def test_top_eigenpair_random(sol):
    rng = np.random.default_rng(0)
    S = _random_psd(rng, (5,), 4) - 2 * np.eye(4)  # symmetric, mixed-sign eigenvalues
    vals, vecs = sol.top_eigenpair(S)
    assert vals.shape == (5,) and vecs.shape == (5, 4), f"got shapes {vals.shape}, {vecs.shape}"
    for b in range(5):
        np.testing.assert_allclose(vals[b], np.linalg.eigvalsh(S[b]).max())
        np.testing.assert_allclose(S[b] @ vecs[b], vals[b] * vecs[b], atol=1e-10,
                                   err_msg=f"vectors[{b}] is not an eigenvector (rows vs columns of V?)")
        assert np.linalg.norm(vecs[b]) == pytest.approx(1.0)


# ---------------------------------------------------------------- psd_sqrt

def test_psd_sqrt_diag(sol, no_loops):
    out = sol.psd_sqrt(np.array([[4.0, 0.0], [0.0, 9.0]]))
    np.testing.assert_allclose(out, [[2.0, 0.0], [0.0, 3.0]], atol=1e-12)
    no_loops(sol.psd_sqrt)


def test_psd_sqrt_batched_rank_deficient(sol):
    rng = np.random.default_rng(0)
    S = _random_psd(rng, (3,), 5, rank=2)  # PSD with zero eigenvalues (may round slightly negative)
    R = sol.psd_sqrt(S)
    assert R.shape == S.shape
    assert not np.isnan(R).any(), "no NaN allowed: clip tiny negative eigenvalues to 0"
    np.testing.assert_allclose(R, np.swapaxes(R, -1, -2), atol=1e-10, err_msg="R should be symmetric")
    np.testing.assert_allclose(R @ R, S, atol=1e-8, err_msg="R @ R should give S back")
