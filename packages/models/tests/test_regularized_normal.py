import numpy as np

from bspf_models.elliptic.regularized_normal import normal_weights


def test_cutoff_preserves_low_modes_and_reduces_gain():
    ratios = np.array([0.1, 0.3, 0.5])
    ri, w = normal_weights(ratios, retained=6)
    _, unregularized = normal_weights(ratios, retained=14)
    for degree in range(7):
        np.testing.assert_allclose(w @ ri**degree, ratios**degree, atol=1e-11)
    for derivative in (1, 2):
        _, dw = normal_weights(ratios, retained=6, derivative=derivative)
        target = 6 * ratios if derivative == 2 else 3 * ratios**2
        np.testing.assert_allclose(dw @ ri**3, target, atol=1e-10)
    assert (
        np.max(np.sum(abs(w), axis=1))
        < np.max(np.sum(abs(unregularized), axis=1)) / 1000
    )


def test_linear_unknown_field_relation():
    _, w = normal_weights([0.15, 0.3, 0.45], retained=6)
    rng = np.random.default_rng(99)
    inward = rng.normal(size=(7, 25, 11))
    outward = rng.normal(size=(7, 3, 11))
    c = rng.normal(size=11)
    relation = outward - np.einsum("es,tsn->ten", w, inward)
    np.testing.assert_allclose(
        relation @ c, outward @ c - np.einsum("es,ts->te", w, inward @ c), atol=1e-12
    )
