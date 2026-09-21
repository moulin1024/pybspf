"""Shared tensor algebra for array operands."""
def tensor_product(left, right, coefficients=None, *, paired=False):
    if paired:
        if coefficients is None:
            return (left[:, :, None] * right[:, None, :]).reshape(len(left), -1)
        return ((left @ coefficients) * right).sum(axis=1)
    if coefficients is None:
        raise ValueError("Tensor-grid evaluation requires coefficients")
    return left @ coefficients @ right.T


def tensor_elliptic_solve(load, denominator, left=None, right=None):
    """Generalized symmetric tensor Poisson inverse in mass-normalized modes.

    Rotations have columns of generalized eigenvectors. Identity rotations
    recover the original stream NS diagonal inertia solve exactly.
    """
    transformed = load if left is None else left.T @ load
    if right is not None:
        transformed = transformed @ right
    result = transformed / denominator
    if left is not None:
        result = left @ result
    if right is not None:
        result = result @ right.T
    return result
