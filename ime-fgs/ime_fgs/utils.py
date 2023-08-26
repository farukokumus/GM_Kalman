# For research and educational use only, do not distribute without permission.
#
# Created by:
# Eike Petersen, Felix Vollmer with contribution from institute members and students
#
# University of Lübeck
# Institute for Electrical Engineering in Medicine
#
import types
from numpy import reciprocal, log, trace, atleast_2d, asscalar, ndim, isnan, absolute, finfo, float64, reshape, zeros, \
    array, shape, eye
from numpy.linalg import det, inv, svd, LinAlgError
from scipy.linalg import block_diag


def inherit_method_docs(cls):
    """
    Class annotator that makes this class inherit method docstrings from method docstrings of the parent class.

    Docstring inheritance is only applied if the method of the subclass does not provide a docstring itself.

    :param cls: The class to annotate.
    :return: The class with docstring inheritance applied.
    """
    for name, func in vars(cls).items():
        if isinstance(func, types.FunctionType) and not func.__doc__:
            for parent in cls.__bases__:
                par_func = getattr(parent, name, None)
                if par_func and getattr(par_func, '__doc__', None):
                    func.__doc__ = par_func.__doc__
                    break
    return cls


def kl_norm(mu1, S1, mu2, S2, sym=False):
    # todo take a look at the R implementation
    mu1 = atleast_2d(mu1)
    S1 = atleast_2d(S1)
    mu2 = atleast_2d(mu2)
    S2 = atleast_2d(S2)

    assert ndim(mu1) == 2
    assert ndim(mu2) == 2
    assert ndim(S1) == 2
    assert ndim(S2) == 2

    if mu1.shape[0] == 1:
        mu1 = mu1.T
    if mu2.shape[0] == 1:
        mu2 = mu2.T

    assert mu1.shape == mu2.shape
    assert S1.shape == S2.shape

    assert mu1.shape[0] == S1.shape[0] == S1.shape[1]
    assert mu1.shape[1] == 1

    l = mu1.shape[0]
    distance = .5 * (log(det(S2) / det(S1)) + trace(inv(S2) @ S1) + (mu2 - mu1).T @ inv(S2) @ (mu2 - mu1) - l)
    distance = asscalar(distance)

    assert distance >= 0.0
    assert not isnan(distance)

    if sym:
        return .5 * (distance + kl_norm(mu2, S2, mu1, S1))

    return distance


def kronecker_delta(x, tol=finfo(float64).eps):
    return (absolute(x) < tol).astype(float64)


def nullspace(A, atol=1e-13, rtol=0):
    """Compute an approximate basis for the nullspace of A.

    The algorithm used by this function is based on the singular value
    decomposition of `A`.

    Parameters
    ----------
    A : ndarray
        A should be at most 2-D.  A 1-D array with length k will be treated
        as a 2-D with shape (1, k)
    atol : float
        The absolute tolerance for a zero singular value.  Singular values
        smaller than `atol` are considered to be zero.
    rtol : float
        The relative tolerance.  Singular values less than rtol*smax are
        considered to be zero, where smax is the largest singular value.

    If both `atol` and `rtol` are positive, the combined tolerance is the
    maximum of the two; that is::
        tol = max(atol, rtol * smax)
    Singular values smaller than `tol` are considered to be zero.

    Return value
    ------------
    ns : ndarray
        If `A` is an array with shape (m, k), then `ns` will be an array
        with shape (k, n), where n is the estimated dimension of the
        nullspace of `A`.  The columns of `ns` are a basis for the
        nullspace; each element in numpy.dot(A, ns) will be approximately
        zero.
    """

    A = atleast_2d(A)
    u, s, vh = svd(A)
    tol = max(atol, rtol * s[0])
    nnz = (s >= tol).sum()
    ns = vh[nnz:].conj().T

    return ns


def szerobasis(A, atol=1e-13, rtol=0):
    """Compute an approximate basis corresponding to zero singular values of A.

    The algorithm used by this function is based on the singular value
    decomposition of `A`.

    Parameters
    ----------
    A : ndarray
        A should be at most 2-D.  A 1-D array with length k will be treated
        as a 2-D with shape (1, k)
    atol : float
        The absolute tolerance for a zero singular value.  Singular values
        smaller than `atol` are considered to be zero.
    rtol : float
        The relative tolerance.  Singular values less than rtol*smax are
        considered to be zero, where smax is the largest singular value.

    If both `atol` and `rtol` are positive, the combined tolerance is the
    maximum of the two; that is::
        tol = max(atol, rtol * smax)
    Singular values smaller than `tol` are considered to be zero.

    Return value
    ------------
    uzero : ndarray
        If `A` is an array with shape (m, k), then `uzero` will be an array
        with shape (m, n), where n is the estimated dimension of the
        nullspace of `A`.  The rows of `ns` are a basis for the
        nullspace;
    vhzero : ndarray
        If `A` is an array with shape (m, k), then `vhzero` will be an array
        with shape (k, n), where n is the estimated dimension of the
        nullspace of `A`.  The rows of `ns` are a basis for the
        nullspace;

    Consequently adding
    uzero*1e-9*vhzero to A, makes A regular
    """

    A = atleast_2d(A)
    u, s, vh = svd(A)
    tol = max(atol, rtol * s[0])
    nnz = (s >= tol).sum()
    snz = s[0:nnz]
    nz = (s < tol).sum()

    uzero = u[:, nnz:shape(A)[1]]
    vhzero = vh[nnz:shape(A)[0], :]
    unonzero = u[:, 0:nnz]
    vhnonzero = vh[0:nnz, :]

    return snz, uzero, vhzero, unonzero, vhnonzero


def robinv(A, atol=1e-13, rtol=0, sinv=1e-9):
    """Compute robust inverse based on numerically non-singular matrix by assigning
    sinv to zero singular values.

    The algorithm used by this function is based on the singular value
    decomposition of `A`.

    Parameters
    ----------
    A : ndarray
        A should be at most 2-D.  A 1-D array with length k will be treated
        as a 2-D with shape (1, k)
    atol : float
        The absolute tolerance for a zero singular value.  Singular values
        smaller than `atol` are considered to be zero.
    rtol : float
        The relative tolerance.  Singular values less than rtol*smax are
        considered to be zero, where smax is the largest singular value.
    sinv : float

    If both `atol` and `rtol` are positive, the combined tolerance is the
    maximum of the two; that is::
        tol = max(atol, rtol * smax)
    Singular values smaller than `tol` are considered to be zero.

    Return value
    ------------
    Ainv : ndarray
        Robust inverse of A

    """
    snz, uzero, vhzero, unonzero, vhnonzero = szerobasis(A, atol, rtol)

    regMatinv = unonzero @ block_diag(*reciprocal(snz)) @ vhnonzero
    singMatinv = uzero @ (eye(uzero.shape[1]) * reciprocal(sinv)) @ vhzero
    Ainv = regMatinv + singMatinv

    return Ainv


def try_inv_else_robinv(A, atol=1e-13, rtol=0, sinv=1e-9):
    """Try to compute inverse and if that fails, compute robust inverse based on
    numerically non-singular matrix by assigning sinv to zero singular values.

    The algorithm used by this function is based on the singular value
    decomposition of `A`.

    Parameters
    ----------
    A : ndarray
        A should be at most 2-D.  A 1-D array with length k will be treated
        as a 2-D with shape (1, k)
    atol : float
        The absolute tolerance for a zero singular value.  Singular values
        smaller than `atol` are considered to be zero.
    rtol : float
        The relative tolerance.  Singular values less than rtol*smax are
        considered to be zero, where smax is the largest singular value.
    sinv : float

    If both `atol` and `rtol` are positive, the combined tolerance is the
    maximum of the two; that is::
        tol = max(atol, rtol * smax)
    Singular values smaller than `tol` are considered to be zero.

    Return value
    ------------
    Ainv : ndarray
        Inverse or robust inverse of A

    """
    try:
        Ainv = inv(A)
    except LinAlgError:
        Ainv = robinv(A, atol, rtol, sinv)
    return Ainv


def rank(A, atol=1e-13, rtol=0):
    """Estimate the rank (i.e. the dimension of the nullspace) of a matrix.

    The algorithm used by this function is based on the singular value
    decomposition of `A`.

    Parameters
    ----------
    A : ndarray
        A should be at most 2-D.  A 1-D array with length n will be treated
        as a 2-D with shape (1, n)
    atol : float
        The absolute tolerance for a zero singular value.  Singular values
        smaller than `atol` are considered to be zero.
    rtol : float
        The relative tolerance.  Singular values less than rtol*smax are
        considered to be zero, where smax is the largest singular value.

    If both `atol` and `rtol` are positive, the combined tolerance is the
    maximum of the two; that is::
        tol = max(atol, rtol * smax)
    Singular values smaller than `tol` are considered to be zero.

    Return value
    ------------
    r : int
        The estimated rank of the matrix.

    See also
    --------
    numpy.linalg.matrix_rank
        matrix_rank is basically the same as this function, but it does not
        provide the option of the absolute tolerance.
    """

    A = atleast_2d(A)
    s = svd(A, compute_uv=False)
    tol = max(atol, rtol * s[0])
    rank = int((s >= tol).sum())

    return rank


def col_vec(np_array_or_list_or_scalar):
    return reshape(np_array_or_list_or_scalar, (-1, 1))


def row_vec(np_array_or_list_or_scalar):
    return reshape(np_array_or_list_or_scalar, (1, -1))


def mat(np_array_or_list_of_lists):
    return atleast_2d(np_array_or_list_of_lists)
