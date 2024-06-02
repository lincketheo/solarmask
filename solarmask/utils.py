import numpy as np
from scipy.stats import kurtosis, skew
import torch


def gradient(data):
    """Numerical Gradient of data

    Args:
        data (2d np.array): A 2d array of scalars

    Returns:
        (np.array, np.array): 2 arrays representing dy, dx respectively
    """
    return np.gradient(data)


def cov(m, rowvar=False):
    """Covariance of a dataset m

    Args:
        m (2d array): Two lists of data (x, y) to take the covariance from
        rowvar (bool): Calculate row variance of the data

    Returns:
        Covariance of the two data sets
    """
    if m.dim() > 2:
        raise ValueError('m has more than 2 dimensions')
    if m.dim() < 2:
        m = m.view(1, -1)
    if not rowvar and m.size(0) != 1:
        m = m.t()
    # m = m.type(torch.double)  # uncomment this line if desired
    fact = 1.0 / (m.size(1) - 1)
    m -= np.mean(m, dim=1, keepdim=True)
    mt = m.t()  # if complex: mt = m.t().conj()
    return fact * m.matmul(mt).squeeze()


def unpad(z):
    """Removes surrounding whitespace (False values) from a binary array z

    Args:
        z (2d array): A binary array

    Returns:
        Same type as z: z without the surrounding values False
    """
    x, y = np.where(z)
    return z[np.min(x):np.max(x) + 1, np.min(y):np.max(y) + 1]


def shannon_entropy(z):
    """Computes shannon entropy of a binary array z

    Args:
        z (2d array): A binary array

    Returns:
        float: The shannon entropy of array z
    """
    try:
        import skimage
    except ImportError:
        raise NotImplementedError("Shannon Entropy isn't implemented without skimage, to continue, plese"
                                  "install skimage")
    return skimage.measure.shannon_entropy(unpad(z))


def stat_moment(data):
    """The statistical \"moment\" of a distribution

    Args:
        data (np.array): A (usually 2d) array containing a distribution of scalars

        For a dataset X,

        The mean measures the "center point" of the data X
        $$mean(X) = \\frac{1}{|X|}\\sum_{x \\in X}x$$

        The standard deviation measures the "spread" of the data X
        $$std(X) = \\sqrt{\\frac{1}{|X|}\\sum_{x \\in X}(x - mean(x))^2}$$

        The skew measures the "shift" of the center point of X
        $$skew(X) = \\frac{1}{|X|}\\sum_{x \\in X}(\\frac{x - mean(x)}{std(x)})^3$$

        The kurtosis measures the "taildness" of a X
        $$kurtosis(X) = \\frac{1}{|X|}\\sum_{x \\in X}(\\frac{x - mean(x)}{std(x)})^4 - 3$$

        $$M(X) = (mean(X), std(X), skew(X), kurtosis(X))^T$$

    Returns:
        A pytorch tensor containing average, standard deviation, skew, kurtosis (in that order)
    """
    avg = np.mean(data)
    std = np.std(data)
    # skw = np.mean(((data - avg)/std)**3)
    # krt = np.mean(((data - avg)/std)**4) - 3.0
    krt = kurtosis(data)
    skw = skew(data)
    return np.array([avg, std, skw, krt])


def p_norm(*args, p=2):
    """The one norm of a set of data elements

    Args:
        args: A List of elements that can be taken the absolute value of
        p: Order of the norm. Defaults to 2

    Returns:
        The result of summing the application of torch.abs() on all elements of data
    """
    n = None
    for i in args:
        if n is None:
            n = i ** p
        else:
            n += i ** p

    return n ** (1 / p)


def shape_check(*args: np.array):
    shape = None
    for i in args:
        if shape is None:
            shape = i.shape
        if i.shape != shape:
            raise ValueError("Shape mismatch")


def stat_moment_label(prefix: str):
    """Labels the statistical moment with an added prefix

    Args:
        prefix (string): Prefix to add to labels

    Returns:
        tuple: List of labeled elements
    """
    return prefix + "_mean", prefix + "_std", prefix + "_skew", prefix + "_kurt"


def get_gpu_dev():
    return torch.device("cpu")
