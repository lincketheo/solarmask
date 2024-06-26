from .parameters import Parameters

import numpy as np

from skimage.filters import threshold_local
from skimage.morphology import square, binary_dilation
from skimage.measure import label

from solarmask.utils import shape_check


def group_pixels(mask):
    """Groups pixels in a binary mask based on if they are touching

    Args:
        mask (np.array): A mask with binary pixels

    Returns:
        labeled: a mask that is labeled from 0 to number of groups (0 is the background) the size of the number in the label doesn't mean anything
        labels: a list of labels in labeled
        sizes: a list of sizes corresponding to each element in labels
    """
    labeled = label(mask, connectivity=2)
    labels = np.unique(labeled)[1:]
    sizes = np.array([np.count_nonzero(labeled == i) for i in labels])
    return labeled, labels, sizes


def remove_small_groups(labels: np.array, sizes: np.array, p=500):
    """Removes groups of pixels smaller than p

    Args:
        labels (np.array): The distinct labels found in labeled
        sizes (np.array): The size of each label in labels
        p (int, optional): Smallest group size. Defaults to 500.

    Returns:
        labels: Labels filtered
        sizes: sizes filtered
    """
    if len(sizes) == 0:
        return labels, sizes
    filt = np.argwhere((sizes < p))
    return np.delete(labels, filt), np.delete(sizes, filt)


def remove_percentage_max(labels, sizes, p=0.01):
    """Removes pixels that border the edge

    Args:
        labels (np.array): The distinct labels found in labeled
        sizes (np.array): The size of each label in labels
        p (int, optional): Smallest group size. Defaults to 500.

    Returns:
        labels: Labels filtered
        sizes: sizes filtered
    """
    if len(sizes) == 0:
        return labels, sizes
    filt = np.argwhere(sizes < p * np.max(sizes))
    return np.delete(labels, filt), np.delete(sizes, filt)


def __largest_n_clusters(labels, sizes, n=6):
    """Removes pixels that border the edge

    Args:
        labels (np.array): The distinct labels found in labeled
        sizes (np.array): The size of each label in labels
        p (int, optional): Smallest group size. Defaults to 500.

    Returns:
        labels: Labels filtered
        sizes: sizes filtered
    """
    if len(sizes) == 0:
        return labels, sizes
    n = min(n, len(labels))
    a = np.partition(sizes, -n)[-n]
    return labels[sizes >= a], sizes[sizes >= a]


class Segments:
    def __init__(self, parameters: Parameters, cont: np.ndarray):
        self.cont = cont
        shape_check(parameters.fields.bz, cont)
        self.shape = cont.shape
        self.parameters = parameters

        # Lazily initialized masks
        self.__full = np.ones(self.shape, dtype=bool)
        self.__background = None
        self.__umbra = None
        self.__penumbra = None
        self.__nl = None

    @property
    def nl(self, thresh=150, radius=5):
        """Generates neutral lines using morphological operations and a flux threshold
        finds neutral lines using the method described by Schrijver in
        
        **C. J. Schrijver. A characteristic magnetic field pattern associated with all major solar flares and its use in flare forecasting. *The Astrophysical Journal, 655(2), 2007.***

        Args:
            radius (int, optional): The found neutral line is one pixel thick, so dilating it slightly adds the surrounding neighborhood. This radius
            is passed to sklearn.morphology.square(radius) in the call to sklearn.morphology.binary_dilation. Defaults to 3
            thresh (int, optional): Flux threshold. Defaults to 150 (as used by Schjriver et. al).
        """
        if self.__nl is None:
            nl_mask = (binary_dilation(self.parameters.fields.bz < -thresh, square(radius)) &
                       binary_dilation(self.parameters.fields.bz > thresh, square(radius)))
            self.__nl = nl_mask.copy()

        return self.__nl

    @property
    def background(self):
        """Generates a background mask. If background is already generated, does nothing
        background is simply $$\\neg (Umbra \\cup Penumbra \\cup Neutral Line)$$
        """
        if self.__background is None:
            self.__background = ~(self.nl | self.umbra | self.penumbra)
        return self.__background

    @property
    def umbra(self):
        if self.__umbra is None:
            self.__assert_umbra_penumbra()
        return self.__umbra

    @property
    def penumbra(self):
        if self.__penumbra is None:
            self.__assert_umbra_penumbra()
        return self.__penumbra

    @property
    def full(self):
        return self.__full

    def __assert_umbra_penumbra(self):
        """An original algorithm for detecting umbras and penumbras from a continuum image

        High Level Algorithm:

        1. bound continuum between 0 and 255

        2. use an adaptive filter on the bounded continuum

        3. group and label touching pixels 

        4. remove groups of pixels that are less than 500 from those remaining (if any) 

        5. remove groups of pixels that border the image (usually noise) from those remaining (if any) 

        6. remove all groups that are smaller than 10% of the size of the maximum group size from those remaining (if any)  

        7. remove take the largest 6 clusters from those remaining (if any) 

        8. The remaining groups are **penumbra outlines**, repeat the above process isolated only to the penumbra outlines
        if the difference between maximum and minimum flux in the mask is greater than 21000 and the resulting clusters are umbras

        9. Keep the remaining 6 largest umbras
        """
        # We first segment large groups (that may be penumbras or umbras)
        cont_bounded = (255 * (self.cont - np.min(self.cont)) / np.ptp(self.cont)).astype(np.uint8)

        block_size = np.min(self.shape)
        if block_size % 2 == 0:
            block_size -= 1

        offset = 10
        binary_adaptive = cont_bounded < (threshold_local(cont_bounded, block_size, offset=offset) - offset)

        labeled_0, labels, sizes = group_pixels(binary_adaptive)
        labels, sizes = self.__remove_bordering_pixels(labeled_0, labels, sizes)

        self.__umbra = (np.isin(labeled_0, labels))

        labels, sizes = remove_small_groups(labels, sizes, 100)

        self.__penumbra = np.zeros(self.shape, dtype=bool)

        # For each large group - determine if this is a penumbra / umbra combo or just umbra
        for i in labels:
            mask = labeled_0 == i
            mx = np.max(self.cont[mask])
            mn = np.min(self.cont[mask])
            t = (mx - mn) / 2 + mn
            um = mask & (self.cont <= t)
            pu = mask & (self.cont > t)

            # Small group = Pore (no penumbra)
            if np.count_nonzero(um) < 50:
                pass

            # PENUMBRA AND UMBRA
            elif mx - mn > 21000:
                # both umbra and penumbra
                self.__penumbra |= pu

            # ONLY UMBRA
            else:
                pass

        # What is not penumbra is umbra
        self.__umbra &= ~self.__penumbra

    def __remove_bordering_pixels(self, labeled: np.array, labels: np.array, sizes: np.array):
        """Removes pixels that border the edge

        Args:
            labeled (np.array): The labeled array returned from group_pixels
            labels (np.array): The distinct labels found in labeled
            sizes (np.array): The size of each label in labels
            p (int, optional): Smallest group size. Defaults to 500.

        Returns:
            labels: Labels filtered 
            sizes: sizes filtered 
        """
        if len(sizes) == 0:
            return labels, sizes
        bordered = []
        for i in range(len(labels)):
            rows, cols = np.where(labeled == labels[i])
            if min(rows) == 0 or min(cols) == 0:
                bordered.append(i)
            if max(cols) == self.shape[1] - 1 or max(rows) == self.shape[0] - 1:
                bordered.append(i)
        return np.delete(labels, bordered), np.delete(sizes, bordered)

    def draw(self, axs):
        axs[0][0].imshow(self.cont, interpolation="none")
        axs[0][1].imshow(self.parameters.fields.bz, interpolation="none")
        axs[1][0].imshow(self.umbra, interpolation="none")
        axs[1][1].imshow(self.penumbra, interpolation="none")
        axs[2][0].imshow(self.nl, interpolation="none")
        axs[2][1].imshow(self.background, interpolation="none")
