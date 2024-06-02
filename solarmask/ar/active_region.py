from solarmask.ar.segments import Segments

import numpy as np
import warnings


class ActiveRegion(Segments):
    def __init__(self, bx: np.array, by: np.array, bz: np.array, cont: np.array):
        """An Active Region is an entry point into 
        parameterization, segmentation, and graph methods. 

        An Active Region is uniquely determined by it's harpnumber (hnum) and date (date)
        If the specified active region given (hnum, date) contains a nan, self.valid is False
        (otherwise self.valid is True) and subsequent calls to ActiveRegion methods may fail 
        given the existence of nans. Check status of the active region by self.valid.
        The existence of nan's implies the image is moving off the solar limb. This is an issue that
        could be turned into a ticket and fixed, but for the time being I am ignoring these images

        Magnetograms and continuums must be organized by the following standard (with root as the root)
        (This standard may be loosened in the future and subsequent documentation will change)

        - root
            - magnetogram
                - sharp_<hnum>
                    - hmi.sharp_cea_720s.<hnum>.<year><month><day>_<hour><minute><second>_TAI.bp.fits
                    - hmi.sharp_cea_720s.<hnum>.<year><month><day>_<hour><minute><second>_TAI.br.fits
                    - hmi.sharp_cea_720s.<hnum>.<year><month><day>_<hour><minute><second>_TAI.bt.fits
            - continuum
                - sharp_<hnum>
                    - hmi.sharp_cea_720s.<hnum>.<year><month><day>_<hour><minute><second>_TAI.continuum.fits
        
        So for example, (this will work on Swami), a new active region call:
        ```python
        ActiveRegion(7115, datetime(2017, 9, 3, 10), "/srv/data/thli2739") 
        ```

        Would **require** the following files or symlinks to exist:

        - /srv/data/thli2739
            - /magnetogram
                - /sharp_7115
                    - /hmi.sharp_cea_720s.7115.20170903_100000_TAI.bp.fits
                    - /hmi.sharp_cea_720s.7115.20170903_100000_TAI.br.fits
                    - /hmi.sharp_cea_720s.7115.20170903_100000_TAI.bt.fits
            - /continuum
                - /sharp_7115
                    - /hmi.sharp_cea_720s.7115.20170903_100000_TAI.continuum.fits

        Args:
            hnum (int): The specified harpnumber - file must exist in root/magnetogram/sharp_{hnum} and root/continuum/sharp_{hnum}
            date (datetime): The specified active region date and time - this date must exist in the specified harpnumber data folder 
            root (string): The path to the data. Root must be a directory that holds both root/magnetogram and root/continuum. Inside both
            of these subfolders, there must be a series of folders labeled sharp_{hnum} that contain the sequence of fits files for extraction
        """

        # Generate xyz components of magnetic field and continuum

        bz[np.abs(bz) < 0.001] = 0.0
        bx[np.abs(bx) < 0.001] = 0.0
        by[np.abs(by) < 0.001] = 0.0

        self.valid = True  # Valid is false
        percent_nan = np.count_nonzero(np.isnan(bz)) / bz.size
        if percent_nan > 0.0:
            self.valid = False
            warnings.warn(f"Hnum {hnum} date {date} is {percent_nan * 100}% nan, skipping")
            return

        # Now bx by bz are defined so generate the parameter class
        super().__init__(bz, by, bx, cont, data_products)

        # The Three data sets
        self.__baseline = None
        self.__segmented = None

    @property
    def baseline_dataset(self):
        if "baseline" not in self.data_products:
            warnings.warn("baseline is not in desired data products, skipping")
            return None

        if self.__baseline is None:
            self.__baseline = self.physical_features(self.full)
            self.__baseline.update(self.meta)
        return self.__baseline

    @property
    def segmented_dataset(self):
        if "segmented" not in self.data_products:
            warnings.warn("Segmented is not in desired data products, skipping")
            return None

        if self.__segmented is None:
            self.__segmented = dict()
            self.__segmented.update(self.physical_features(self.nl, "nl_"))
            self.__segmented.update(self.physical_features(self.umbra, "umbra_"))
            self.__segmented.update(self.physical_features(self.penumbra, "penumbra_"))
            self.__segmented.update(self.meta)

        return self.__segmented
