import os
from datetime import datetime
import numpy as np

from solarmask.sharp_cea_720s_nrt import series
from solarmask.sharp_cea_720s_nrt.file_locator import data_path_for_hnum_date, series_root, segment_root_for_hnum, \
    hnum_root, fname_to_date


class LocalSHARPSegmentDataSource:

    def __init__(self, download_root: str = "./downloads"):
        self.root = download_root

    def save_segment(self, hnum: int, segment: str, date: datetime, data: np.array):
        """
        Saves a segment belonging to hnum, segment, date
        """
        np.save(data_path_for_hnum_date(self.root, series, hnum, segment, date), data)

    def get_segment(self, hnum: int, date: datetime, segment: str) -> np.array:
        """
        Retrieves a local segment (if it exists)
        """
        return np.load(data_path_for_hnum_date(self.root, series, hnum, segment, date))

    def get_hnums(self):
        """
        Returns a list of harpnumbers that have "at least some information stored in them"
        """
        return [int(x) for x in os.listdir(series_root(self.root, series))]

    def get_dates(self, hnum: int, segment: str):
        """
        Retrieves all the elements that have been saved in hnum, segment
        """
        f = segment_root_for_hnum(self.root, series, hnum, segment)
        return [fname_to_date(f) for f in os.listdir(f)]

    def get_saved_segments_for_hnum(self, hnum):
        """
        Retrieves a list of segments that have some information in them
        """
        f = hnum_root(self.root, series, hnum)
        return [d for d in os.listdir(f)]

    def get_mag_cont_dates_intersection(self, hnum: int):
        """
        Retrieves all the dates stored where all segments saved have some data in them
        """
        segments = self.get_saved_segments_for_hnum(hnum)
        ret = set()
        for segment in segments:
            ret = ret & set(self.get_dates(hnum, segment))
        return sorted(list(ret))
