import os
from datetime import datetime

from solarmask.sharp_cea_720s_nrt import datefmt


def date_to_fname(date: datetime):
    return date.strftime(datefmt) + ".npy"


def fname_to_date(fname):
    return datetime.strptime(fname, f"{datefmt}.npy")


def series_root(root: str, series: str):
    """
    <root>/<series>
    """
    return os.path.join(root, series)


def hnum_root(root: str, series: str, hnum: int) -> str:
    """
    <root>/<series>/<hnum>
    """
    return os.path.join(series_root(root, series), str(hnum))


def segment_root_for_hnum(root: str, series: str, hnum: int, segment: str) -> str:
    """
    <root>/hmi.sharp.../<hnum>/<segment>
    """
    return os.path.join(hnum_root(root, series, hnum), segment)


def data_path_for_hnum_date(root: str, series: str, hnum: int, segment: str, date: datetime) -> str:
    """
    <root>/hmi.sharp.../<hnum>/<segment>/<date>.npy
    """
    return os.path.join(segment_root_for_hnum(root, series, hnum, segment), date_to_fname(date))
