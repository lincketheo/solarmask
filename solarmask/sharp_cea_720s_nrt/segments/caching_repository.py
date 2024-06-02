import os
from datetime import datetime
import numpy as np
from solarmask.sharp_cea_720s_nrt.segments.local import LocalSHARPSegmentDataSource
from solarmask.sharp_cea_720s_nrt.segments.network import NetworkSHARPSegmentDataSource

datefmt = "%Y.%m.%d_%H:%M:%S_TAI"
src_name = "hmi.sharp_cea_720s_nrt"


class Sharp720sRepository:
    def __init__(self, local: LocalSHARPSegmentDataSource, network: NetworkSHARPSegmentDataSource):
        self.local = local
        self.network = network

    def get_segment(self, hnum: int, date: datetime, segment: int) -> np.array:
        pass
