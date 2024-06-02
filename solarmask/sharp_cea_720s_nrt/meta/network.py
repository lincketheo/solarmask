import drms

from solarmask.sharp_cea_720s_nrt import series


class NetworkSHARPMetaDataSource:
    def __init__(self, cache_root: str, client=drms.Client()):
        self.root = cache_root
        self.client = client

    def get_hnums_available(self):
        pass

    def get_dates_available(self, hnum: int):
        pass

    def get_segments_available(self):
        return self.client.pkeys(series)
