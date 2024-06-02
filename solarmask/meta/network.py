import drms


class NetworkDRMSDataSource:
    def __init__(self, cache_root: str, client=drms.Client()):
        self.root = cache_root
        self.client = client

    def get_series_available(self):
        return self.client.series()

    def get_pkeys(self, series: str):
        return self.client.pkeys(series)
