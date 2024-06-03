import drms


class DRMSFacade:
    def __init__(self, client=drms.Client()):
        self.client = client

    def get_series_available(self):
        return self.client.series()

    def get_pkeys(self, series: str):
        return self.client.pkeys(series)
