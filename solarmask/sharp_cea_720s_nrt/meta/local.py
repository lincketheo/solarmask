class LocalSHARPMetaDataSource:
    def __init__(self, cache_root: str):
        self.root = cache_root

    def get_hnums_available(self):
        pass

    def get_dates_available(self, hnum: int):
        pass

    def get_segments_available(self, hnum: int):
        pass
