import drms
from astropy.io import fits

from solarmask.sharp_cea_720s_nrt import series


class SegmentEntry:
    """
    A data class representing all the information needed to fetch
    and store a segment
    """

    def __init__(self, segment: str, hnum: int, tai_date_str: str, path: str):
        self.segment = segment
        self.hnum = hnum
        self.tai_date_str = tai_date_str
        self.path = path


class NetworkSHARPSegmentDataSource:
    url_prefix = 'http://jsoc.stanford.edu'

    def __init__(self, client=drms.Client()):
        self.client = client

    def get_entries_for_hnum_seg(self, hnums: list[int], segments: list[str]):
        """
        Returns a list of SegmentEntries (just the meta information) for a list of
        harpnumbers and segments
        :param hnums: The hnums to fetch
        :param segments: The segments to fetch
        :return: A list of SegmentEntries (not ordered in any way)
        """
        hnums_str = ','.join([str(h) for h in hnums])
        keys, query = self.client.query(series + f'[{hnums_str}]', seg=", ".join(segments), pkeys=True)
        ret = []
        for i in range(len(query)):
            for segment in segments:
                path, hnum, date = query.iloc[i][segment], keys.iloc[i].HARPNUM, keys.iloc[i].T_REC
                if len(path) <= 0 or path[0] != '/':
                    print("Skipping:", hnum, date, "path was: ", path)
                else:
                    entry = SegmentEntry(segment, hnum, date, path)
                    ret.append(entry)
        return ret

    def fetch_entry_data(self, entry: SegmentEntry):
        url = self.url_prefix + entry.path
        data = fits.getdata(url)
        return data
