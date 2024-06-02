from solarmask.meta.network import NetworkDRMSDataSource

g = NetworkDRMSDataSource("./downloads")

for series in g.get_series_available():
    print("===================")
    print(f"SERIES: {series}")
    print(g.get_pkeys(series))
    print("===================")