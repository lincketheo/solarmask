import drms
import os
from astropy.io import fits
import numpy as np

client = drms.Client()
series = 'hmi.sharp_cea_720s_nrt'
data_root = "downloads"


