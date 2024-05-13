"""
[download_unzip_dumpfile.py]
The purpose of this script is to download and unzip a comment dump file
from pushshift for a given year, month.
"""


import os
import sys
import time
import subprocess


dest = f'...' # path on machine where dump file will be downloaded

year = int(sys.argv[1])
month = int(sys.argv[2])
data_type = sys.argv[3]

prefix = 'RC' if data_type == 'comments' else 'RS'

if 1 <= month <= 9:
    zipped = f'{prefix}_{year}-0{month}.zst'
else:
    zipped = f'{prefix}_{year}-{month}.zst'

loc = f'https://files.pushshift.io/reddit/{data_type}/{zipped}'

# STAGE 2: download the dump file if it does not exist

if not zipped in os.listdir(dest + '/'):
    cmd = f'wget -P {dest} {loc}'
    os.system('nohup ' + cmd + ' &>/dev/null &')

# STAGE 3: decompress dump file

cmd = f'zstd -d {dest}/{zipped} --long=31'
os.system('nohup ' + cmd + ' &>/dev/null &')

# STAGE 4: remove zipped dump file

cmd = f'rm {dest}/{zipped}'
os.system('nohup ' + cmd + ' &>/dev/null &')