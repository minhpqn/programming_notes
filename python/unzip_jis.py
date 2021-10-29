import os
import zipfile
import argparse

f = r'Keisushozei作業.zip'

with zipfile.ZipFile(f) as z:
    for info in z.infolist():
        info.filename = info.orig_filename.encode('cp437').decode('cp932')
        if os.sep != "/" and os.sep in info.filename:
            info.filename = info.filename.replace(os.sep, "/")
        z.extract(info)