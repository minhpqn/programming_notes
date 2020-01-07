"""
Usage: get_dropbox_directlink.py < file
"""
import re
import sys


def get_direct_link(link):
    link_ = re.sub(r"dropbox\.com", "dl.dropboxusercontent.com", link)
    link_ = re.sub(r"\?dl=0", "", link_)
    return link_


for line in sys.stdin:
    line = line.strip()
    print(get_direct_link(line))


