import os
import argparse
import requests
from time import sleep
from logzero import logger
from bs4 import BeautifulSoup


def listFD(url, ext=''):
    page = requests.get(url).text
    soup = BeautifulSoup(page, 'html.parser')
    return [node.get('href') for node in soup.find_all('a') if node.get('href').endswith(ext)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--link", required=True, help="Path to http directory")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--ext", default="tgz", help="File extension")
    parser.add_argument("--sleep", default=20, type=int, help="Sleep time")
    parser.add_argument("--interval", default=5, type=int, help="Sleep interval to avoid connection refused")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    i = 0
    for file in listFD(args.link, args.ext):
        filename = os.path.join(args.output_dir, file)
        url = args.link + '/' + file
        i += 1
        logger.info("Download file %s" % file)
        if not os.path.isfile(filename):
            content = None
            while content is None:
                try:
                    r = requests.get(url, allow_redirects=True, verify=False, timeout=60)
                    content = r.content
                    open(filename, 'wb').write(r.content)
                except requests.exceptions.ConnectionError as e:
                    logger.info("Sleep for {} seconds and try again".format(args.sleep))
                    sleep(args.sleep)
    
    logger.info("Downloaded {} files".format(i))

    

