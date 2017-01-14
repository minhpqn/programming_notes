import urllib2
import urllib
import json

# Get image information from mediawiki by using imageinfo api
# Reference:
# Imageinfo API: https://www.mediawiki.org/wiki/API:Imageinfo
# HOWTO Fetch Internet Resources Using urllib2: https://docs.python.org/2.7/howto/urllib2.html

def imageinfo(image_name):
    data = {}
    data['action'] = 'query'
    data['titles'] = image_name
    data['prop'] = 'imageinfo'
    data['format'] = 'json'
    data['iiprop'] = 'url'
    url_values = urllib.urlencode(data)
    url = 'https://en.wikipedia.org/w/api.php'
    full_url = url + '?' + url_values
    
    data = urllib2.urlopen(full_url)
    info = data.read()
    image_data = json.loads(info)

    pages = image_data['query']['pages']
    image_url = ''
    for key in pages.keys():
        image_url = pages[key]['imageinfo'][0]['url']
    return image_url

def main():
    image_name = 'File:Albert Einstein Head.jpg'
    image_url = imageinfo(image_name)
    print image_name
    print image_url

if __name__ == '__main__':
    main()


