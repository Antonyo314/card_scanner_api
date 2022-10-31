import requests
from pprint import pprint
url = 'http://0.0.0.0:5000/api'
my_img = {'image': open('/Users/anton/Desktop/image.jpg', 'rb')}
r = requests.post(url, data={'token': 'e121166d02ab7dde5dff5bffed004b72'}, files=my_img)

pprint(r.json())

