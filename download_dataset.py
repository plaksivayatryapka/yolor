import urllib.request
import zipfile


print('downloading pylons dataset')
url = "https://morescience.app:443/charts/pylons.zip"
urllib.request.urlretrieve(url, 'pylons.zip')

print('downloading weights')
url = "https://morescience.app:443/charts/yolor_p6.pt"
urllib.request.urlretrieve(url, 'yolor_p6.pt')

print('unpacking dataset')
with zipfile.ZipFile('pylons.zip', 'r') as zip_ref:
    zip_ref.extractall('datasets')