import urllib.request
import json
   
if __name__ == "__main__":
    restUri = 'http://172.16.16.169:8081/facedetect'
    PostParam = './image/test2.jpg'
    DATA = PostParam.encode('utf8')
    req = urllib.request.Request(url = restUri, data=DATA, method='POST')
    req.add_header('Content-type', 'application/form-data')
    r = urllib.request.urlopen(req).read()
    print(r.decode('utf8'))
    org_obj = json.loads(r.decode('utf8'))
    print(org_obj['token'])