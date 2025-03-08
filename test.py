# encoding:utf-8
import requests


def police():
    print("police")
    token = '243eb0062a18451c90ac6d123edf34e0' #在pushplus网站中可以找到
    title= '报警！' #改成你要的标题内容
    content ='有人员入侵!' #改成你要的正文内容
    url = 'http://www.pushplus.plus/send?token='+token+'&title='+title+'&content='+content
    requests.get(url)

if __name__ == '__main__':
    police()

