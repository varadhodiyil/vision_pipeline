from lxml import etree
import requests
import re
import urllib3
import os
#import cookielib
import json

def get_soup(url,header):
    # html = requests.get(url, header).content
    # html = html.decode('ascii',errors="ignore")
    # # print(html)
    # w = open('th.html','w')
    # w.writelines(html)
    h = open("th.html","r")
    _html = h.read()
    # return BeautifulSoup( html, 'html.parser')
    parser = etree.HTMLParser()
    return etree.fromstring(_html , parser=parser)

    # return BeautifulSoup(urllib3.urlopen(urllib2.Request(url,headers=header)),'html.parser')


query = "sedan"
# query = raw_input("query image")# you can change the query for the image  here
image_type="ActiOn"
query= query.split()
query='+'.join(query)
url="https://www.google.co.in/search?q="+query+"&source=lnms&tbm=isch"
print(url)
#add the directory for your image here
DIR="Pictures"
header={
'Host': "www.google.co.in",
"User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:73.0) Gecko/20100101 Firefox/73.0",
"Accept": "*/*",
"Accept-Language": "en-US,en;q=0.5",
"Accept-Encoding": "gzip, deflate, br",
"Referer": "https://www.google.co.in/",
"X-Same-Domain": "1",
"Content-Type": "application/x-www-form-urlencoded;charset=utf-8",
"Content-Length": "133",
"Origin": "https://www.google.co.in",
"DNT": "1",
"Connection": "keep-alive",
"Cookie": "NID=200=WOYVWs5k3yagIphUuSjPFYnw9RG0ppTZp3rfCnkU7oCU8FyFgEGWXB82Cry-NCgwHzpqkAnPrDoX8_i6b8SUeHB64w0jnKZwV5BRtBDTchZuaLEbZKY0tyYDsVLiiSXuBZiZ4AdKopp_TpMbDm2RcndrgLMi5J-jJrOulyecI_c; 1P_JAR=2020-03-19-20; CONSENT=WP.284cb8; OTZ=5372410_56_56_123900_52_436380",
"TE": "Trailers"
}
soup = get_soup(url,header)


# ActualImages=[]# contains the link for Large original images, type of  image
for a in soup.xpath("//td/a/@href"):
    print(a)
    

# print("there are total" , len(ActualImages),"images")
