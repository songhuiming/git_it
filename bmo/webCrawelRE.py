#run in ipython     %run h:/python/webCrawelRE.py

# encoding:utf-8
import re
import urllib as urllib

domain = 'http://www.liaoxuefeng.com'
pythonpage = '/wiki/001374738125095c955c1e6d8bb493182103fac9270762a000'
#urllib.urlopen(r'file:/h:/python/JavaScript.html')
f = urllib.urlopen(domain + pythonpage)
fpage = f.read()
f.close()

headhtml = open(r'/home/shm/Dropbox/python/interesting/0.html', 'r').read()

fpage = fpage.split(r'<div id="x-wiki-prev-next" class="uk-clearfix uk-margin-left uk-margin-right">')   # remove duplicated contents index
fpage = fpage[0]

links = re.findall(r'/wiki/\w*/\w*', fpage)
links.insert(0, pythonpage)

for i in links:
    url = 'http://www.liaoxuefeng.com' + i
    f = urllib.urlopen(url)
    html = f.read()

    #get title
    titlere = re.compile(r'<title>(.*)</title>')
    html_title = re.findall(titlere, html)[0]
    title = html_title.decode('utf-8')
    titlelist = title.split(' - ')                  #to remove the -liao xue feng de bo ke
    titlelist.pop(-1)
    title = ''.join(titlelist).replace(r'/', '-')            #for the 'map/reduce' title 

    #get text
    html = html.split(r'<div class="x-wiki-content">')[1]
    html = html.split(r'<div id="x-wiki-prev-next" class="uk-clearfix uk-margin-left uk-margin-right">')[0]
    #download pic, like  <p><img src="/files/attachments/00138595453161126cc9f11f1d441b0934661239528fa55000/0" alt="tpci_trends"></p>
    html = html.replace(r'src="', 'src="' + domain)

    html = headhtml + html + "</body></html>"

    # output file
    output = open('/home/shm/Dropbox/python/interesting/liaoxuefeng/' + "%d" % links.index(i) + title + '.html', 'w')
    output.write(html)
    output.close()