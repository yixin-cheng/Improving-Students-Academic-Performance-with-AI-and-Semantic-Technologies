"""
This file is written for retrieving paper from Springer by using BeautifulSoup
"""

import requests
from bs4 import BeautifulSoup
import xlwt
from selenium import webdriver
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
def request_springer(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.text
    except requests.RequestException:
        return None

paper = xlwt.Workbook(encoding='utf-8', style_compression=0)

sheet = paper.add_sheet('crawl result', cell_overwrite_ok=True)
sheet.write(0, 0, 'Title')
sheet.write(0, 1, 'Link')
sheet.write(0, 2, 'Abstract')

n = 1

def get_abstract(url):
    html = request_springer(url)
    soup = BeautifulSoup(html, 'lxml')
    ab=soup.find(class_='Abstract')
    if(ab!=None):
        ab=ab.find('p').getText
    else:
        driver = webdriver.Chrome(ChromeDriverManager().install()) # Chromedriver version must be Chrome version 90
        driver.get(url)
        try:
            ab = driver.find_element_by_id(id_='Abs1-content').text
        except:
            ab = 'invalid'

    return ab

def save_to_excel(soup):
    list = soup.find(class_='content-item-list').find_all('li')

    for paper in list:
        title = paper.find(class_='title').string
        if(title==None):
            continue
        link = 'https://link.springer.com'+paper.find('h2').find('a').get('href')

        abstract=str(get_abstract(link))


        print('result：'  + ' | ' + title +'|' + link +' |'+abstract)

        global n

        sheet.write(n, 0, title)
        sheet.write(n, 1, link)
        sheet.write(n, 2, abstract)
        n = n + 1

def main(page):
    url='https://link.springer.com/search/page/'+str(page)+ '?facet-language=%22En%22&showAll=false&query=%28%22computer+science%22+OR+%22computer+engineering%22+OR+%22informatic%22%29+AND+%28%22curriculum%22+OR+%22course+description%22++OR+%22learning+outcomes%22+OR+%22curricula%22+OR+%E2%80%9Csyllabus%E2%80%9D%29+AND+%28%22semantic%22+OR+%22ontology%22+OR+%22linked+data%22+OR+%22knowledge+tracing%22+OR+%22Linked+open+data%22+OR+%22open+data%22%29'
    html = request_springer(url)
    soup = BeautifulSoup(html, 'lxml')
    save_to_excel(soup)

if __name__ == '__main__':

    for i in range(1, 3):
        print(i)
        main(i)
#save the result to local file
paper.save(u'datasetbyabstract.csv')