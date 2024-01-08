import requests
from bs4 import BeautifulSoup
from selenium import webdriver
import html




def get_investing_dot_com_news_rss():
    import re
    rss_urls = ["https://www.investing.com/rss/market_overview_Opinion.rss",
                "https://www.investing.com/rss/market_overview_Technical.rss",
                "https://www.investing.com/rss/market_overview_Fundamental.rss",
                "https://www.investing.com/rss/market_overview_investing_ideas.rss",
                "https://www.investing.com/rss/news_25.rss"
                ]
    item_list = list()
    
    browser = webdriver.Chrome()
   

    for rss_url in rss_urls:

        htmlp = browser.page_source
        browser.get(rss_url)

        
        soup = BeautifulSoup(htmlp, "html.parser")
        html_string =  html.unescape(str(soup.contents))
        soup = BeautifulSoup(html_string, 'lxml')
        items = soup.find_all('item')        
        
        for item in items:

            title = item.find('title').text
            pub_date = item.find('pubdate').text
            pattern = re.compile(r'<link/>(.*?)</item>', re.DOTALL)
            match = pattern.search(str(item))
            if match:
                link = match.group(1)
            
            print(f"Title: {title}")
            print(f"Link: {link}")
            print(f"Published Date: {pub_date}")
            print("\n" + "-" * 50 + "\n")
            
            item_list.append((title,link,pub_date))


    return item_list



def get_news_text_selenium(url):

    from selenium import webdriver
    from selenium.webdriver.support.ui import WebDriverWait
    import re
    import html

    browser = webdriver.Chrome()
    browser.get(url)
    WebDriverWait(browser, 1).until(lambda driver: driver.execute_script("return document.readyState === 'complete'"))

    htmlp = browser.page_source
    soup = BeautifulSoup(htmlp, "html.parser")
        
    story_items = soup.find('div', class_='WYSIWYG articlePage')
    story_items = filter(lambda x: x.name == 'p', story_items)
    readable_text = ""
    for item in story_items:
        readable_text += re.sub("<[^>]*>", "",(item.get_text(separator='\n', strip=True)))

    return readable_text

from datetime import datetime
def get_YYYY_MM_DD_idc(date_string:str) -> datetime :
    date_object = datetime.strptime(date_string, "%b %d, %Y %H:%M %Z")
    print(date_object)
    try:        
        return date_object.strftime("%Y-%m-%d")
    except ValueError:
        raise ValueError("Invalid date format")

 
import unittest
import logging  
import pandas as pd
   
class TestinvestingDotCom(unittest.TestCase):

    # def test_single_news(self):
    #     url ="https://www.investing.com/analysis/how-patience-and-delayed-gratification-can-fuel-longterm-gains-200643447"
    #     news_text   = get_news_text_selenium(url)
    #     print(news_text)
    
    def test_rss(self):
        import os

        print("Start...")
        top_news = get_investing_dot_com_news_rss()
       
        for n in top_news:
            print("------------ " + n[0])
            get_news_text_selenium(n[1])

        self.assertGreater(len(top_news),0) 
         


if __name__ == "__main__":
    unittest.main()