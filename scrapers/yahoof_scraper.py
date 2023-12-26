import requests
from bs4 import BeautifulSoup

def get_yahoo_finance_news_rss():
    import re
    rss_url = "https://finance.yahoo.com/rss/stock-market-news/"
    response = requests.get(rss_url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        items = soup.find_all('item')
        item_list = list()
        for item in items:
            
            title = item.find('title').text
            pub_date = item.find('pubdate').text
            pattern = re.compile(r'<link/>(.*?)<pubdate>', re.DOTALL)
            match = pattern.search(str(item))
            if match:
                link = match.group(1)
            '''
            print(f"Title: {title}")
            print(f"Link: {link}")
            print(f"Published Date: {pub_date}")
            print("\n" + "-" * 50 + "\n")
            '''
           
            item_list.append((title,link,pub_date))
        return item_list
        

def get_news_text(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        #print(response.text)
        story_items = soup.find('div', class_='caas-body')
        print(story_items)
        for item in story_items:
            readable_text = item.get_text(separator='\n', strip=True)
            print(readable_text)
        


 
import unittest
import logging  
import pandas as pd
   
class TestYahoof(unittest.TestCase):
    
    def test_rss(self):
        print("Start...")
        top_news = get_yahoo_finance_news_rss()
       
        for n in top_news:
            print("------------ " + n[0])
            get_news_text(n[1])

        self.assertGreater(len(top_news),0)
        

if __name__ == "__main__":
    unittest.main()