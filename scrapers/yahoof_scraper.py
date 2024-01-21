import requests
from bs4 import BeautifulSoup

def get_yahoo_finance_news_rss():
    import re
    rss_urls = ["https://finance.yahoo.com/rss/stock-market-news/","https://finance.yahoo.com/rss/economic-news/"]
    item_list = list()

    for rss_url in rss_urls:
        response = requests.get(rss_url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            items = soup.find_all('item')        
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
        story_items = soup.find('div', class_='caas-body')
        readable_text = ""
        for item in story_items:
            readable_text += item.get_text(separator='\n', strip=True)   
        return readable_text
    else:
        print(f"Error response status code : {response.status_code}")
        return None



YAHOO_F_DATA_FILE_NAME_CSV = "scrapers/yahoo_finance_sentiment.csv"

def generate_data_set_csv(item_list):
    import csv,os
    if not os.path.exists(YAHOO_F_DATA_FILE_NAME_CSV):
        # Create the file if it doesn't exist
        with open(YAHOO_F_DATA_FILE_NAME_CSV, 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['title', 'link', 'date','sentiment'])

    with open(YAHOO_F_DATA_FILE_NAME_CSV, 'a') as csvfile:
        writer = csv.writer(csvfile)

        for row in item_list:
            a_row = row + ('neutral',)
            writer.writerow(a_row)    


from datetime import datetime
def get_YYYY_MM_DD_yh(date_string:str) -> datetime :
    date_object = datetime.strptime(date_string, '%Y-%m-%dT%H:%M:%SZ')
    print(date_object)
    try:        
        return date_object.strftime("%Y-%m-%d")
    except ValueError:
        raise ValueError("Invalid date format")
    
 
import unittest
import logging  
import pandas as pd
   
class TestYahoof(unittest.TestCase):
    
    def test_rss(self):
        import os

        print("Start...")
        top_news = get_yahoo_finance_news_rss()
       
        for n in top_news:
            print("------------ " + n[0])
            get_news_text(n[1])

        self.assertGreater(len(top_news),0) 
        generate_data_set_csv (top_news)
        self.assertEqual(os.path.exists(YAHOO_F_DATA_FILE_NAME_CSV),True)
         


if __name__ == "__main__":
    unittest.main()