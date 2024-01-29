from transformers import DistilBertTokenizer, DistilBertForSequenceClassification,AdamW
import torch

class TrainingDataSet(object):

    def __init__(self, training_model = None) -> None:
        self.training_model = training_model
    
    def get_training_model(self):
        from datasets import load_dataset
        dataset = None
        if (self.training_model == "eengel7" ):
            dataset = load_dataset("eengel7/sentiment_analysis_training")
        elif (self.training_model == "FinGPT" ):
            dataset = load_dataset("FinGPT/fingpt-sentiment-train")
        return dataset["train"]


    def get_training_model_keys(self):
        from datasets import load_dataset
        kstr  = None
        if (self.training_model == "eengel7" ):
            kstr = 'Headline_string'
        elif (self.training_model == "FinGPT" ):
            kstr = 'input'
        return kstr
    

    def get_training_model_values(self):
        from datasets import load_dataset
        kstr  = None
        if (self.training_model == "eengel7" ):
            kstr = 'Sentiment',3
        elif (self.training_model == "FinGPT" ):
            kstr = 'output',9
        return kstr
    
    
    def string2int_value_mapping(self, string_value: str) -> str: 
         if (self.training_model == "eengel7" ):
             return int(string_value)
         elif (self.training_model == "FinGPT" ):
             string_to_string_dict = {
                                        'strong negative': 0,
                                        'negative': 1,
                                        'moderately negative': 2,
                                        'mildly negative': 3,
                                        'neutral': 4,
                                        'moderately positive': 5,
                                        'mildly positive': 6,
                                        'positive': 7,
                                        'strong positive': 8
                                        }
             return string_to_string_dict[string_value]
             


class SentimentModel(object):

    def __init__(self, fine_tune : bool = False, training_model = None) -> None:
        # Load pre-trained sentiment analysis model and tokenizer
       
        if (fine_tune):
                tm = TrainingDataSet(training_model = training_model)
                field,nlabels = tm.get_training_model_values()
                self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', num_labels=nlabels)
                self.sentiment_model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=nlabels)
                self.__fine_tune(tm)
        else:
            self.tokenizer = DistilBertTokenizer.from_pretrained('fine_tuned_sentiment_model')
            self.sentiment_model = DistilBertForSequenceClassification.from_pretrained('fine_tuned_sentiment_model')



    def __fine_tune(self, training_model = None):
        # Fine-tune the model on your sentiment analysis dataset
        optimizer = AdamW(self.sentiment_model.parameters(), lr=1e-5)
        num_epochs = 3
        

        train_data = training_model.get_training_model()
        if train_data == None:
            raise Exception("Training model is not valid")
        
        batch_keys   = training_model.get_training_model_keys()
        batch_values = training_model.get_training_model_values()
      

        for epoch in range(num_epochs):
            bindex = 0
            for batch in train_data:
                # Tokenize the text and convert labels to PyTorch tensor
                #print(batch)
                inputs = self.tokenizer(batch[batch_keys], return_tensors='pt', truncation=True, padding=True)
                #print(training_model.string2int_value_mapping(str(batch[batch_values[0]])))
                labels = torch.tensor(training_model.string2int_value_mapping(str(batch[batch_values[0]])))
    
                # Forward pass to get the logits
                outputs = self.sentiment_model(**inputs, labels=labels)
                loss = outputs.loss

                # Backward pass and optimization
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                bindex += 1
                if(bindex % 100  == 0):
                    print (" ----> " + str(bindex))

        # Save the fine-tuned model if needed
        self.sentiment_model.save_pretrained("fine_tuned_sentiment_model")
        self.tokenizer.save_pretrained("fine_tuned_sentiment_model")


    def get_sentiment_score(self,news_text):
   
        self.sentiment_model.eval()  # Set the model to evaluation mode

        # Use the 'text' parameter for tokenization

        inputs = self.tokenizer(text=news_text, return_tensors='pt', truncation=True, padding=True)
        # Specify 'labels' to suppress the warning
        labels = torch.tensor([1]).unsqueeze(0)  # Assuming 1 corresponds to positive sentiment
        outputs = self.sentiment_model(**inputs, labels=labels)

        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        #print(logits)
        #print(probabilities)
        import numpy as np
        sentiment_score =  torch.argmax( probabilities[0] ) / len(probabilities[0])  # Assuming 1 corresponds to positive sentiment
        return sentiment_score


        
from  price_estimator.const_and_utils import FOLDER_MARKET_DATA

NEWS_DATA_FILE_NAME_CSV  =  'news_scores.csv'

def save_to_csv(data):
    import csv,os,pandas as pd
    data = data.encode('ascii', 'ignore').decode()
    print("Saving --> " + data)
   
    news_df = pd.DataFrame()
    if not os.path.exists(NEWS_DATA_FILE_NAME_CSV):
        # Create the file if it doesn't exist
        with open(NEWS_DATA_FILE_NAME_CSV, 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['src', 'news', 'Date', 'Scores', "news_text"])
    else: 
        news_df = pd.read_csv(NEWS_DATA_FILE_NAME_CSV)
        
    # !!! this is very inefficient !!!. We should add an md5 as Key or however a unique numeric id to the news.
    # I'm planning to store into a db going forward. For the moment this is fine.   
    with open(NEWS_DATA_FILE_NAME_CSV, mode='a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        src, news, date, score, news_text = data.split('|')
        if ( news_df.empty 
            or 
            (not news_df.empty and news_df[news_df['news'].apply(lambda nt: nt == news.split(':')[1])].empty)
            ):
             writer.writerow([src.split(':')[1],news.split(':')[1], date.split(':')[1], score.split(':')[1],news_text.split(':', 1)[1]])
        else: 
            print("Duplicated news : {}".format(news.split(':')[1]))

    dfcopy = pd.read_csv(NEWS_DATA_FILE_NAME_CSV)
    # copy the updated file to the data folder as well. 
    dfcopy.to_csv(FOLDER_MARKET_DATA+NEWS_DATA_FILE_NAME_CSV)

def save_news_text(news_text:str, sm: SentimentModel) -> float: 
    news_text.replace("|","-")
    score = sm.get_sentiment_score(news_text)
    ns = f"src:Investing.com|news:{n[0]:s}|Date:{get_YYYY_MM_DD_idc(n[2]):s}|Scores:{score:.6f}|news_text:{news_text:s}"
    save_to_csv(ns)
    return score



import unittest
import logging  
import pandas as pd
from scrapers.yahoof_scraper    import get_yahoo_finance_news_rss,get_news_text
from scrapers.investing_dot_com import get_investing_dot_com_news_rss, get_news_text_selenium, get_YYYY_MM_DD_idc
   
class TestSentimentModel(unittest.TestCase):
    
    '''
    # this is commented as it redefine the calibration. 
    def test_fine_tune(self):
        sm = SentimentModel(fine_tune=True, training_model = 'eengel7')
        self.assertNotEqual(sm,None)
    
    '''

    def test_score_yahoo(self):
        print("Start Yahoo ...")
        top_news = get_yahoo_finance_news_rss()
        acc_score = 0
        counter   = 0
        for n in top_news:
            news_text = get_news_text(n[1])
            if(news_text is not None):
                try:
                    score = save_news_text(news_text,sm)
                    acc_score += score
                    counter += 1
                    self.assertGreater(score,0) 
                except:
                    continue
        print("Accumulated score {:.5f}".format((acc_score/counter)))
        

    def test_score_investing(self):
        print("Start Investing.com ...")
        top_news = get_investing_dot_com_news_rss()
        acc_score = 0
        counter   = 0
        for n in top_news:
            news_text = get_news_text_selenium(n[1])
            if(news_text is not None):
                try:
                    score = save_news_text(news_text,sm)
                    acc_score += score
                    counter += 1
                    self.assertGreater(score,0) 
                except:
                    continue
        print("Accumulated score {:.5f}".format((acc_score/counter)))




if __name__ == "__main__":

    import sys
    if (len(sys.argv) > 2) :

        if (sys.argv[2] == "test_unit"):
            unittest.main()
        elif(sys.argv[2] == "fine_tune"):
            sm = SentimentModel(fine_tune=True, training_model = 'FinGPT')
        else:
            print("{} is not a valid value".format(sys.argv[2]))

    else:

        sm = SentimentModel()
        acc_score = 0
        counter   = 0
        
        print("Start Yahoo ...")
        top_news = get_yahoo_finance_news_rss()
        for n in top_news:
            news_text = get_news_text(n[1])
            if(news_text is not None):
                try:
                    score = save_news_text(news_text,sm)
                    acc_score += score
                    counter += 1
                except:
                    continue
        
        
        print("Start Investing.com ...")
        top_news = get_investing_dot_com_news_rss()
        for n in top_news:
            news_text = get_news_text_selenium(n[1])
            if(news_text is not None):
                try:
                    score = save_news_text(news_text,sm)
                    acc_score += score
                    counter += 1
                except:
                    continue

        print("Accumulated score {:.5f}".format((acc_score/counter)))
       
        #import os
        #os.system('cp {} {}'.format(NEWS_DATA_FILE_NAME_CSV,FOLDER_MARKET_DATA+NEWS_DATA_FILE_NAME_CSV))
