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

        


 
import unittest
import logging  
import pandas as pd
from scrapers.yahoof_scraper import get_yahoo_finance_news_rss,get_news_text
   
class TestSentimentModel(unittest.TestCase):
    
    '''
    def test_fine_tune(self):
        sm = SentimentModel(fine_tune=True, training_model = 'FinGPT')
        self.assertNotEqual(sm,None)
    
    '''
    def test_score(self):
        print("Start...")
        top_news = get_yahoo_finance_news_rss()
        sm = SentimentModel()
        acc_score = 0
        counter   = 0
        for n in top_news:
            news_text = get_news_text(n[1])
            if(news_text is not None):
                score = sm.get_sentiment_score(news_text)
                print(f"------------ news {n[0]:s} , score = {score:.6f}")
                acc_score += score
                counter += 1
                self.assertGreater(score,0) 
        print("Accumulated score {:.5f}".format((acc_score/counter)))


            

if __name__ == "__main__":
    unittest.main()