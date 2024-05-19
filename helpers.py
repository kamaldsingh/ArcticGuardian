import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import requests
import scipy.optimize as sco

from pandas_datareader import data as pdr
import yfinance as yf
yf.pdr_override() 

from transformers import BertTokenizer, BertForSequenceClassification, pipeline

import os
from datetime import datetime, date
import datetime

import itertools
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)

from dateutil import parser
import torch
from transformers import pipeline

pd.options.display.max_columns = 999
import gc
import re
from wordcloud import WordCloud, STOPWORDS
from os.path import exists

stopwords = set(STOPWORDS)

from finvizfinance.quote import finvizfinance
from dotenv import load_dotenv
import replicate

finbert_esg = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-esg',num_labels=4)
tokenizer_esg = BertTokenizer.from_pretrained('yiyanghkust/finbert-esg')
esg_label_pip = pipeline("text-classification", model=finbert_esg, tokenizer=tokenizer_esg)

finbert_sentiment = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone',num_labels=3)
tokenizer_sentiment = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
sentiment_pipeline = pipeline("text-classification", model=finbert_sentiment, tokenizer=tokenizer_sentiment)

def parse_dt(s):
    try:
        return(parser.parse(str(s)))
    except:
        return(np.nan)

def get_esg_label_transcript(tr):
    sent_label_scores = []

    for sent in sent_tokenize(tr):
        all_esg_labels = esg_label_pip(sent)
        non_none_labels = [x for x in all_esg_labels if x['label']!='None']
        if(len(non_none_labels)>0):
            sent_label_scores.append([non_none_labels[0]['label'],non_none_labels[0]['score'],sent])
    df = pd.DataFrame(sent_label_scores, columns=['esg_label', 'label_score', 'sent'])
    return(df)

def create_sentiment_output(all_labels):
    non_none_labels = [x for x in all_labels if x['label']!='None']
    if(len(non_none_labels)>0):
        label = non_none_labels[0]['label']
        score = non_none_labels[0]['score']
        sentiment = 0
        if(label=='Positive'):
            return(1*score)
        elif(label=='Negative'):
            return(-1*score)
        else:
            return 0
    else:
        return 0
    

def calc_returns(df):
    df['daily_return'] = 100 * (df['Close'].pct_change())
    return(df)

def get_monthly_ret(df):
    df.set_index('Date',inplace=True)

    # # Daily return 
    # df['daily_return'] = df.Close.pct_change()*100
    newdf = df['daily_return'].resample('M').ffill().pct_change().reset_index()
    return(newdf)




# stock_prices = pd.read_csv('./returns/AAPL.csv')
# # stock_prices.drop(columns='Close',inplace=True)
# stock_prices.Date = pd.to_datetime(stock_prices.Date,utc=True)

# stock_prices = stock_prices.groupby(['ticker']).apply(calc_returns)
# .droplevel(0).reset_index().drop(columns='index')

# monthly_ret = stock_prices.drop(columns=['Open','High','Low','Volume','Stock Splits']).copy()
# monthly_ret = monthly_ret.groupby('ticker').apply(get_monthly_ret).reset_index().rename(columns={'daily_return':'monthly_return'})\
#                             .drop(columns='level_1')
# stock_prices.daily_return.sum()

# stock_prices = stock_prices.merge(sp500_tickers[['Symbol','GICS Sector','GICS Sub-Industry']], left_on = 'ticker', right_on='Symbol', how='left')