import os 
import pandas as pd
import yfinance as yf
from dagster import Config
from sklearn.ensemble import IsolationForest
from datetime import date
import pickle

class TSConfig():
    tickers_load_url: str = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    outdir: str = '/app/data'
    raw_tickers_path: str = ''
    start_date: str = "2017-01-01"  

def load_tickers(config):
    tables = pd.read_html(config.tickers_load_url)
    sp500_table = tables[0]
    tickers = sp500_table['Symbol'].tolist()
    with open(f'{config.outdir}/tickers_list.pkl', 'wb') as fp:
        pickle.dump(tickers, fp)

def load_ticker_data(config):
    tickers_list = []
    with open (f'{config.outdir}/tickers_list.pkl', 'rb') as fp:
        tickers_list = pickle.load(fp)
    #TODO: could be parallelized  
    for ticker in tickers_list:
        data = yf.download(ticker, start=config.start_date)
        data['ticker'] = ticker
        data.reset_index().to_parquet(f'{config.raw_tickers_path}/{ticker}.parquet')


def combine_data(config):
    tickers_list = []
    with open (f'{config.outdir}/tickers_list.pkl', 'rb') as fp:
        tickers_list = pickle.load(fp)
    dfs = []
    for ticker in tickers_list:
        df = pd.read_parquet(f'{config.raw_tickers_path}/{ticker}.parquet')
        dfs.append(df)
    combined_data = pd.concat(dfs,axis=0).reset_index().rename(str.lower, axis='columns').set_index(['date','ticker']).sort_index().drop(columns=['index'])
    combined_data.reset_index().to_parquet(f'{config.outdir}/combined_data.parquet')

def clean_data(config):
    data = pd.read_parquet(f'{config.outdir}/combined_data.parquet').set_index(['date','ticker'])
    isolation_forest = IsolationForest(contamination=0.01, random_state=42)
    predictions = isolation_forest.fit_predict(data)
    cleaned_data = data[predictions == 1].copy()
    cleaned_data['target_1d'] = cleaned_data['adj close'].groupby('ticker').pct_change()
    cleaned_data.reset_index().to_parquet(f'{config.outdir}/combined_cleaned_data.parquet')

def calc_features(config):
    pass

def train_models(config):
    pass

def build_preds(config):
    pass

def main(config):
    config = TSConfig()

    today = str(date.today())   
    today_directory = f'{config.outdir}/{today}'
    if not os.path.exists(today_directory):
        os.mkdir(today_directory)
        config.outdir = today_directory
        os.mkdir(f'{today_directory}/raw_data_tickers')
        config.raw_tickers_path = f'{today_directory}/raw_data_tickers'
