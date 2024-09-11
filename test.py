import os 
import pandas as pd
import yfinance as yf
import numpy as np
from dagster import Config
from datetime import date
from sklearn.ensemble import IsolationForest
import pickle
from scipy.stats import norm, spearmanr

class TSConfig():
    tickers_load_url: str = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    outdir: str = '/app/data'
    raw_tickers_path: str = ''
    start_date: str = "2017-01-01"  

def annualized_sharpe_ratio(returns, risk_free_rate=0, trading_days=252):
    excess_returns = returns - risk_free_rate
    sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns, ddof=1)
    return sharpe_ratio * np.sqrt(trading_days)

def max_drawdown(returns):
    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    return drawdown.min()

def information_coefficient(predicted_returns, actual_returns):
    return spearmanr(predicted_returns, actual_returns)[0]

def hit_rate(predicted_returns, actual_returns):
    return np.mean(np.sign(predicted_returns) == np.sign(actual_returns))

def annualized_volatility(returns, trading_days=252):
    return np.std(returns) * np.sqrt(trading_days)

def annualized_return(returns, trading_days=252):
    cumulative_return = (1 + returns).prod()
    return cumulative_return**(trading_days / len(returns)) - 1

def calmar_ratio(returns, trading_days=252):
    ann_return = annualized_return(returns, trading_days)
    max_dd = max_drawdown(returns)
    return ann_return / abs(max_dd) if max_dd != 0 else np.nan

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
    cleaned_data.dropna().reset_index().to_parquet(f'{config.outdir}/combined_cleaned_data.parquet')

def calc_strategy(config):
    # https://www.msci.com/documents/1296102/8473352/Volatility-brochure.pdf/f9cac8cb-f467-470d-9292-0298a597799e
    data = pd.read_parquet(f'{config.outdir}/combined_cleaned_data.parquet').set_index(['date','ticker'])
    data['return_1d'] = data['target_1d'].groupby('ticker').shift(1)
    data['volatilty_1y'] = data['return_1d'].groupby('ticker').ewm(halflife=126).std().droplevel(0)
    data['vol_perc'] = data['volatilty_1y'].groupby('date').rank(pct=True).dropna()
    data['alpha'] = data.groupby('date')['vol_perc'].transform(lambda x: norm.ppf(x.clip(lower=0.000001, upper=0.999999)))
    data[['alpha','target_1d']].dropna().reset_index().to_parquet(f'{config.outdir}/res_data.parquet')    

def calc_scores(config):
    data = pd.read_parquet(f'{config.outdir}/res_data.parquet').set_index(['date','ticker'])
    data['alpha_norm'] = data.groupby('date')['alpha'].transform(lambda x: x - x.mean())
    data['alpha_norm'] = data['alpha_norm'].groupby('date').transform(lambda x: x * (2 / x.abs().sum()))
    
    res_df = pd.DataFrame(((data['alpha_norm'] * data['target_1d']).groupby('date').sum()),columns=['returns'])
    res_df['pnl'] = (1 + res_df['returns'].cumsum())
    res_df.dropna().reset_index().to_parquet(f'{config.outdir}/res_rets.parquet')
    scores = {}
    scores['sharpe'] = annualized_sharpe_ratio(res_df['returns'])
    scores['max_dd'] = max_drawdown(res_df['returns'])
    scores['volatility'] = annualized_volatility(res_df['returns'])
    scores['ann_returns'] = annualized_return(res_df['returns'])
    scores['calmar_ratio'] = calmar_ratio(res_df['returns'])
    scores['information_coefficient']  = information_coefficient(data['alpha_norm'],data['target_1d'])
    scores['hit_rate']  = hit_rate(data['alpha_norm'],data['target_1d'])
    scores_df = pd.DataFrame.from_dict(scores, orient='index',columns=['score'])
    scores_df.dropna().reset_index().to_parquet(f'{config.outdir}/scores.parquet')    

def main(config):
    config = TSConfig()

    today = str(date.today())   
    today_directory = f'{config.outdir}/{today}'
    if not os.path.exists(today_directory):
        os.mkdir(today_directory)
        config.outdir = today_directory
        os.mkdir(f'{today_directory}/raw_data_tickers')
        config.raw_tickers_path = f'{today_directory}/raw_data_tickers'
