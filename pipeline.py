import os
import pandas as pd
import yfinance as yf
from datetime import date
from sklearn.ensemble import IsolationForest
import pickle
from scipy.stats import norm
from dagster import asset, Definitions, define_asset_job, AssetSelection, ScheduleDefinition, DefaultScheduleStatus
from scores_calculations import annualized_return, max_drawdown, annualized_volatility, annualized_sharpe_ratio, calmar_ratio, information_coefficient, hit_rate

OUTDIR = '/app/data'
TICKERS_LOAD_URL: str = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
START_DATE = '2017-01-01'

def get_date_path(today_date=None):
    if today_date is None:
        today_date = str(date.today())
    return f'{OUTDIR}/{today_date}'

def set_today_directory(today_path):
    if not os.path.exists( f'{today_path}'):
        os.mkdir( f'{today_path}')
        os.mkdir(f'{today_path}/raw_data_tickers')

@asset(group_name='strategy_calc')
def load_tickers():
    # Giving bias with this small universe 
    set_today_directory(get_date_path())
    tables = pd.read_html(TICKERS_LOAD_URL)
    sp500_table = tables[0]
    tickers = sp500_table['Symbol'].tolist()
    with open(f'{get_date_path()}/tickers_list.pkl', 'wb') as fp:
        pickle.dump(tickers, fp)

@asset(deps=[load_tickers],group_name='strategy_calc')
def load_ticker_data():
    tickers_list = []
    with open (f'{get_date_path()}/tickers_list.pkl', 'rb') as fp:
        tickers_list = pickle.load(fp)
    #TODO: could be parallelized  
    for ticker in tickers_list:
        data = yf.download(ticker, start=START_DATE)
        data['ticker'] = ticker
        data.reset_index().to_parquet(f'{get_date_path()}/raw_data_tickers/{ticker}.parquet')


@asset(deps=[load_ticker_data],group_name='strategy_calc')
def combine_data():
    tickers_list = []
    with open (f'{get_date_path()}/tickers_list.pkl', 'rb') as fp:
        tickers_list = pickle.load(fp)
    dfs = []
    for ticker in tickers_list:
        df = pd.read_parquet(f'{get_date_path()}/raw_data_tickers/{ticker}.parquet')
        dfs.append(df)
    combined_data = pd.concat(dfs,axis=0).reset_index().rename(str.lower, axis='columns').set_index(['date','ticker']).sort_index().drop(columns=['index'])
    combined_data.reset_index().to_parquet(f'{get_date_path()}/combined_data.parquet')

@asset(deps=[combine_data],group_name='strategy_calc')
def clean_data():
    data = pd.read_parquet(f'{get_date_path()}/combined_data.parquet').set_index(['date','ticker'])
    isolation_forest = IsolationForest(contamination=0.01, random_state=42)
    predictions = isolation_forest.fit_predict(data)
    cleaned_data = data[predictions == 1].copy()
    cleaned_data['target_1d'] = cleaned_data['adj close'].groupby('ticker').pct_change()
    cleaned_data.dropna().reset_index().to_parquet(f'{get_date_path()}/combined_cleaned_data.parquet')

@asset(deps=[clean_data],group_name='strategy_calc')
def calc_strategy():
    # https://www.msci.com/documents/1296102/8473352/Volatility-brochure.pdf/f9cac8cb-f467-470d-9292-0298a597799e
    data = pd.read_parquet(f'{get_date_path()}/combined_cleaned_data.parquet').set_index(['date','ticker'])
    data['return_1d'] = data['target_1d'].groupby('ticker').shift(1)
    data['volatilty_1y'] = data['return_1d'].groupby('ticker').ewm(halflife=126).std().droplevel(0)
    data['vol_perc'] = data['volatilty_1y'].groupby('date').rank(pct=True).dropna()
    data['alpha'] = data.groupby('date')['vol_perc'].transform(lambda x: norm.ppf(x.clip(lower=0.000001, upper=0.999999)))
    data[['alpha','target_1d']].dropna().reset_index().to_parquet(f'{get_date_path()}/res_data.parquet')    

@asset(deps=[calc_strategy],group_name='strategy_calc')
def calc_scores():
    data = pd.read_parquet(f'{get_date_path()}/res_data.parquet').set_index(['date','ticker'])
    data['alpha_norm'] = data.groupby('date')['alpha'].transform(lambda x: x - x.mean())
    data['alpha_norm'] = data['alpha_norm'].groupby('date').transform(lambda x: x * (2 / x.abs().sum()))
    
    res_df = pd.DataFrame(((data['alpha_norm'] * data['target_1d']).groupby('date').sum()),columns=['returns'])
    res_df['pnl'] = (1 + res_df['returns'].cumsum())
    res_df.dropna().reset_index().to_parquet(f'{get_date_path()}/res_rets.parquet')
    scores = {}
    scores['sharpe'] = annualized_sharpe_ratio(res_df['returns'])
    scores['max_dd'] = max_drawdown(res_df['returns'])
    scores['volatility'] = annualized_volatility(res_df['returns'])
    scores['ann_returns'] = annualized_return(res_df['returns'])
    scores['calmar_ratio'] = calmar_ratio(res_df['returns'])
    scores['information_coefficient']  = information_coefficient(data['alpha_norm'],data['target_1d'])
    scores['hit_rate']  = hit_rate(data['alpha_norm'],data['target_1d'])
    scores_df = pd.DataFrame.from_dict(scores, orient='index',columns=['score'])
    scores_df.dropna().reset_index().to_parquet(f'{get_date_path()}/scores.parquet')    


strategy_calc_job = define_asset_job(
    "strategy_calc_job", AssetSelection.groups("strategy_calc")
)

trading_sys_schedule = ScheduleDefinition(
    job=strategy_calc_job,
    cron_schedule="30 5 * * 1-5",
    default_status=DefaultScheduleStatus.RUNNING,
)

defs = Definitions(
    assets=[load_tickers, load_ticker_data, combine_data, clean_data, calc_strategy, calc_scores],
    jobs=[strategy_calc_job],
    schedules=[trading_sys_schedule]
)

