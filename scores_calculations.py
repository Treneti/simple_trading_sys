import numpy as np
import pandas as pd
from scipy.stats import spearmanr

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

def omega_ratio(returns, threshold=0):
    returns = np.sort(returns)
    excess_returns_above = returns[returns > threshold] - threshold
    excess_returns_below = threshold - returns[returns <= threshold]
    sum_above = np.sum(excess_returns_above)
    sum_below = np.sum(excess_returns_below)
    
    if sum_below == 0:
        return np.inf 
    
    return sum_above / sum_below

def calculate_scores(res_df,data):
    scores = {}
    scores['sharpe'] = annualized_sharpe_ratio(res_df['returns'])
    scores['max_dd'] = max_drawdown(res_df['returns'])
    scores['volatility'] = annualized_volatility(res_df['returns'])
    scores['ann_returns'] = annualized_return(res_df['returns'])
    scores['calmar_ratio'] = calmar_ratio(res_df['returns'])
    scores['omega_ratio'] = omega_ratio(res_df['returns'])
    
    scores['information_coefficient']  = information_coefficient(data['alpha_norm'],data['target_1d'])
    scores['hit_rate']  = hit_rate(data['alpha_norm'],data['target_1d'])
    scores_df = pd.DataFrame.from_dict(scores, orient='index',columns=['score'])
    return scores_df