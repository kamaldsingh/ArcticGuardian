from helpers import *


## OPTIMISATION HELPERS
def portfolio_annualised_performance(weights, mean_returns, cov_matrix):
    returns = np.sum(mean_returns*weights ) *252
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    return std, returns

def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate, sp500_weight_norm, climate_risk_norm, ESGRisk):
    p_var, p_ret = portfolio_annualised_performance(weights, mean_returns, cov_matrix)
    if(ESGRisk):
        # print(f'ret: {-(p_ret - risk_free_rate)}')
        # print(f'climate: {np.sum(weights*climate_risk_norm)}')
        obj_func = (-(p_ret - risk_free_rate)) + (np.sum(climate_risk_norm)) #+ np.sum(np.abs(weights - sp500_weight_norm))
        # print(f'obj: {obj_func}')
    else:  
        obj_func = (-(p_ret - risk_free_rate)) #+ np.sum(np.abs(weights - sp500_weight_norm))
    return obj_func

def max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate, sp500_weight_norm, climate_risk_norm, ESGRisk):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_free_rate, sp500_weight_norm, climate_risk_norm, ESGRisk)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (0.5/num_assets,2/num_assets)
    bounds = tuple(bound for asset in range(num_assets))
    result = sco.minimize(neg_sharpe_ratio, num_assets*[1./num_assets,], args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)  
    return result

def trade_reco(row):
        if(row.optimised_weight-row.sp500_weight_norm > 0):
            return "Overweigh"
        else:
            return "Underweigh"
        
def returns_table_for_opti(df):
    # pd.DataFrame(yf.Ticker('AMZN').fast_info.last_price).reset_index()
    # yf.Ticker('AMZN').fast_info.last_price

    current_date = datetime.date.today()
    six_months_ago = current_date - datetime.timedelta(days=6*30)
    all_tic_prices = []
    for tic in df.ticker.unique():
        stock_prices = pd.DataFrame(yf.Ticker(tic).history(start=six_months_ago.strftime("%Y-%m-%d"),  
                                        end=current_date.strftime("%Y-%m-%d"))).reset_index()
        stock_prices['ticker'] = tic
        stock_prices.Date = pd.to_datetime(stock_prices.Date,utc=True)  
        all_tic_prices.append(stock_prices[['Date','ticker','Close']])

    all_tic_prices_df = pd.concat(all_tic_prices)
    all_tic_prices_df.set_index('Date', inplace=True)
    table = all_tic_prices_df.pivot(columns='ticker')
    return(table)


# def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
#     p_var, p_ret = portfolio_annualised_performance(weights, mean_returns, cov_matrix)
#     return -(p_ret - risk_free_rate) / p_var

# def max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate):
#     num_assets = len(mean_returns)
#     args = (mean_returns, cov_matrix, risk_free_rate)
#     constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
#     bound = (0.0,1.0)
#     bounds = tuple(bound for asset in range(num_assets))
#     result = sco.minimize(neg_sharpe_ratio, num_assets*[1./num_assets,], args=args,
#                         method='SLSQP', bounds=bounds, constraints=constraints)
#     return result

## EFF Frontier
def efficient_return(mean_returns, cov_matrix, target):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)

    def portfolio_return(weights):
        return portfolio_annualised_performance(weights, mean_returns, cov_matrix)[1]

    constraints = ({'type': 'eq', 'fun': lambda x: portfolio_return(x) - target},
                   {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0,1) for asset in range(num_assets))
    result = sco.minimize(portfolio_volatility, num_assets*[1./num_assets,], args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    return result


def efficient_frontier(mean_returns, cov_matrix, returns_range):
    efficients = []
    for ret in returns_range:
        efficients.append(efficient_return(mean_returns, cov_matrix, ret))
    return efficients

## Min vol


def portfolio_volatility(weights, mean_returns, cov_matrix):
    return portfolio_annualised_performance(weights, mean_returns, cov_matrix)[0]

def min_variance(mean_returns, cov_matrix):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (0.0,1.0)
    bounds = tuple(bound for asset in range(num_assets))

    result = sco.minimize(portfolio_volatility, num_assets*[1./num_assets,], args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)

    return result



# Simulation
def random_portfolios(num_portfolios, mean_returns, cov_matrix, risk_free_rate):
    results = np.zeros((3,num_portfolios))
    weights_record = []
    for i in range(num_portfolios):
        weights = np.random.random(4)
        weights /= np.sum(weights)
        weights_record.append(weights)
        portfolio_std_dev, portfolio_return = portfolio_annualised_performance(weights, mean_returns, cov_matrix)
        results[0,i] = portfolio_std_dev
        results[1,i] = portfolio_return
        results[2,i] = (portfolio_return - risk_free_rate) / portfolio_std_dev
    return results, weights_record
    
def display_simulated_ef_with_random(mean_returns, cov_matrix, num_portfolios, risk_free_rate):
    results, weights = random_portfolios(num_portfolios,mean_returns, cov_matrix, risk_free_rate)
    
    max_sharpe_idx = np.argmax(results[2])
    sdp, rp = results[0,max_sharpe_idx], results[1,max_sharpe_idx]
    max_sharpe_allocation = pd.DataFrame(weights[max_sharpe_idx],index=table.columns,columns=['allocation'])
    max_sharpe_allocation.allocation = [round(i*100,2)for i in max_sharpe_allocation.allocation]
    max_sharpe_allocation = max_sharpe_allocation.T
    
    min_vol_idx = np.argmin(results[0])
    sdp_min, rp_min = results[0,min_vol_idx], results[1,min_vol_idx]
    min_vol_allocation = pd.DataFrame(weights[min_vol_idx],index=table.columns,columns=['allocation'])
    min_vol_allocation.allocation = [round(i*100,2)for i in min_vol_allocation.allocation]
    min_vol_allocation = min_vol_allocation.T
    
    print("-"*80)
    print("Maximum Sharpe Ratio Portfolio Allocation\n")
    print("Annualised Return:", round(rp,2))
    print("Annualised Volatility:", round(sdp,2))
    print(max_sharpe_allocation)
    print("-"*80)

    # print "Minimum Volatility Portfolio Allocation\n"
    # print "Annualised Return:", round(rp_min,2)
    # print "Annualised Volatility:", round(sdp_min,2)
    # print "\n"
    # print min_vol_allocation
    
    plt.figure(figsize=(10, 7))
    plt.scatter(results[0,:],results[1,:],c=results[2,:],cmap='YlGnBu', marker='o', s=10, alpha=0.3)
    plt.colorbar()
    plt.scatter(sdp,rp,marker='*',color='r',s=500, label='Maximum Sharpe ratio')
    plt.scatter(sdp_min,rp_min,marker='*',color='g',s=500, label='Minimum volatility')
    plt.title('Simulated Portfolio Optimization based on Efficient Frontier')
    plt.xlabel('annualised volatility')
    plt.ylabel('annualised returns')
    plt.legend(labelspacing=0.8)
