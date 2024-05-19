# %%

from opti_helpers import *
from helpers import *
import streamlit as st 
import pathlib

st.set_page_config(layout="wide")

# df = pd.read_csv('./data/esg_returns_news_100.csv')
df = pd.read_csv('./data/complete_data_arctic_risk_returns_100.csv')
sp_weights = pd.read_html(requests.get('https://www.slickcharts.com/sp500',
                      headers={'User-agent': 'Mozilla/5.0'}).text)[0]

sp_weights.rename(columns={'Portfolio%':'sp500_weight','Symbol':'ticker'}, inplace=True)
sp_weights.sp500_weight = sp_weights.sp500_weight.apply(lambda x: str(x).replace('%', '')).astype(float)

df = df.merge(sp_weights[['ticker','sp500_weight','Company']], on='ticker', how='left')
df['sp500_weight_norm'] = (df.sp500_weight/df.sp500_weight.sum())
df = df.sort_values('sp500_weight', ascending=False).reset_index(  drop=True)
df['climate_risk_norm'] = df.climate_risk_count/np.max(df.climate_risk_count)

# %%

def print_example_df(idx = 15):
    print(f"ticker: {df.iloc[idx].ticker} with weight: {df.iloc[idx].sp500_weight} ")
    print(df.iloc[idx].climate_risks)

# print_example_df(15)
# print_example_df(16)
# print_example_df(12)
# %%

num_portfolios = 2500
risk_free_rate = 0.02

# print(df.head(2))

# %%
# @st.cache
def optimize_portfolio(max_stocks, MarketCap, ESGRisk):
    if MarketCap:
        samp_tics = df.sort_values('sp500_weight', ascending=False).ticker.head(max_stocks).tolist()
    else:
        samp_tics = df.ticker.sample(n=max_stocks, random_state=42)
    finaldf = df[df.ticker.isin(samp_tics)]
    finaldf['sp500_weight_norm'] = (finaldf.sp500_weight/finaldf.sp500_weight.sum())*100
    table = returns_table_for_opti(finaldf)
    returns = table.pct_change()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()

    max_sharpe = max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate, finaldf.sp500_weight_norm.tolist(), finaldf.climate_risk_norm.tolist(), ESGRisk)
    sdp, rp = portfolio_annualised_performance(max_sharpe['x'], mean_returns, cov_matrix)
    max_sharpe_allocation = pd.DataFrame(max_sharpe.x,index=table.columns,columns=['allocation'])
    max_sharpe_allocation.allocation = [round(i*100,2)for i in max_sharpe_allocation.allocation]
    max_sharpe_allocation = max_sharpe_allocation.T
    opti_df = max_sharpe_allocation.T.reset_index()
    opti_df = opti_df[['ticker','allocation']].rename(columns={'allocation':'optimised_weight'})

    min_vol = min_variance(mean_returns, cov_matrix)
    sdp_min, rp_min = portfolio_annualised_performance(min_vol['x'], mean_returns, cov_matrix)
    min_vol_allocation = pd.DataFrame(min_vol.x,index=table.columns,columns=['allocation'])
    min_vol_allocation.allocation = [round(i*100,2)for i in min_vol_allocation.allocation]
    min_vol_allocation = min_vol_allocation.T
    # an_vol = np.std(returns) * np.sqrt(252)
    an_vol = finaldf.climate_risk_norm.tolist()
    an_rt = mean_returns * 252

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(an_vol,an_rt,marker='o',s=200)
    for i, txt in enumerate(table.columns):
        ax.annotate(txt, (an_vol[i],an_rt[i]), xytext=(10,0), textcoords='offset points')
    ax.scatter(sdp,rp,marker='*',color='r',s=500, label='Maximum Sharpe ratio')
    ax.scatter(sdp_min,rp_min,marker='*',color='g',s=500, label='Current Portfolio')
    target = np.linspace(rp_min, 0.34, 50)
    efficient_portfolios = efficient_frontier(mean_returns, cov_matrix, target)
    ax.plot([p['fun'] for p in efficient_portfolios], target, linestyle='-.', color='black', label='efficient frontier')
    ax.set_title('Portfolio Optimization with Individual Stocks - Climate Risk(x) & Return(y)')
    ax.set_xlabel('Climate Risk')
    ax.set_ylabel('annualised returns')
    ax.legend(labelspacing=0.8)

    df_with_opti = finaldf.merge(opti_df, on='ticker', how='left').sort_values('optimised_weight', ascending=False)

    ### barplot
    opti_esg = np.sum(df_with_opti['climate_risk_count']*(df_with_opti['optimised_weight']/100))
    orig_esg = np.sum(df_with_opti['climate_risk_count']*(df_with_opti['sp500_weight_norm']/100))
    fig2, ax2 = plt.subplots(figsize=(10, 10))
    sns.barplot(x=['Optimised Portfolio','Original Portfolio'], y=[opti_esg, orig_esg], ax=ax2)

    df_with_opti['trade_reco'] = df_with_opti.apply(trade_reco, axis=1)

    df_show = df_with_opti[['ticker','Company','returns_6m','sp500_weight_norm','optimised_weight','climate_risk_norm','trade_reco']]\
                            .sort_values('sp500_weight_norm',ascending=False)

    st.write(df_show)
    st.pyplot(fig)
    st.pyplot(fig2)

    return df_show

max_stocks = st.selectbox('Max Stocks', [5, 10, 100], index = 0)
MarketCap = st.selectbox('Market Cap', [True, False], index=1)
ESGRisk = st.selectbox('ESG Risk', [True, False], index=0)

if st.button('Optimize'):
    result = optimize_portfolio(max_stocks, MarketCap, ESGRisk)

show_example_st = st.checkbox("Show Example Climate Risk Statements")
if show_example_st:
    dfst = pd.read_csv('./data/climate_risks_statements_examples.csv')
    example_sts = np.random.choice(dfst.climate_risks.tolist(), 5)
    # example_sts = df_with_opti.climate_risks.tolist()
    # st_disp = []
    # for x in example_sts:
    #     for i in x.split('###'):
    #         st_disp.append(i)
    # display_st = np.unique(st_disp)
    for l in example_sts:
        st.markdown('*'+l+'*')


st.divider()
st.caption(pathlib.Path("data/disclaimer.txt").read_text())

