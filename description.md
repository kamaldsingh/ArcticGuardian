## Inspiration

Climate Change is perhaps the biggest threat to humanity and the effects of climate change are already disrupting not just our lives but also companies worldwide be it in supply chain issues or financial risks or asset damages. 

```New study shows that 1°C increase in global temperature leads to a 12% decline in world gross domestic product (GDP). Each additional ton of carbon will cost the global economy $1,056. Under a business-as-usual scenario, climate change will cause global welfare losses of 31%. ```

Thus its **crucial that investors protect their investments from adverse impacts of climate change and position their portfolio accordingly**

This tool can be used by financial institutions like banks, asset managers and individual(retail investors) thus its applicability is very broad and it would enable them to manage and optimize their portfolio to maximize returns while minimising climate risks. 

## What it does
This AI tool provides retail or professional Investors the ability to protect their portfolio from the adverse impacts of climate change by analyzing & quantifying the climate risks that their portfolio is subjected to and positioning your portfolio in order to avoid such risk. 
Our model 
1. Collects latest Earning call transcripts. Earnings call transcripts are written records of unrehearsed discussions between company executives & analysts during quarterly conference calls, detailing financial results, strategy, Sustainability and outlook. (These discussions are unrehearsed)
2. Extracts relevant Climate Risk statements with `Snowflake Arctic` 
3. Enrich the above data with Stock Returns over the last 6 months from `yahoo finance` API.  
4. Run a `Scipy` non-linear optimization on Return, Volatility and Climate Risk. The model will attempt to increase the returns, lower overall risk/volatility, decrease the Climate Risk and thus balance and improve all the 3 objectives at the same time. 
5. Select the number of stocks and produce Optimal Investment Strategy and portfolio weights to Guard/protect against Climate Risks 
6. Trade Recommendations ie Overweigh/Buy-Sell recommendation for your current portfolio to position it best for the future. 
7. Visualise where your current portfolio and optimal portfolio lies on the Efficient Frontier and analyze metrics to evaluate if the optimization was successful (like current portfolio risk vs optimal portfolio risk)

To view the plots check out the presentation ArcticGuardian.pdf

## How we built it
1. We collected the earnings call transcripts dataset available from [Kaggle](https://www.kaggle.com/datasets/tpotterer/motley-fool-scraped-earnings-call-transcripts) and limit the universe of (companies) stocks to 100 randomly selected stocks from the `S&P 500` index for now. Earnings call transcripts are written records of discussions between company executives & analysts during quarterly conference calls, detailing financial results, strategy, Sustainability and outlook. `Data_arctic_risk_retuns.ipynb`
2. We use `snowflake-arctic-instruct` LLM via [Replicate](https://replicate.com/snowflake/snowflake-arctic-instruct/api) and create a Prompt to extract all relevant Climate Risks for these randomly sampled 100 companies. Additionally we quantify and count the various Climate Risks a company faces.  Code - `Data_arctic_risk_retuns.ipynb`
3. We enrich the above data with Stock Returns over the last 6 months from `yahoo finance` API. We get `S&P 500` constiutent weights (Note: **when you invest in  any S&P ETF you essentially invest in these constituent weights which might not be optimal for Climate risks** ). Code - `Data_arctic_risk_retuns.ipynb`
4. `Streamlit` app - `app.py` 
	a)We calculate Return, Volatility(Overall portfolio risk), Climate Risk and feed them into our `Scipy` optimization where we add them to the [Sharpe Ratio](https://www.investopedia.com/terms/s/sharperatio.asp)  in order to optimize all 3 metrics simultaneously. Thus our model will produce the optimized weights . The Shapre Ratio optimization approach comes from [Modern Portfolio Theory](https://www.investopedia.com/terms/h/harrymarkowitz.asp) for which Markowitz won the Nobel Prize and we extend the idea to include Climate Risks
	b) In `Streamlit` app  you can select - the number of stocks, whether you want to choose these stocks based on Market Cap or not (if chosen True then you get mega cap stocks like MSFT/ AAPL else its randomly selected), whether you choose to use Climate/ESG Risk in the optimizer or not. 
	b)After that you get the optimal portflio with the optimized Sharpe Ratio and Comparison of Climate Risk - You can see lower risk here - `media/comparison climate risk` 
	c) Dataframe with the the Optimised weights vs S&P default weights that you're using if you invest in S&P 500 ETF directly - `media/table`
	d) You can select more stocks by selecting higher number 10/15/20/100 to create larger portfolios. 

## Challenges we ran into
1. Streamlit in Snowflake doesnt allow py file imports. 

## Accomplishments that we're proud of
1. Out AI tool provides **actionable intelligence** with next best steps for your portfolio to avoid Climate risks. 
2. Novel approach of including `Climate Risk` measured by `AI and LLM models`  applied to Nobel Prize winning Quantitative Finance optimization approaches. 

## What we learned

## What's next for ArcticGuardian

