import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import CAPMautoscrape
#print('statsmodels or scikit')
#package_choice = input()

bitcoin = CAPMautoscrape.CMCscrape('bitcoin')
ethereum = CAPMautoscrape.CMCscrape('ethereum')
ripple = CAPMautoscrape.CMCscrape('ripple')

share = CAPMautoscrape.NASDAQscrape('spy')
s = share.get_table()
df = pd.DataFrame(s)
df.columns = ['Day', 'Open', 'High', 'Low', 'Close', 'Volume', 'NA']

dfb = df[df['Date'].isin(dfs[0].iloc[:,0])].reset_index(drop=True)
monthly_prices = pd.concat([dfb['Close'], dfs[0]['Close']],axis=1)
monthly_prices = monthly_prices.reindex(index=monthly_prices.index[::-1]).reset_index(drop=True)
monthly_prices.columns = ['Bit', '^GSPC']
monthly_returns = monthly_prices.pct_change(1)
clean_monthly_returns = monthly_returns.dropna(axis=0)  # drop first missing row
X = clean_monthly_returns['^GSPC']
y = clean_monthly_returns['Bit']
plt.clf()
plt.style.use('ggplot')
plt.scatter(X,y,c='b')

# Add a constant to the independent value
X1 = sm.add_constant(X)

# make regression model
model = sm.OLS(y, X1)

# fit model and print results
results = model.fit()
print(results.summary())

def run_model(asset):
    benchmark = CAPMautoscrape.NASDAQscrape('spy').get_table()
    target = CAPMautoscrape.CMCscrape(asset)
    monthly_prices = pd.concat(target, benchmark, axis=1)
    monthly_prices = monthly_prices.reindex(index=monthly_prices.index[::-1]).reset_index(drop=True)
    monthly_prices.columns = ['Bit', '^GSPC']
    monthly_returns = monthly_prices.pct_change(1)
    clean_monthly_returns = monthly_returns.dropna(axis=0)  # drop first missing row
    X = clean_monthly_returns['^GSPC']
    y = clean_monthly_returns['Bit']
    X1 = sm.add_constant(X)
    model = sm.OLS(y, X1)
    # fit model and print results
    results = model.fit()
    print(results.summary())
    return model