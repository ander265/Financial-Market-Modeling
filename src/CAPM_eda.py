import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import CAPMautoscrape

bitcoin = CAPMautoscrape.CMCscrape('bitcoin')
ethereum = CAPMautoscrape.CMCscrape('ethereum')
ripple = CAPMautoscrape.CMCscrape('ripple')

stocks = ['spy','wfc','jpm','tsla',
'ge','xom','aapl','amzn','csco','unh']

dfs = CAPMautoscrape.scrape_to_df(stocks)

dfc = [bitcoin, ethereum, ripple]
alldf = dfc + dfs

closes = []
for i in range(len(alldf)):
    close = alldf[i]['Close']
    closes.append(close)
closes = pd.DataFrame(closes).T
labels=['BTC','ETH','XRP','SPY','WFC','JPM','TSLA','GE','XOM','AAPL','AMZN','CSCO','UNH']


plt.clf()
cm = np.corrcoef(closes, rowvar=False)
plt.imshow(cm,interpolation='nearest', cmap=plt.cm.RdBu)
plt.yticks(list(range(13)),list(labels))
plt.xticks(range(13),labels,rotation=90)
plt.colorbar()
plt.show()

