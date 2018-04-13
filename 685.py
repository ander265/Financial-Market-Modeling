import importlib
import os
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from lxml import html
import requests
import sys
import datetime
os.getcwd()
# os.chdir()



page = requests.get('https://coinmarketcap.com/currencies/bitcoin/historical-data/?start=20170218&end=20180218')
tree = html.fromstring(page.content)

# for loops to pull out the data
[i for i in range(1, 20)]

# Bitcoin/Ethereum vs. stocks by industries
# Which industries have stocks that behave most like these popular currencies
# same metrics
# price per share
# price to earnings ratio
# both are tradeable

# USD vs. Bitcoin
# Exchange Rate
# inherent correlation between usd and stock market
# no economic worth to cryptocurrency
# currency without regulation (e.g. federal reserve)



#Begin

#extract bitcoin data, from beginning date to most recent entry

current=datetime.date.today()
print("https://coinmarketcap.com/currencies/bitcoin/historical-data/?start=20171121&end=2018"+str(current.month).zfill(2)+str(current.day).zfill(2))
#page1 = requests.get("https://coinmarketcap.com/currencies/bitcoin/historical-data/?start=20171121&end=20180222")
page1 = requests.get("https://coinmarketcap.com/currencies/bitcoin/historical-data/?start=20171121&end=2018"+str(current.month).zfill(2)+str(current.day).zfill(2))
soups1 = BeautifulSoup(page1.content, "html.parser")

#establishing objects to convert data into a PANDAS Dataframe
list_of_rows = []
thread = []
df = pd.DataFrame()

# identifying relevant HTML tag to create dataframe
names = soups1.findAll("thead")
for th in names:
    cols = th.findAll("th")
    dfnames = [c.text for c in cols]

rows = soups1.findAll("tr")
# [r.text for r in rows]
for tr in rows:
    cols = tr.findAll("td")
    thread.append(tr.findAll("td"))
# data = [c.text for c in cols]
dates = thread[1:len(thread) - 1]
for i in range(len(dates)):
    day = dates[i]
    datas = [c.text for c in day]
    list_of_rows.append(datas)
# incomplete, finish with pandas!
df = pd.DataFrame(list_of_rows)
df.columns = dfnames
rawdates = df.iloc[:, 0]
df[df.columns[1:5]] = df[df.columns[1:5]].astype(float)
df["Volume"] = df["Volume"].str.replace(",", "").astype(float)
df["Market Cap"] = df["Market Cap"].str.replace(",", "").astype(float)


#stock data
class Tester(object):
    def __init__(self, name):
        self.name = str(name).upper().strip()

        try:
            self.page = requests.get("http://www.nasdaq.com/symbol/" + self.name + "/historical")
            self.tree = html.fromstring(self.page.content)
            self.page_text = self.page.text

            '''
			timeframe=self.page_text.find('<option value="3m" selected="selected">3 Months</option>')
			print(timeframe)

			new1=self.page_text.replace('<option value="3m" selected="selected">3 Months</option>','<option value="3m">3 Months</option>')
			new2=new1.replace('<option value="10y">10 Years</option>','<option value="10y" selected="selected">10 Years</option>')

			'''
            info_start = self.page_text.find('</thead>')
            info_end = self.page_text.find('<!-- end genTable-->')

            self.info = self.page_text[info_start:info_end]

        except Exception as e:
            print("ERROR:\n" + str(e.args) + "\n\nTerminating execution...")
            sys.exit()

    def get_table(self):

        day = self.info.split("</tr>")
        day_new = [days.replace("\r\n", "") for days in day]
        day1 = [y.replace(" ", "") for y in day_new]
        day2 = [y.replace('<td>', '') for y in day1]
        day3 = [y.replace('</thead>', '') for y in day2]
        day4 = [y.replace('<tbody>', '') for y in day3]
        day5 = [y.replace('<tr>', '') for y in day4]
        day6 = [y.replace('</div>', '') for y in day5]
        day7 = [y.replace('</table>', '') for y in day6]
        day8 = [y.replace('</tbody>', '') for y in day7]

        day_list = [x.split("</td>") for x in day8]
        return day_list

share0 = Tester('spy')
share1 = Tester('wfc')  # finance
share2 = Tester('jpm')  # finance
share3 = Tester('tsla')  # energy/tech
share4 = Tester('ge')  # energy
share5 = Tester('xom')  # energy
share6 = Tester('aapl')  # technology
share7 = Tester('amzn')  # tech/consumer goods & services
share8 = Tester('csco')  # telecomunnications
share9 = Tester('unh')  # healthcare

s0 = share0.get_table()
df0 = pd.DataFrame(s0)
s1 = share1.get_table()
df1 = pd.DataFrame(s1)
s2 = share2.get_table()
df2 = pd.DataFrame(s2)
s3 = share3.get_table()
df3 = pd.DataFrame(s3)
s4 = share4.get_table()
df4 = pd.DataFrame(s4)
s5 = share5.get_table()
df5 = pd.DataFrame(s5)
s6 = share6.get_table()
df6 = pd.DataFrame(s6)
s7 = share7.get_table()
df7 = pd.DataFrame(s7)
s8 = share8.get_table()
df8 = pd.DataFrame(s8)
s9 = share9.get_table()
df9 = pd.DataFrame(s9)

df0.columns = df1.columns = df2.columns = df3.columns = df4.columns = \
    df5.columns = df6.columns = df7.columns = df8.columns = \
    df9.columns = ['Day', 'Open', 'High', 'Low', 'Close', 'Volume', 'NA']

dfs = [df0, df1, df2, df3, df4, df5, df6, df7, df8, df9]
#cleaning the stock datasets
for i in range(len(dfs)):
    dfs[i] = dfs[i][1:-1].reset_index(drop=True) #extraneous rows at beginning and end
    dfs[i] = dfs[i].iloc[:, 0:6] #extraneous column at end
    dfs[i][dfs[i].columns[1:5]] = dfs[i][dfs[i].columns[1:5]].astype(float)
    dfs[i]["Volume"] = dfs[i]["Volume"].str.replace(",", "").astype(float)



#Converting Dates
dtf = pd.Series('', index=range(len(rawdates)))

for i in range(len(rawdates)):
    if rawdates[i][0:3] == 'Feb':
        month = '02'
    elif rawdates[i][0:3] == 'Jan':
        month = '01'
    elif rawdates[i][0:3] == 'Dec':
        month = '12'
    elif rawdates[i][0:3] == 'Mar':
        month = '03'
    elif rawdates[i][0:3] == 'Apr':
        month = '04'
    elif rawdates[i][0:3] == 'May':
        month = '05'
    elif rawdates[i][0:3] == 'Jun':
        month = '06'
    elif rawdates[i][0:3] == 'Jul':
        month = '07'
    elif rawdates[i][0:3] == 'Aug':
        month = '08'
    elif rawdates[i][0:3] == 'Sep':
        month = '09'
    elif rawdates[i][0:3] == 'Oct':
        month = '10'
    else:
        month = '11'
    day = rawdates[i][4:6]
    year = rawdates[i][8:12]
    dtf[i] = month + '/' + day + '/' + year

df.iloc[:,0] = dtf

bitcoin = df #more correlation for crypto comparison
#bitcoin dataset that matches dates of stock data
dfb = df[df['Date'].isin(dfs[0].iloc[:,0])].reset_index(drop=True)

page2 = requests.get("https://coinmarketcap.com/currencies/ethereum/historical-data/?start=20171121&end=2018"+str(current.month).zfill(2)+str(current.day).zfill(2))
soups2 = BeautifulSoup(page2.content, "html.parser")

list_of_rows = []
thread = []
dfe = pd.DataFrame()

rows = soups2.findAll("tr")
# [r.text for r in rows]
for tr in rows:
    cols = tr.findAll("td")
    thread.append(tr.findAll("td"))
# data = [c.text for c in cols]
dates = thread[1:len(thread) - 1]
for i in range(len(dates)):
    day = dates[i]
    datas = [c.text for c in day]
    list_of_rows.append(datas)
# incomplete, finish with pandas!
dfe = pd.DataFrame(list_of_rows)
dfe.columns = dfnames
dfe[dfe.columns[1:5]] = dfe[dfe.columns[1:5]].astype(float)
dfe["Volume"] = dfe["Volume"].str.replace(",", "").astype(float)
dfe["Market Cap"] = dfe["Market Cap"].str.replace(",", "").astype(float)



#dff = dfe.iloc[:, 0]
#dtf = pd.Series('', index=range(len(dff)))


dfe.iloc[:,0] = dtf
ethereum = dfe #more data for cryptos for correlation comparison
dfe = dfe[dfe['Date'].isin(dfs[0].iloc[:,0])].reset_index(drop=True)


page3 = requests.get("https://coinmarketcap.com/currencies/ripple/historical-data/?start=20171121&end=2018"+str(current.month).zfill(2)+str(current.day).zfill(2))
soups3 = BeautifulSoup(page3.content, "html.parser")

list_of_rows = []
thread = []
dfr = pd.DataFrame()

rows = soups3.findAll("tr")
# [r.text for r in rows]
for tr in rows:
    cols = tr.findAll("td")
    thread.append(tr.findAll("td"))
# data = [c.text for c in cols]
dates = thread[1:len(thread) - 1]
for i in range(len(dates)):
    day = dates[i]
    datas = [c.text for c in day]
    list_of_rows.append(datas)
# incomplete, finish with pandas!
dfr = pd.DataFrame(list_of_rows)
dfr.columns = dfnames
dfr[dfr.columns[1:5]] = dfr[dfr.columns[1:5]].astype(float)
dfr["Volume"] = dfr["Volume"].str.replace(",", "").astype(float)
dfr["Market Cap"] = dfr["Market Cap"].str.replace(",", "").astype(float)

dfr.iloc[:,0] = dtf
ripple = dfr # more data for cryptos for correlation comparison
dfr = dfr[dfr['Date'].isin(dfs[0].iloc[:,0])].reset_index(drop=True)

dfc = [dfb,dfe,dfr]
alldf = dfc + dfs
#statistical analyses

np.corrcoef(bitcoin.Close,ethereum.Close)
np.corrcoef(bitcoin.Close,ripple.Close)
np.corrcoef(ethereum.Close,ripple.Close)

c = [bitcoin.Close,ethereum.Close,ripple.Close]
np.corrcoef(c)

closes = []
for i in range(len(alldf)):
    close = alldf[i]['Close']
    closes.append(close)
closes = pd.DataFrame(closes).T
labels=['BTC','ETH','XRP','SPY','WFC','JPM','TSLA','GE','XOM','AAPL','AMZN','CSCO','UNH']

closes.columns = labels

import matplotlib.pyplot as plt
plt.clf()
cm = np.corrcoef(closes, rowvar=False)
plt.imshow(cm,interpolation='nearest', cmap=plt.cm.RdBu)
plt.yticks(list(range(13)),list(labels))
plt.xticks(range(13),labels,rotation=90)
plt.colorbar()
plt.show()
ticks=np.arange(3,4,2)

plt.yticks()
#plt.style.use(ggplot2)
import seaborn as sns
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.gofplots import ProbPlot



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
plt.style.use('ggplot')
plt.scatter(X,y,c='b')
plt.plot(X,results.predict(),'r--')
plt.ylabel('Bitcoin % Change')
plt.savefig('BTCpc.png')
plot_acf(y) #deprecated since use
plt.title('Bitcoin % Change Autocorrelation')
plt.savefig('BTCacfpc.png')

plt.style.use('seaborn') # pretty matplotlib plots
plt.rc('font', size=14)
plt.rc('figure', titlesize=18)
plt.rc('axes', labelsize=15)
plt.rc('axes', titlesize=18)
# fitted values (need a constant term for intercept)
model_fitted_y = results.fittedvalues
# model residuals
model_residuals = results.resid
# normalized residuals
model_norm_residuals = results.get_influence().resid_studentized_internal
# absolute squared normalized residuals
model_norm_residuals_abs_sqrt = np.sqrt(np.abs(model_norm_residuals))
# absolute residuals
model_abs_resid = np.abs(model_residuals)
# leverage, from statsmodels internals
model_leverage = results.get_influence().hat_matrix_diag
# cook's distance, from statsmodels internals
model_cooks = results.get_influence().cooks_distance[0]
plot_lm_1 = plt.figure(1)
plot_lm_1.set_figheight(8)
plot_lm_1.set_figwidth(12)
plot_lm_1.axes[0] = sns.residplot(model_fitted_y, 'Bit', data=clean_monthly_returns,
                          lowess=True,
                          scatter_kws={'alpha': 0.5},
                          line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})
plot_lm_1.axes[0].set_title('Bitcoin')
plot_lm_1.axes[0].set_xlabel('Fitted Values')
plot_lm_1.axes[0].set_ylabel('Residuals')
# annotations
abs_resid = model_abs_resid.sort_values(ascending=False)
abs_resid_top_7 = abs_resid[:7]
for i in abs_resid_top_7.index:
    plot_lm_1.axes[0].annotate(i,
                               xy=(model_fitted_y[i],
                                   model_residuals[i]));
plt.savefig('BTCfitres.png')
QQ = ProbPlot(model_norm_residuals)
plot_qq_1 = QQ.qqplot(line='45', alpha=0.5, color='#4C72B0', lw=1)
plot_qq_1.set_figheight(8)
plot_qq_1.set_figwidth(12)
plot_qq_1.axes[0].set_title('Normal Q-Q')
plot_qq_1.axes[0].set_xlabel('Theoretical Quantiles')
plot_qq_1.axes[0].set_ylabel('Standardized Residuals');
# annotations
abs_norm_resid = np.flip(np.argsort(np.abs(model_norm_residuals)), 0)
abs_norm_resid_top_4 = abs_norm_resid[:4]
for r, i in enumerate(abs_norm_resid_top_4):
    plot_qq_1.axes[0].annotate(i,
                               xy=(np.flip(QQ.theoretical_quantiles, 0)[r],
                                   model_norm_residuals[i]));
plt.savefig('BTCqq.png')

plot_cd_1 = plt.figure(4)
plot_cd_1.set_figheight(8)
plot_cd_1.set_figwidth(12)
plt.scatter(model_leverage, model_norm_residuals, alpha=0.5)
sns.regplot(model_leverage, model_norm_residuals,
            scatter=False,
            ci=False,
            lowess=True,
            line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})
plot_cd_1.axes[0].set_xlim(0, 0.35)
plot_cd_1.axes[0].set_ylim(-3, 5)
plot_cd_1.axes[0].set_title('Residuals vs Leverage')
plot_cd_1.axes[0].set_xlabel('Leverage')
plot_cd_1.axes[0].set_ylabel('Standardized Residuals')
# annotations
leverage_top_3 = np.flip(np.argsort(model_cooks), 0)[:3]
for i in leverage_top_3:
    plot_cd_1.axes[0].annotate(i,
                               xy=(model_leverage[i],
                                   model_norm_residuals[i]))
plt.savefig('BTClev.png')

monthly_prices = pd.concat([dfe['Close'], dfs[0]['Close']], axis=1)
monthly_prices = monthly_prices.reindex(index=monthly_prices.index[::-1]).reset_index(drop=True)
monthly_prices.columns = ['Ether', '^GSPC']
monthly_returns = monthly_prices.pct_change(1)
clean_monthly_returns = monthly_returns.dropna(axis=0)  # drop first missing row
X = clean_monthly_returns['^GSPC']
y = clean_monthly_returns['Ether']
plt.clf()
plt.style.use('ggplot')
plt.scatter(X,y,c='b')
# Add a constant to the independent value
X1 = sm.add_constant(X)

# make regression model
model2 = sm.OLS(y, X1)

# fit model and print results
results2 = model2.fit()
print(results2.summary())
plt.plot(X,results2.predict(),'r--')
plt.ylabel('Ethereum % Change')
plt.savefig('ETHpc.png')
plot_acf(y) #deprecated since use
plt.title('Ethereum % Change Autocorrelation')
plt.savefig('ETHafcpc.png')

# fitted values (need a constant term for intercept)
model_fitted_y2 = results2.fittedvalues
# model residuals
model_residuals2 = results2.resid
# normalized residuals
model_norm_residuals2 = results2.get_influence().resid_studentized_internal
# absolute squared normalized residuals
model_norm_residuals_abs_sqrt2 = np.sqrt(np.abs(model_norm_residuals2))
# absolute residuals
model_abs_resid2 = np.abs(model_residuals2)
# leverage, from statsmodels internals
model_leverage2 = results2.get_influence().hat_matrix_diag
# cook's distance, from statsmodels internals
model_cooks2 = results2.get_influence().cooks_distance[0]
plot_lm_2 = plt.figure(2)
plot_lm_2.set_figheight(8)
plot_lm_2.set_figwidth(12)
plot_lm_2.axes[0] = sns.residplot(model_fitted_y2, 'Ether', data=clean_monthly_returns,
                          lowess=True,
                          scatter_kws={'alpha': 0.5},
                          line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})
plot_lm_2.axes[0].set_title('Ethereum')
plot_lm_2.axes[0].set_xlabel('Fitted Values')
plot_lm_2.axes[0].set_ylabel('Residuals')

# annotations
abs_resid2 = model_abs_resid2.sort_values(ascending=False)
abs_resid2_top_6 = abs_resid2[:6]
for i in abs_resid2_top_6.index:
    plot_lm_2.axes[0].annotate(i,
                               xy=(model_fitted_y2[i],
                                   model_residuals2[i]));
plt.savefig('ETHfitres.png')
QQ2 = ProbPlot(model_norm_residuals2)
plot_qq_2 = QQ2.qqplot(line='45', alpha=0.5, color='#4C72B0', lw=1)
plot_qq_2.set_figheight(8)
plot_qq_2.set_figwidth(12)
plot_qq_2.axes[0].set_title('Normal Q-Q')
plot_qq_2.axes[0].set_xlabel('Theoretical Quantiles')
plot_qq_2.axes[0].set_ylabel('Standardized Residuals');
# annotations
abs_norm_resid2 = np.flip(np.argsort(np.abs(model_norm_residuals2)), 0)
abs_norm_resid2_top_3 = abs_norm_resid2[:3]
for r, i in enumerate(abs_norm_resid2_top_3):
    plot_qq_2.axes[0].annotate(i,
                               xy=(np.flip(QQ2.theoretical_quantiles, 0)[r],
                                   model_norm_residuals2[i]));
plt.savefig('ETHqq.png')
plot_cd_2 = plt.figure(5)
plot_cd_2.set_figheight(8)
plot_cd_2.set_figwidth(12)
plt.scatter(model_leverage2, model_norm_residuals2, alpha=0.5)
sns.regplot(model_leverage2, model_norm_residuals2,
            scatter=False,
            ci=False,
            lowess=True,
            line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})
plot_cd_2.axes[0].set_xlim(0, 0.35)
plot_cd_2.axes[0].set_ylim(-3, 5)
plot_cd_2.axes[0].set_title('Residuals vs Leverage')
plot_cd_2.axes[0].set_xlabel('Leverage')
plot_cd_2.axes[0].set_ylabel('Standardized Residuals')
# annotations
leverage_top_3 = np.flip(np.argsort(model_cooks2), 0)[:3]
for i in leverage_top_3:
    plot_cd_2.axes[0].annotate(i,
                               xy=(model_leverage2[i],
                                   model_norm_residuals2[i]))
plt.savefig('ETHlev.png')



monthly_prices = pd.concat([dfr['Close'], dfs[0]['Close']], axis=1)
monthly_prices = monthly_prices.reindex(index=monthly_prices.index[::-1]).reset_index(drop=True)
monthly_prices.columns = ['Ripple', '^GSPC']
monthly_returns = monthly_prices.pct_change(1)
clean_monthly_returns = monthly_returns.dropna(axis=0)  # drop first missing row
X = clean_monthly_returns['^GSPC']
y = clean_monthly_returns['Ripple']

plt.clf()
plt.scatter(X,y)
plt.style.use('ggplot')
plt.scatter(X,y,c='b')

# Add a constant to the independent value
X1 = sm.add_constant(X)

# make regression model
model3 = sm.OLS(y, X1)

# fit model and print results
results3 = model3.fit()
print(results3.summary())
plt.plot(X,results3.predict(),'r--')
plt.ylabel('Ripple % Change')
plt.savefig('XRPpc.png')
plt.clf()
plot_acf(y) #deprecated since use
plt.title('Ripple % Change Autocorrelation')
plt.savefig('XRPacfpc.png')


# fitted values (need a constant term for intercept)
model_fitted_y3 = results3.fittedvalues
# model residuals
model_residuals3 = results3.resid
# normalized residuals
model_norm_residuals3 = results3.get_influence().resid_studentized_internal
# absolute squared normalized residuals
model_norm_residuals_abs_sqrt3 = np.sqrt(np.abs(model_norm_residuals2))
# absolute residuals
model_abs_resid3 = np.abs(model_residuals3)
# leverage, from statsmodels internals
model_leverage3 = results3.get_influence().hat_matrix_diag
# cook's distance, from statsmodels internals
model_cooks3 = results3.get_influence().cooks_distance[0]
plot_lm_3 = plt.figure(3)
plot_lm_3.set_figheight(8)
plot_lm_3.set_figwidth(12)
plot_lm_3.axes[0] = sns.residplot(model_fitted_y3, 'Ripple', data=clean_monthly_returns,
                          lowess=True,
                          scatter_kws={'alpha': 0.5},
                          line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})
plot_lm_3.axes[0].set_title('Ripple')
plot_lm_3.axes[0].set_xlabel('Fitted Values')
plot_lm_3.axes[0].set_ylabel('Residuals')

# annotations
abs_resid3 = model_abs_resid3.sort_values(ascending=False)
abs_resid3_top_3 = abs_resid3[:3]
for i in abs_resid3_top_3.index:
    plot_lm_3.axes[0].annotate(i,
                               xy=(model_fitted_y3[i],
                                   model_residuals3[i]));
plt.savefig('XRPfitres.png')
QQ3 = ProbPlot(model_norm_residuals3)
plot_qq_3 = QQ3.qqplot(line='45', alpha=0.5, color='#4C72B0', lw=1)
plot_qq_3.set_figheight(8)
plot_qq_3.set_figwidth(12)
plot_qq_3.axes[0].set_title('Normal Q-Q')
plot_qq_3.axes[0].set_xlabel('Theoretical Quantiles')
plot_qq_3.axes[0].set_ylabel('Standardized Residuals');
# annotations
abs_norm_resid3 = np.flip(np.argsort(np.abs(model_norm_residuals3)), 0)
abs_norm_resid3_top5 = abs_norm_resid3[:5]
for r, i in enumerate(abs_norm_resid3_top5):
    plot_qq_3.axes[0].annotate(i,
                               xy=(np.flip(QQ3.theoretical_quantiles, 0)[r],
                                   model_norm_residuals3[i]));
plt.savefig('XRPqq.png')

plot_cd_3 = plt.figure(6)
plot_cd_3.set_figheight(8)
plot_cd_3.set_figwidth(12)
plt.scatter(model_leverage3, model_norm_residuals3, alpha=0.5)
sns.regplot(model_leverage3, model_norm_residuals3,
            scatter=False,
            ci=False,
            lowess=True,
            line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})
plot_cd_3.axes[0].set_xlim(0, 0.35)
plot_cd_3.axes[0].set_ylim(-3, 5)
plot_cd_3.axes[0].set_title('Residuals vs Leverage')
plot_cd_3.axes[0].set_xlabel('Leverage')
plot_cd_3.axes[0].set_ylabel('Standardized Residuals')
# annotations
leverage_top_3 = np.flip(np.argsort(model_cooks3), 0)[:3]
for i in leverage_top_3:
    plot_cd_3.axes[0].annotate(i,
                               xy=(model_leverage3[i],
                                   model_norm_residuals3[i]))
plt.savefig('XRPlev.png')

plot_acf(dfb.Close)
plt.title('Bitcoin Autocorrelation')
plt.savefig('BTCacf.png')
plot_acf(dfe.Close)
plt.title('Ethereum Autocorrelation')
plt.savefig('ETHacf.png')
plot_acf(dfr.Close)
plt.title('Ripple Autocorrelation')
plt.savefig('XRPacf.png')
plot_acf(closes.SPY)
plt.title('Stock Autocorrelation')
plt.savefig('S&P500afc.png')
plot_acf(X)
plt.title('Stock % Change Autocorrelation')
plt.savefig('S&P500acfpc.png')