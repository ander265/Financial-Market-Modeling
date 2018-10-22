import importlib
import os
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from lxml import html
import requests
import sys
import datetime

#sys.version,os.getcwd()

def format_dates(rawdates):
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
    return dtf

def CMCscrape(crypto_name):
    current=datetime.date.today()
    print("https://coinmarketcap.com/currencies/bitcoin/historical-data/?start=20171121&end=2018"+str(current.month).zfill(2)+str(current.day).zfill(2))
    #page1 = requests.get("https://coinmarketcap.com/currencies/bitcoin/historical-data/?start=20171121&end=20180222")
    page = requests.get("https://coinmarketcap.com/currencies/bitcoin/historical-data/?start=20171121&end=2018"+str(current.month).zfill(2)+str(current.day).zfill(2))
    soups = BeautifulSoup(page1.content, "html.parser")

    #establishing objects to convert data into a PANDAS Dataframe
    list_of_rows = []
    thread = []
    df = pd.DataFrame()

    # identifying relevant HTML tag to create dataframe
    names = soups.findAll("thead")
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
    #rawdates = df.iloc[:, 0]
    df.iloc[:,0] = format_dates(df.iloc[:,0])
    df[df.columns[1:5]] = df[df.columns[1:5]].astype(float)
    df["Volume"] = df["Volume"].str.replace(",", "").astype(float)
    df["Market Cap"] = df["Market Cap"].str.replace(",", "").astype(float)
    
    return df

class NASDAQscrape(object):
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

#sand publishing

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


def scrape_to_df(stock_name):
    dfs = []
    share = NASDAQscrape(stock_name)
    s = share.get_table()
    df = pd.DataFrame(s)
    df.columns = ['Day', 'Open', 'High', 'Low', 'Close', 'Volume', 'NA']
    dfs.append(df)
    return dfs
