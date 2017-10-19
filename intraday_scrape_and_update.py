from stock_utils import *
from bs4 import BeautifulSoup
import requests
import re
import time

def update_CSV_intraday(path_to_csv, backup=False):
    if backup is True:
        raise ValueError('Not yet supported')
        
    ticker = ticker_from_csv(path_to_csv)
        
    ## Open Yahoo Finance data
    url = "https://finance.yahoo.com/quote/{sym}/history?p={sym}".format(sym=ticker)
    r = requests.get(url)
    soup = BeautifulSoup(r.text, "html.parser")
    
    ## Store all rows of historical data for this stock in 'all_rows'
    all_rows = []
    historical_rows = soup.findAll('tr', attrs={'class':'BdT Bdc($c-fuji-grey-c) Ta(end) Fz(s) Whs(nw)'})
    todays_row = historical_rows[0]
    table_cells = todays_row.findAll('td')
    row_list = []
    for cell in table_cells:
        txt = cell.text.strip()
        row_list.append(txt)
        
    ## Add the intraday data to the CSV
    df = pd.DataFrame()
    df = df.from_csv(path_to_csv)
    
    if len(row_list) != 7:  ## "Dividend" for example
        return

    print('s_close: ' + str(row_list[5]))
    print('s_volume: ' + str(row_list[6]))
    print('s_open: ' + str(row_list[1]))
    print('s_high: ' + str(row_list[2]))
    print('s_low: ' + str(row_list[3]))
    print()
    ## DEBUG

    s_close = row_list[5]
    s_volume = row_list[6].replace(',', '')
    s_open = row_list[1]
    s_high = row_list[2]
    s_low = row_list[3]
   
    df.loc[pd.to_datetime('now') - pd.Timedelta('04:00:00')] = [s_close, s_volume, s_open, s_high, s_low]
      
    df = df.sort_index(axis=0)
    df.to_csv(path_to_csv)

g = glob.glob('intraday_stock_data/*.csv')
for filename in g:
    print(filename)
    update_CSV_intraday(filename)

