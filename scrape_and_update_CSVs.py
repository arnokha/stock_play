from stock_utils import *
from bs4 import BeautifulSoup
import requests
import re
import time

def update_CSV(path_to_csv, backup=False):
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
    for row in historical_rows:
        table_cells = row.findAll('td')
        row_list = []
        for cell in table_cells:
            #print(cell.text.strip())
            txt = cell.text.strip()
            ## Missing values
            #if txt == '-':
            #    print('!! MISSING VALUE found while scraping !!')
            #    print('Aborting update for this stock.')
            #    print('ticker is: ' + ticker)
            #    return
            row_list.append(txt)
        all_rows.append(row_list)
        
    ## Add any days that are missing to the CSV
    df = pd.DataFrame()
    df = df.from_csv(path_to_csv)
    
    for i in range(len(all_rows)):
        if len(all_rows[i]) != 7:  ## "Dividend" for example
            print('CONTINUE!?')
            continue
        if pd.to_datetime(all_rows[i][0]) not in df.index:
            ## DEBUG
            print('s_close: ' + str(all_rows[i][5]))
            print('s_volume: ' + str(all_rows[i][6]))
            print('s_open: ' + str(all_rows[i][1]))
            print('s_high: ' + str(all_rows[i][2]))
            print('s_low: ' + str(all_rows[i][3]))
            print()
            ## DEBUG

            s_close = all_rows[i][5]
            s_volume = all_rows[i][6].replace(',', '')
            s_open = all_rows[i][1]
            s_high = all_rows[i][2]
            s_low = all_rows[i][3]
            df.loc[pd.to_datetime(all_rows[i][0])] = [s_close, s_volume, s_open, s_high, s_low]
      
    df = df.sort_index(axis=0)
    df.to_csv(path_to_csv)

g = glob.glob('stock_data/*.csv')
for filename in g:
    update_CSV(filename)

g = glob.glob('new_stock_data/*.csv')
for filename in g:
    print('-------------------------')
    print(filename) ## DEBUG
    print('-------------------------')
    update_CSV(filename)
