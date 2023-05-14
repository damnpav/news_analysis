import yfinance as yf
import pandas as pd

index_df = pd.read_excel('descr_stat.xlsx', sheet_name='index_tickets')  # загружаем файл с тикерами индекса
imoex_data = yf.download('IMOEX.ME', period='max', interval='1d')  # загружаем котировки значений индекса IMOEX

tickers_data = {}  # загружаем итеративно котировки бумаг из индекса IMOEX
for ticker in index_df['Ticker'].to_list():
    tickers_data[ticker] = yf.download(f'{ticker.upper()}.ME', period='max', interval='1d')

merged_data = pd.DataFrame()  # совмещаем котировки бумаг с котировкой индекса
merged_data['IMOEX'] = imoex_data['Close']
for ticker in tickers_data.keys():
    buf_df = tickers_data[ticker]['Close'].to_frame()
    buf_df = buf_df.rename(columns={'Close': ticker})
    merged_data = merged_data.join(buf_df)

merged_data = merged_data.dropna(how='all', axis=1).fillna(0)
merged_data_corr = merged_data.corr()  # рассчитываем корреляционную матрицу

