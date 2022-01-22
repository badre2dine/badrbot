from binance import Client
from binance.enums import HistoricalKlinesType
import pandas as pd
import numpy as np

client = Client()
records = np.array(
    client.get_historical_klines(
        "BTCUSDT",
        "1m",
        "60 min ago",
        klines_type=HistoricalKlinesType.FUTURES,
    ),
)


df = pd.DataFrame(records)
df = df.iloc[:, :6]
df.columns = ["Time", "Open", "High", "Low", "Close", "Volume"]
df = df.set_index("Time")
df.index = pd.to_datetime(df.index, unit="ms")
df = df.astype(float)
print(df)
