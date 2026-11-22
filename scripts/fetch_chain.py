"""Download a real option chain with yfinance and save it as data/chain_<TICKER>.csv.

Usage (needs internet):
    pip install yfinance pandas
    python scripts/fetch_chain.py SPY

Output columns: strike, T (years), price (bid/ask mid), kind, spot, r, q.
Quotes with no bid, no ask or tiny volume are dropped. The rate and dividend
yield are simple flat assumptions you can edit below.
"""
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf

RATE = 0.04        # flat risk-free rate assumption (edit to the current level)
DIV_YIELD = 0.013  # flat dividend yield assumption


def main(ticker: str = "SPY", max_expiries: int = 6):
    tk = yf.Ticker(ticker)
    spot = float(tk.history(period="1d")["Close"].iloc[-1])
    now = datetime.now(timezone.utc)
    rows = []
    for expiry in tk.options[:max_expiries]:
        T = (datetime.strptime(expiry, "%Y-%m-%d").replace(tzinfo=timezone.utc) - now).days / 365.0
        if T < 7 / 365:
            continue
        chain = tk.option_chain(expiry)
        for kind, df in (("call", chain.calls), ("put", chain.puts)):
            df = df[(df.bid > 0) & (df.ask > 0) & (df.volume.fillna(0) > 10)]
            # out-of-the-money options only: they are the most liquid
            df = df[df.strike >= spot] if kind == "call" else df[df.strike <= spot]
            for _, row in df.iterrows():
                rows.append(dict(strike=row.strike, T=T, price=0.5 * (row.bid + row.ask),
                                 kind=kind, spot=spot, r=RATE, q=DIV_YIELD))
    out = pd.DataFrame(rows)
    path = f"data/chain_{ticker}.csv"
    out.to_csv(path, index=False)
    print(f"saved {len(out)} quotes to {path} (spot {spot:.2f})")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "SPY")
