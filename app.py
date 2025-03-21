from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel
from typing import List, Optional
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

app = FastAPI()



# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (replace with your front-end URL in production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic model for the response
class StockDataResponse(BaseModel):
    open: float
    high: float
    low: float
    close: float
    volume: int
    trend: str
    candlestick: List[dict]
    rsi: List[dict]
    macd: List[dict]
    parameters: List[dict]

def calculate_initial_rsi(prices, period=14):
    changes = prices.diff()
    gains = changes.where(changes > 0, 0)
    losses = -changes.where(changes < 0, 0)

    avg_gain = gains.iloc[1:period + 1].mean()
    avg_loss = losses.iloc[1:period + 1].mean()

    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    initial_rsi = 100 - (100 / (1 + rs)) if avg_loss != 0 else 100

    return initial_rsi, avg_gain, avg_loss

def calculate_updated_rsi(prices, avg_gain, avg_loss, period=14):
    changes = prices.diff()
    gains = changes.where(changes > 0, 0)
    losses = -changes.where(changes < 0, 0)

    updated_avg_gain = pd.Series(index=prices.index, dtype="float64")
    updated_avg_loss = pd.Series(index=prices.index, dtype="float64")
    rsi_values = pd.Series(index=prices.index, dtype="float64")

    updated_avg_gain.iloc[period - 1] = avg_gain
    updated_avg_loss.iloc[period - 1] = avg_loss

    for i in range(period, len(prices)):
        updated_avg_gain.iloc[i] = ((updated_avg_gain.iloc[i - 1] * (period - 1)) + gains.iloc[i]) / period
        updated_avg_loss.iloc[i] = ((updated_avg_loss.iloc[i - 1] * (period - 1)) + losses.iloc[i]) / period

        rs = updated_avg_gain.iloc[i] / updated_avg_loss.iloc[i] if updated_avg_loss.iloc[i] != 0 else 0
        rsi_values.iloc[i] = 100 - (100 / (1 + rs)) if updated_avg_loss.iloc[i] != 0 else 100

    return rsi_values

# Helper function to calculate MACD
def calculate_macd(prices, fast_period=12, slow_period=26, signal_period=9):
    exp1 = prices.ewm(span=fast_period, adjust=False).mean()
    exp2 = prices.ewm(span=slow_period, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    return macd, signal

@app.get("/", response_model=StockDataResponse)
async def get_all_data(index: str = "^NSEI", start_date: Optional[str] = None, end_date: Optional[str] = None):
    print("index",index)
    try:
        # Fetch data from Yahoo Finance
        ticker = yf.Ticker(f"{index}")  # Use .NS for NSE (Indian market)
        if not start_date or not end_date :
            start_date = "2025-01-01"
            end_date = "2025-03-01"
        data = ticker.history(period="6mo" , interval="1d")
        #         end_date = datetime.now().strftime("%Y-%m-%d")
        #         start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        # data = ticker.history(start=start_date, end=end_date, interval="1d")


        if data.empty:
            raise HTTPException(status_code=404, detail="No data found for the given index and date range")
            # print("no data found")
        data=data.round(2)
        # Extract required metrics
        open_price = data['Open'].iloc[-1]
        high_price = data['High'].iloc[-1]
        low_price = data['Low'].iloc[-1]
        close_price = data['Close'].iloc[-1]
        volume = data['Volume'].iloc[-1]
        trend = "Bullish" if close_price > open_price else "Bearish"
        print(2)
        # Prepare candlestick data
        candlestick = [{
            "x": date.strftime("%Y-%m-%d"),
            "y": [row['Open'], row['High'], row['Low'], row['Close']]
        } for date, row in data.iterrows()]

        # Calculate RSI
        prices=data['Close']
        rsi_period = 14
        # rsi = calculate_rsi(data['Close'].values)
        initial_rsi, avg_gain, avg_loss = calculate_initial_rsi(prices, rsi_period)
        rsi = calculate_updated_rsi(prices, avg_gain, avg_loss, rsi_period)
      
        rsi_data = [{"x": date.strftime("%Y-%m-%d"), "y": rsi[i]} for i, date in enumerate(data.index)]

        # Calculate MACD
        macd, signal = calculate_macd(data['Close'])
        macd_data = [{"x": date.strftime("%Y-%m-%d"), "y": macd[i]} for i, date in enumerate(data.index)]

        # Mock best parameters (replace with actual logic if needed)
        parameters = [
            {"duration": "1 Day", "rsi": 14, "macd": "12,26,9", "profit": 15.5},
            {"duration": "3 Days", "rsi": 14, "macd": "12,26,9", "profit": 18.2}
        ]

        # Return the response
        return {
            "open": open_price,
            "high": high_price,
            "low": low_price,
            "close": close_price,
            "volume": int(volume),
            "trend": trend,
            "candlestick": candlestick,
            "rsi": rsi_data,
            "macd": macd_data,
            "parameters": parameters
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        # print("error",str(e))


@app.get("/stock_data")
async def get_stock_data(index: str = "^NSEI", start_date: Optional[str] = None, end_date: Optional[str] = None):
    try:
        ticker = yf.Ticker(f"{index}")  
        if not start_date or not end_date :
            start_date = "2025-01-01"
            end_date = "2025-03-01"
        data = ticker.history(period="6mo" , interval="1d")
        candlestick = [{
            "x": date.strftime("%Y-%m-%d"),
            "y": [row['Open'], row['High'], row['Low'], row['Close']]
        } for date, row in data.iterrows()]

        return {"Response":candlestick}
    except Exception as e:
        print(str(e))
        return {"Respone":"error"+str(e)}



@app.get("/rsi")
def rsi_data():

    return {"Response":"Rsi_data"}

@app.get("/macd")
def macd_data():

    return {"Response":"Macd data"}
