from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import yfinance as yf
from datetime import datetime
import numpy as np
import pandas as pd
import warnings
import json
from fastapi import FastAPI, Body, HTTPException, Depends
from typing import Dict, Any, Optional, List
from openai import OpenAI
import json
from datetime import datetime
from parameters import api_key

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


# Define request and response models
class ChatRequest(BaseModel):
    message: str
    context: Optional[Dict[str, Any]] = None
    conversation_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    conversation_id: str

class ChatMessage(BaseModel):
    role: str
    content: str

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
    # print(type(rsi_values))
    return rsi_values.round(2)

# Helper function to calculate MACD
def calculate_macd(prices, fast_period=12, slow_period=26, signal_period=9):
    exp1 = prices.ewm(span=fast_period, adjust=False).mean()
    exp2 = prices.ewm(span=slow_period, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    # print("macd type:",type(macd),type(signal))
    return macd.round(2), signal

@app.get("/", response_model=StockDataResponse)
async def get_all_data(index: str = "^NSEI",timeframe: str = '6mo' ):#, start_date: Optional[str] = None, end_date: Optional[str] = None):
async def get_all_data(index: str = "^NSEI",timeframe: str = '6mo' ):#, start_date: Optional[str] = None, end_date: Optional[str] = None):
    print("index",index)
    try:
        # Fetch data from Yahoo Finance
        ticker = yf.Ticker(f"{index}")  # Use .NS for NSE (Indian market)

        data = ticker.history(period=timeframe , interval="1d")



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
        # print(2)
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

        # parameters = [
        #     {"duration": "1 Day", "rsi": 14, "macd": "12,26,9", "profit": 15.5},
        #     {"duration": "3 Days", "rsi": 14, "macd": "12,26,9", "profit": 18.2}
        # ]
        with open('data/parameters.json', 'r') as f:
            parameters = json.load(f)
        # print("parameters",type(parameters),type(parameters[0]))
        
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


# Configure OpenAI client
# In production, use environment variables for API keys
client=OpenAI(api_key=api_key)
# Store conversations (in a real app, use a database)
conversations = {}

# Function to get current market data
async def get_current_market_data(index: str, timeframe: str):
    """Fetch the current market data for context"""
    try:
        # Use your existing API endpoint
        from httpx import AsyncClient
        async with AsyncClient() as client:
            response = await client.get(f"http://127.0.0.1:8000/?index={index}&timeframe={timeframe}")
            if response.status_code == 200:
                data = response.json()
                # Extract the most relevant data for the chat context
                return {
                    "index": index,
                    "timeframe": timeframe,
                    "current_price": data["close"],
                    "trend": data["trend"],
                    "rsi_value": data["rsi"][-1]["y"] if data["rsi"] else None,
                    "macd_value": data["macd"][-1]["y"] if data["macd"] else None,
                }
            return None
    except Exception as e:
        print(f"Error fetching market data: {str(e)}")
        return None

# System prompt template
SYSTEM_PROMPT = """
Core Identity
You are an expert technical analysis assistant for a stock market analysis web application focusing on NIFTY 50 and SENSEX indices.
Application Features

Candlestick charting with time period selection (1m, 3m, 6m, 1y, 2y, 5y)
Technical indicators:

RSI (0-100 scale; >70 overbought, <30 oversold)
MACD (signal line crossovers indicate potential trend changes)


Market data display (OHLCV metrics)
Trend identification (Bullish/Bearish)

Assistant Responsibilities

Explain technical analysis concepts clearly
Help interpret chart patterns and indicator signals
Guide users toward informed trading decisions
Educate on technical analysis principles

Communication Style

Provide concise, precise responses
Answer questions directly without unnecessary elaboration
Maintain a conversational, natural tone
Structure responses in an organized format
Keep messages brief and focused
Leave proper space in the text 
maximum 4 lines only.

Your job is to help users understand technical analysis concepts, interpret the current chart data, 
and make informed decisions. Be concise, accurate, and educational.

{additional_context}
"""
# @app.post("/chat", response_model=ChatResponse)
# def chat(request: ChatRequest=Body(...)):
#     print(request)
#     return {"response":"thanks for calling","conversation_id":"01"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest = Body(...)):
    """
    Process chat messages using OpenAI API and provide technical analysis insights
    """
    print("chat Entered")
    try:
        # Generate a conversation ID if not provided
        conversation_id = request.conversation_id or datetime.now().strftime("%Y%m%d%H%M%S")
        print(1)
        # Get current market data if context is provided
        market_data = None
        if request.context and "index" in request.context and "timeframe" in request.context:
            market_data = await get_current_market_data(
                request.context["index"], 
                request.context["timeframe"]
            )
        print(2)
        
        # Build additional context based on market data
        additional_context = ""
        if market_data:
            additional_context = f"""
Current market context:
- Index: {market_data['index']}
- Timeframe: {market_data['timeframe']}
- Current Price: {market_data['current_price']}
- Current Trend: {market_data['trend']}
- Current RSI: {market_data['rsi_value']}
- Current MACD: {market_data['macd_value']}
            """
        print(additional_context)
        print(3)
        # Initialize or retrieve conversation history
        if conversation_id not in conversations:
            conversations[conversation_id] = [
                {"role": "system", "content": SYSTEM_PROMPT.format(additional_context=additional_context)}
            ]
        
        # Add user message to conversation
        conversations[conversation_id].append({"role": "user", "content": request.message})
        
        # Keep only the last 10 messages to avoid token limits
        if len(conversations[conversation_id]) > 12:  # system prompt + 10 messages
            # Always keep the system prompt (first message)
            conversations[conversation_id] = [
                conversations[conversation_id][0]
            ] + conversations[conversation_id][-10:]
        print(4)
        # Call OpenAI API
        # response = openai.ChatCompletion.create(
        #     model="gpt-4o-mini",  # or "gpt-4" if you have access
        #     messages=conversations[conversation_id],
        #     temperature=0.7,
        #     max_tokens=500
        # )
        # print(conversations[conversation_id])

        # response = client.chat.completions.create(
        #             model="gpt-4o-mini",
        #             messages=conversations[conversation_id]
        #         )
        
        # assistant_message = response.choices[0].message.content
        print(5)
        # Extract and store the assistant's response
        assistant_message ="thank you.."
        conversations[conversation_id].append({"role": "assistant", "content": assistant_message})
        
        return {
            "response": assistant_message,
            "conversation_id": conversation_id
        }
    
    except Exception as e:
        print("error",str(e))
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")

# Endpoint to get chat history (useful for debugging or UI state restoration)
@app.get("/chat/{conversation_id}")
async def get_chat_history(conversation_id: str):
    if conversation_id not in conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    # Return messages excluding the system prompt
    return {"messages": conversations[conversation_id][1:]}

# Optional: Endpoint to clear a conversation
@app.delete("/chat/{conversation_id}")
async def clear_chat_history(conversation_id: str):
    if conversation_id in conversations:
        # Keep only the system prompt
        system_prompt = conversations[conversation_id][0]
        conversations[conversation_id] = [system_prompt]
        return {"status": "Conversation cleared"}
    raise HTTPException(status_code=404, detail="Conversation not found")
