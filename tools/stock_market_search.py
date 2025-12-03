# Imports
import os
import yfinance as yf
from prophet import Prophet
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import pandas as pd
import math
load_dotenv()

# GROQ Setup
GROQ_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_KEY:
    # don't raise here — let tools handle gracefully, but log helpful message
    print("Warning: GROQ_API_KEY not set. LLM calls will likely fail.")

llm = ChatGroq(
    api_key=GROQ_KEY,
    model="llama-3.1-8b-instant",
    temperature=0.5,
    disable_streaming=True
)

# Tools defining
@tool
def fetch_stock_data(query: str) -> str:
    """
    Fetch historical stock data for Indian stocks.
    Input examples: "RELIANCE 6mo 1d", "TCS", "INFY 3mo 1d"
    """
    try:
        if not query or not isinstance(query, str):
            return "❌ Invalid query. Example: 'RELIANCE 6mo 1d' or just 'TCS'"

        parts = query.strip().split()
        symbol = parts[0].upper()
        period = parts[1] if len(parts) > 1 else "6mo"
        interval = parts[2] if len(parts) > 2 else "1d"

        ticker = yf.Ticker(f"{symbol}")
        df = ticker.history(period=period, interval=interval)

        if df is None or df.empty:
            return f"❌ Error: No data found for {symbol}. Check symbol or try: RELIANCE, TCS, INFY, HDFC"

        # defensive access
        if 'Close' not in df.columns:
            return f"❌ Error: Data fetched for {symbol} is missing 'Close' column."

        latest_price = float(df['Close'].iloc[-1])
        prev_close = float(df['Close'].iloc[-2]) if len(df) > 1 else latest_price
        change_pct = ((latest_price - prev_close) / prev_close * 100) if prev_close != 0 else 0.0

        highest = float(df['High'].max()) if 'High' in df.columns else math.nan
        lowest = float(df['Low'].min()) if 'Low' in df.columns else math.nan
        volume_mean = int(df['Volume'].mean()) if 'Volume' in df.columns else 0

        latest_date = df.index[-1]
        # ensure datetime string without tz issues
        if hasattr(latest_date, 'tz'):
            latest_date = latest_date.tz_convert(None)
        latest_date_str = pd.to_datetime(latest_date).strftime('%Y-%m-%d')

        summary = (
            f"Stock Data: {symbol}\n"
            f"─────────────────────────\n"
            f"Period: {period} | Interval: {interval}\n"
            f"Latest Price: ₹{latest_price:.2f}\n"
            f"Change: {change_pct:+.2f}%\n"
            f"Highest price in specified time period: ₹{highest:.2f}\n"
            f"Lowest price in specified time period: ₹{lowest:.2f}\n"
            f"Volume (avg): {volume_mean}\n\n"
            f"Data Points: {len(df)}\n"
            f"Latest Date: {latest_date_str}"
        )

        return summary

    except Exception as e:
        return f"❌ Error in fetch_stock_data: {str(e)}"


@tool
def predict_stock_price(query: str) -> str:
    """
    Predict future stock prices using Prophet.
    Input: "RELIANCE 15" -> predicts next 15 days
    """
    try:
        if not query or not isinstance(query, str):
            return "❌ Invalid query. Example: 'RELIANCE 15'"

        parts = query.strip().split()
        if len(parts) < 2:
            return "❌ Error: Format should be 'SYMBOL DAYS'. Example: 'RELIANCE 15'"

        symbol = parts[0].upper()
        try:
            days = int(parts[1])
        except Exception:
            return "❌ Error: Days must be an integer. Example: 'RELIANCE 15'"

        if days <= 0 or days > 365:
            return "❌ Error: Days must be between 1 and 365."

        ticker = yf.Ticker(f"{symbol}")
        df = ticker.history(period="6mo", interval="1d")

        if df is None or df.empty:
            return f"❌ Error: Cannot fetch data for {symbol}"

        if len(df) < 30:
            return f"❌ Error: Not enough data for {symbol}. Need at least 30 data points."

        # Prepare data for Prophet
        data = df.reset_index()[["Date", "Close"]].copy()
        data.columns = ["ds", "y"]
        data["ds"] = pd.to_datetime(data["ds"]).dt.tz_localize(None)

        model = Prophet(daily_seasonality=False, yearly_seasonality=False)
        model.fit(data)

        future = model.make_future_dataframe(periods=days)
        forecast = model.predict(future)

        future_forecast = forecast.tail(days)[["ds", "yhat", "yhat_lower", "yhat_upper"]]

        summary_lines = [
            f"Stock Price Prediction: {symbol}",
            "─────────────────────────────────",
            f"Prediction Period: {days} days",
            "Predictions:"
        ]

        for _, row in future_forecast.iterrows():
            date = pd.to_datetime(row["ds"]).strftime("%Y-%m-%d")
            predicted = float(row["yhat"])
            lower = float(row["yhat_lower"])
            upper = float(row["yhat_upper"])
            summary_lines.append(f"\n{date}: ₹{predicted:.2f} (Range: ₹{lower:.2f} - ₹{upper:.2f})")

        return "\n".join(summary_lines)

    except Exception as e:
        return f"❌ Error in predict_stock_price: {str(e)}"


@tool
def analyze_trend(query: str) -> str:
    """
    Analyze stock trend (Bullish/Bearish/Neutral) using LLM reasoning.
    Input: "RELIANCE"
    """
    try:
        if not query or not isinstance(query, str):
            return "❌ Invalid query. Example: 'RELIANCE'"

        symbol = query.strip().upper()
        ticker = yf.Ticker(f"{symbol}")
        df = ticker.history(period="3mo", interval="1d")

        if df is None or df.empty:
            return f"❌ Error: Cannot fetch data for {symbol}"

        df['SMA_20'] = df['Close'].rolling(window=20, min_periods=1).mean()
        df['SMA_50'] = df['Close'].rolling(window=50, min_periods=1).mean()

        latest_price = float(df['Close'].iloc[-1])
        sma_20 = float(df['SMA_20'].iloc[-1])
        sma_50 = float(df['SMA_50'].iloc[-1])
        change_3m = ((df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0] * 100)

        technical_data = (
            f"Symbol: {symbol}\n"
            f"Latest Price: ₹{latest_price:.2f}\n"
            f"3-Month Change: {change_3m:+.2f}%\n"
            f"SMA 20: ₹{sma_20:.2f}\n"
            f"SMA 50: ₹{sma_50:.2f}\n"
            f"Highest (3m): ₹{float(df['High'].max()):.2f}\n"
            f"Lowest (3m): ₹{float(df['Low'].min()):.2f}\n"
        )

        prompt = f"""
You are the Trend Analysis Expert. Analyze this stock data and determine the trend:

{technical_data}

Based on technical indicators and price action, provide:
1. Trend (Bullish/Bearish/Neutral)
2. Key Reasons (3 clear bullet points)
3. Risk Level (Low/Medium/High)

Be concise and factual.
"""
        response = llm.invoke(prompt)
        # handle both object and string return possibilities
        if hasattr(response, "content"):
            return response.content
        return str(response)

    except Exception as e:
        return f"❌ Error in analyze_trend: {str(e)}"


@tool
def calculate_stoploss(query: str) -> str:
    """
    Calculate recommended stop loss using LLM reasoning.
    Input: "RELIANCE BULLISH" or "TCS BEARISH"
    """
    try:
        if not query or not isinstance(query, str):
            return "❌ Invalid query. Example: 'RELIANCE BULLISH'"

        parts = query.strip().split()
        if len(parts) < 2:
            return "❌ Error: Format should be 'SYMBOL TREND'. Example: 'RELIANCE BULLISH'"

        symbol = parts[0].upper()
        trend = parts[1].upper()

        ticker = yf.Ticker(f"{symbol}")
        df = ticker.history(period="1mo", interval="1d")

        if df is None or df.empty:
            return f"❌ Error: Cannot fetch data for {symbol}"

        current_price = float(df['Close'].iloc[-1])
        volatility = float(df['Close'].pct_change().std() * 100)

        prompt = f"""
You are a risk management expert. Calculate stop loss for this stock:

Symbol: {symbol}
Current Price: ₹{current_price:.2f}
Trend: {trend}
Volatility: {volatility:.2f}%

Determine:
1. Appropriate buffer (2%-10% based on volatility)
2. Stop Loss Price
3. Risk Justification

For {trend}:
- If BULLISH: SL = Price - (Price * buffer%)
- If BEARISH: SL = Price + (Price * buffer%)
- If NEUTRAL: SL = Price ± (Price * 5%)

Respond with:
Stop Loss: ₹XX.XX
Buffer: X%
Risk Level: Low/Medium/High
Justification: ...
Future Insights for investment with reasoning: ...
"""
        response = llm.invoke(prompt)
        if hasattr(response, "content"):
            return response.content
        return str(response)

    except Exception as e:
        return f"❌ Error in calculate_stoploss: {str(e)}"

