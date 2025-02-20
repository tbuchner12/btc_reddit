import streamlit as st
import praw
import requests
import pandas as pd
import datetime
from textblob import TextBlob
from pytrends.request import TrendReq
import plotly.graph_objects as go

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Real-Time Bitcoin Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.title("Real-Time Bitcoin Dashboard")
st.write("This dashboard shows live Bitcoin prices, Reddit sentiment from r/bitcoin, Google search trends for 'bitcoin', and a live candlestick chart of Bitcoin. Additionally, at the bottom there is a linear regression model to predict the price of bitcoin using Reddit Sentiment and Google Trends.")
st.write("Refresh the page to accumulate more historical data."
         "WARNING: Due to rate limits, excessive refreshing will cause Data to NA")

# ---------------------------------
# Initialize session state for historical data
# ---------------------------------
if "data" not in st.session_state:
    st.session_state.data = pd.DataFrame(
        columns=["time", "bitcoin_price", "reddit_sentiment", "google_trends"]
    )

# Record the current timestamp
current_time = datetime.datetime.now()

# -----------------------------
# Section 1: Bitcoin Price Data via CoinGecko
# -----------------------------
st.header("Bitcoin Price (USD)")

def get_bitcoin_price():
    try:
        url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd"
        response = requests.get(url)
        data = response.json()
        price = data["bitcoin"]["usd"]
        return price
    except Exception as e:
        st.error(f"Error fetching Bitcoin price: {e}")
        return None

bitcoin_price = get_bitcoin_price()
if bitcoin_price is not None:
    st.metric("Current Bitcoin Price", f"${bitcoin_price:,.2f}")
else:
    st.metric("Current Bitcoin Price", "N/A")

# -----------------------------
# Section 2: Reddit Sentiment Analysis (r/bitcoin)
# -----------------------------
st.header("Reddit Sentiment Analysis (r/bitcoin)")

# Initialize Reddit API with your credentials
reddit = praw.Reddit(
    client_id="z1wABScVU-4ThC4dqxEJ9Q",
    client_secret="MRTvNBog-bIqbA66-D7O80MkKtgFtw",
    user_agent="myredditapp:v1.0 (by u/Much_Nail3900)"
)

def get_reddit_sentiment():
    try:
        subreddit = reddit.subreddit("bitcoin")
        posts = list(subreddit.hot(limit=10))
        sentiments = []
        for post in posts:
            text = post.title + " " + post.selftext
            if text.strip():
                analysis = TextBlob(text)
                sentiments.append(analysis.sentiment.polarity)
        if sentiments:
            return sum(sentiments) / len(sentiments)
        else:
            return 0
    except Exception as e:
        st.error(f"Error fetching Reddit sentiment: {e}")
        return 0

reddit_sentiment = get_reddit_sentiment()
st.metric("Average Reddit Sentiment", f"{reddit_sentiment:.2f} (scale -1 to 1)")

# -----------------------------
# Section 3: Google Trends Data for 'bitcoin'
# -----------------------------
st.header("Google Search Trends for 'Bitcoin'")

@st.cache_data(ttl=9000)  # Cache for 10 minutes to reduce request frequency
def get_google_trend_value():
    try:
        pytrends = TrendReq(hl='en-US', tz=360)
        kw_list = ['bitcoin']
        pytrends.build_payload(kw_list, cat=0, timeframe='now 1-d', geo='', gprop='')
        trends_df = pytrends.interest_over_time()
        if not trends_df.empty:
            trends_df = trends_df.reset_index()
            return trends_df["bitcoin"].iloc[-1]
        else:
            return None
    except Exception as e:
        st.error(f"Error fetching Google Trends data: {e}")
        return None

google_trends_value = get_google_trend_value()
if google_trends_value is not None:
    st.metric("Google Trends (bitcoin)", google_trends_value)
else:
    st.metric("Google Trends (bitcoin)", "N/A")

# -----------------------------
# Append new data point to the historical DataFrame
# -----------------------------
new_row = {
    "time": current_time,
    "bitcoin_price": bitcoin_price,
    "reddit_sentiment": reddit_sentiment,
    "google_trends": google_trends_value
}
new_row_df = pd.DataFrame([new_row])
st.session_state.data = pd.concat([st.session_state.data, new_row_df], ignore_index=True)

# Prepare the DataFrame for plotting historical trends
df = st.session_state.data.copy()
df["time"] = pd.to_datetime(df["time"])
df.sort_values("time", inplace=True)
df.set_index("time", inplace=True)

# -----------------------------
# Relative Sizes of Metrics in 3 Circles (Bubble Chart)
# -----------------------------
st.header("Relative Sizes of Metrics")

# Set reference values for normalization:
btc_ref = 200000   # Reference maximum Bitcoin price (e.g. $200k)
reddit_ref = 1     # Maximum sentiment magnitude (1)
google_ref = 100   # Maximum Google Trends value

# Ensure that values are available (or set to 0 if not)
btc_val = bitcoin_price if bitcoin_price is not None else 0
# For sentiment, we can use the absolute value (or choose to display the actual signed value)
reddit_val = abs(reddit_sentiment)
google_val = google_trends_value if google_trends_value is not None else 0

# Compute a relative size for each (adjust multipliers to get desired circle sizes)
btc_size = (btc_val / btc_ref) * 200   # Scale factor for Bitcoin price
reddit_size = (reddit_val / reddit_ref) * 200  # Scale factor for sentiment
google_size = (google_val / google_ref) * 200   # Scale factor for Google Trends

# Define positions for each circle along the x-axis
x_positions = [1, 2, 3]
y_positions = [1, 1, 1]  # All on the same horizontal line

# Create a bubble chart with Plotly
fig_bubbles = go.Figure(data=[go.Scatter(
    x=x_positions,
    y=y_positions,
    mode="markers+text",
    marker=dict(
        size=[btc_size, reddit_size, google_size],
        sizemode='area',
        color=["blue", "red", "green"],
        # sizeref and sizemin help control the marker scaling
        sizeref=2. * max(btc_size, reddit_size, google_size) / (100**2),
        sizemin=10
    ),
    text=[
        "Bitcoin Price<br>${:,.0f}".format(btc_val),
        "Reddit Sentiment<br>{:.2f}".format(reddit_sentiment),
        "Google Trends<br>{}".format(google_val)
    ],
    textposition="middle center"
)])

# Remove x and y axis labels/ticks for a cleaner look
fig_bubbles.update_layout(
    xaxis=dict(visible=False),
    yaxis=dict(visible=False),
    title="Relative Sizes of Metrics"
)

st.plotly_chart(fig_bubbles, use_container_width=True)




# -----------------------------
# Section 4: Live Bitcoin Candlestick Chart (OHLC)
# -----------------------------
st.header("Live Bitcoin Candlestick Chart (Last 1 Day)")

def get_bitcoin_ohlc():
    try:
        url = "https://api.coingecko.com/api/v3/coins/bitcoin/ohlc?vs_currency=usd&days=1"
        response = requests.get(url)
        data = response.json()
        df_ohlc = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close"])
        df_ohlc["time"] = pd.to_datetime(df_ohlc["timestamp"], unit="ms")
        return df_ohlc
    except Exception as e:
        st.error(f"Error fetching Bitcoin OHLC data: {e}")
        return pd.DataFrame()

ohlc_df = get_bitcoin_ohlc()
if not ohlc_df.empty:
    fig_candle = go.Figure(data=[go.Candlestick(
        x=ohlc_df["time"],
        open=ohlc_df["open"],
        high=ohlc_df["high"],
        low=ohlc_df["low"],
        close=ohlc_df["close"],
        name="BTC Candlestick"
    )])
    fig_candle.update_layout(
        title="Live Bitcoin Candlestick Chart (Last 1 Day)",
        xaxis_title="Time",
        yaxis_title="Price (USD)"
    )
    st.plotly_chart(fig_candle, use_container_width=True)
else:
    st.write("No OHLC data available.")

st.write("Refresh the page to update data and build the time series.")

import numpy as np
import statsmodels.api as sm
import pandas as pd
import datetime

st.header("BTC Price Prediction Model (Using Live Data)")

if len(df) < 5:
    st.write("Not enough historical data for prediction. Please refresh the page a few more times to accumulate more data.")
else:
    # Use a dropdown (selectbox) for selecting the number of days into the future.
    days_out = st.selectbox(
        "Select number of days into the future for prediction",
        options=list(range(1, 31)),
        index=6  # Default to 7 days out if available
    )
    
    # Create a new column 'future_price' that is the bitcoin price shifted by the number of days.
    df["future_price"] = df["bitcoin_price"].shift(-days_out)
    
    # Drop rows where the future price is missing (i.e. the most recent rows)
    df_model = df.dropna(subset=["future_price"]).copy()
    
    # Convert columns to numeric to avoid dtype issues
    df_model["reddit_sentiment"] = pd.to_numeric(df_model["reddit_sentiment"], errors="coerce")
    df_model["google_trends"] = pd.to_numeric(df_model["google_trends"], errors="coerce")
    df_model["future_price"] = pd.to_numeric(df_model["future_price"], errors="coerce")
    
    if len(df_model) < 2:
        st.write("Not enough data points after shifting for prediction. Please refresh again.")
    else:
        # Use Reddit sentiment and Google Trends as predictors
        X = df_model[["reddit_sentiment", "google_trends"]].astype(float)
        # Force the addition of a constant (intercept)
        X = sm.add_constant(X, has_constant='add')
        y = df_model["future_price"].astype(float)
        
        # Build the regression model using statsmodels
        model = sm.OLS(y, X).fit()
        
        # Use the most recent row's predictors from df
        current_row = df.iloc[-1]
        current_reddit = float(current_row["reddit_sentiment"])
        current_google = float(current_row["google_trends"])
        X_new = pd.DataFrame({
            "const": [1.0],
            "reddit_sentiment": [current_reddit],
            "google_trends": [current_google]
        })
        X_new = X_new.astype(float)
        
        predicted_price = model.predict(X_new)[0]
        
        # Compute the predicted date (today + days_out)
        predicted_date = datetime.date.today() + datetime.timedelta(days=days_out)
        
        st.subheader("Predicted Future Bitcoin Price")
        st.write(f"In {days_out} day(s), through Reddit / Google data, the model predicts Bitcoins price for {predicted_date} to be: **${predicted_price:,.2f}**")
