import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlite3
import hashlib
import threading
import time
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import warnings
warnings.filterwarnings('ignore')

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="CoinCast",
    page_icon="🪙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- SESSION STATE INIT ----------
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = None
if 'selected_coin' not in st.session_state:
    st.session_state.selected_coin = 'bitcoin'
if 'currency' not in st.session_state:
    st.session_state.currency = 'GBP'
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'
if 'last_update' not in st.session_state:
    st.session_state.last_update = datetime.now()
if 'price_cache' not in st.session_state:
    st.session_state.price_cache = {}
if 'forex_rates' not in st.session_state:
    st.session_state.forex_rates = {'GBP': 1, 'USD': 1.27, 'EUR': 1.17}  # fallback

# ---------- DATABASE SETUP ----------
def init_db():
    conn = sqlite3.connect('coincast.db')
    c = conn.cursor()
    # Users table
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (username TEXT PRIMARY KEY, password TEXT)''')
    # Portfolio table
    c.execute('''CREATE TABLE IF NOT EXISTS portfolio
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT,
                  coin_id TEXT,
                  amount_invested REAL,
                  entry_price REAL,
                  currency TEXT,
                  FOREIGN KEY(username) REFERENCES users(username))''')
    # Alerts table
    c.execute('''CREATE TABLE IF NOT EXISTS alerts
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT,
                  coin_id TEXT,
                  target_price REAL,
                  currency TEXT,
                  triggered INTEGER DEFAULT 0,
                  FOREIGN KEY(username) REFERENCES users(username))''')
    conn.commit()
    conn.close()

init_db()

# ---------- HELPER FUNCTIONS ----------
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username, password):
    conn = sqlite3.connect('coincast.db')
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users VALUES (?, ?)", (username, hash_password(password)))
        conn.commit()
        return True
    except:
        return False
    finally:
        conn.close()

def login_user(username, password):
    conn = sqlite3.connect('coincast.db')
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username=? AND password=?", (username, hash_password(password)))
    user = c.fetchone()
    conn.close()
    return user is not None

def get_portfolio(username):
    conn = sqlite3.connect('coincast.db')
    df = pd.read_sql_query("SELECT * FROM portfolio WHERE username=?", conn, params=(username,))
    conn.close()
    return df

def add_to_portfolio(username, coin_id, amount, entry_price, currency):
    conn = sqlite3.connect('coincast.db')
    c = conn.cursor()
    c.execute("INSERT INTO portfolio (username, coin_id, amount_invested, entry_price, currency) VALUES (?,?,?,?,?)",
              (username, coin_id, amount, entry_price, currency))
    conn.commit()
    conn.close()

def remove_from_portfolio(portfolio_id):
    conn = sqlite3.connect('coincast.db')
    c = conn.cursor()
    c.execute("DELETE FROM portfolio WHERE id=?", (portfolio_id,))
    conn.commit()
    conn.close()

def get_alerts(username):
    conn = sqlite3.connect('coincast.db')
    df = pd.read_sql_query("SELECT * FROM alerts WHERE username=? AND triggered=0", conn, params=(username,))
    conn.close()
    return df

def add_alert(username, coin_id, target_price, currency):
    conn = sqlite3.connect('coincast.db')
    c = conn.cursor()
    c.execute("INSERT INTO alerts (username, coin_id, target_price, currency) VALUES (?,?,?,?)",
              (username, coin_id, target_price, currency))
    conn.commit()
    conn.close()

def trigger_alert(alert_id):
    conn = sqlite3.connect('coincast.db')
    c = conn.cursor()
    c.execute("UPDATE alerts SET triggered=1 WHERE id=?", (alert_id,))
    conn.commit()
    conn.close()

# ---------- API CLIENTS ----------
@st.cache_data(ttl=60)
def get_top_coins(currency='gbp', per_page=90):
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        'vs_currency': currency,
        'order': 'market_cap_desc',
        'per_page': per_page,
        'page': 1,
        'sparkline': False,
        'price_change_percentage': '24h,7d,30d,1y'
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        return r.json()
    except:
        return []

@st.cache_data(ttl=60)
def get_coin_data(coin_id, currency='gbp'):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}"
    params = {
        'localization': False,
        'tickers': False,
        'market_data': True,
        'community_data': True,
        'developer_data': False
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
        market = data['market_data']
        community = data.get('community_data', {})
        return {
            'name': data['name'],
            'symbol': data['symbol'].upper(),
            'current_price': market['current_price'][currency],
            'market_cap': market['market_cap'][currency],
            'total_volume': market['total_volume'][currency],
            'high_24h': market['high_24h'][currency],
            'low_24h': market['low_24h'][currency],
            'price_change_24h': market['price_change_percentage_24h'],
            'price_change_7d': market.get('price_change_percentage_7d', 0),
            'price_change_30d': market.get('price_change_percentage_30d', 0),
            'price_change_1y': market.get('price_change_percentage_1y', 0),
            'total_holders': community.get('twitter_followers', 0)  # mock "Total Users"
        }
    except:
        return None

@st.cache_data(ttl=300)
def get_historical_data(coin_id, days=30, currency='gbp'):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {
        'vs_currency': currency,
        'days': days,
        'interval': 'daily' if days > 90 else 'hourly'
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
        prices = data['prices']  # [timestamp, price]
        df = pd.DataFrame(prices, columns=['timestamp', 'price'])
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df[['date', 'price']]
    except:
        # fallback synthetic data
        dates = pd.date_range(end=datetime.now(), periods=days)
        prices = np.random.normal(50000, 5000, days).cumsum()
        return pd.DataFrame({'date': dates, 'price': prices})

def get_forex_rates(base='GBP'):
    # Free API from exchangerate-api.com
    url = f"https://api.exchangerate-api.com/v4/latest/{base}"
    try:
        r = requests.get(url, timeout=5)
        return r.json()['rates']
    except:
        return st.session_state.forex_rates

def get_news_sentiment(coin_name):
    # Mock sentiment based on random or simple rule
    # In production, use NewsAPI and NLP
    import random
    sentiments = ['Bullish', 'Bearish', 'Neutral']
    return random.choice(sentiments)

# ---------- BACKGROUND PRICE UPDATER & ALERT CHECKER ----------
def update_prices_and_check_alerts():
    while True:
        time.sleep(60)  # update every minute
        # Get all active alerts
        conn = sqlite3.connect('coincast.db')
        c = conn.cursor()
        c.execute("SELECT id, username, coin_id, target_price, currency FROM alerts WHERE triggered=0")
        alerts = c.fetchall()
        conn.close()

        for alert in alerts:
            alert_id, username, coin_id, target, cur = alert
            # fetch current price
            data = get_coin_data(coin_id, cur.lower())
            if data and data['current_price'] >= target:
                # Trigger alert (store in session to show in UI)
                # For simplicity, we'll just mark triggered in DB and set a flag
                trigger_alert(alert_id)
                # We'll show notification via Streamlit's session state
                # but Streamlit is not thread-safe, so we use a queue or just mark DB
                # The UI will check for triggered alerts on each rerun

# Start background thread (only once)
if 'background_thread' not in st.session_state:
    thread = threading.Thread(target=update_prices_and_check_alerts, daemon=True)
    thread.start()
    st.session_state.background_thread = True

# ---------- TECHNICAL INDICATORS ----------
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(series, fast=12, slow=26, signal=9):
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

def calculate_bollinger(series, window=20, num_std=2):
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    upper = rolling_mean + (rolling_std * num_std)
    lower = rolling_mean - (rolling_std * num_std)
    return upper, rolling_mean, lower

def preprocess_data(df):
    # Remove missing, duplicates, outliers
    df = df.dropna().drop_duplicates(subset=['date'])
    # Simple outlier removal (z-score > 3)
    from scipy import stats
    z_scores = np.abs(stats.zscore(df['price']))
    df = df[(z_scores < 3)]
    return df

# ---------- MACHINE LEARNING PREDICTION ----------
def predict_with_lstm(df, days_to_predict=1):
    # Prepare data
    prices = df['price'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_prices = scaler.fit_transform(prices)

    # Create sequences
    def create_sequences(data, seq_length=60):
        X, y = [], []
        for i in range(seq_length, len(data)):
            X.append(data[i-seq_length:i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)

    seq_length = min(60, len(scaled_prices)//2)
    if len(scaled_prices) < seq_length+1:
        return None, "Insufficient data for LSTM"

    X, y = create_sequences(scaled_prices, seq_length)
    if len(X) == 0:
        return None, "Insufficient sequences"

    # Reshape for LSTM (samples, timesteps, features)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # Build LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(seq_length, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    # Train (quick, for demo)
    model.fit(X, y, epochs=5, batch_size=32, verbose=0)

    # Predict next day
    last_sequence = scaled_prices[-seq_length:].reshape(1, seq_length, 1)
    pred_scaled = model.predict(last_sequence, verbose=0)[0,0]
    pred_price = scaler.inverse_transform([[pred_scaled]])[0,0]

    current_price = prices[-1,0]
    return pred_price, current_price

def predict_with_linear(df):
    if len(df) < 5:
        return None, "Insufficient data"
    df = df.reset_index(drop=True)
    X = np.arange(len(df)).reshape(-1, 1)
    y = df['price'].values
    model = LinearRegression()
    model.fit(X, y)
    pred = model.predict([[len(df)]])[0]
    return pred, df['price'].iloc[-1]

# ---------- UI CUSTOMIZATION ----------
def set_theme(theme):
    if theme == 'dark':
        st.markdown("""
        <style>
            .stApp { background-color: #1E1E1E; color: white; }
            .css-1r6slb0, .css-12oz5g7 { background-color: #2E2E2E !important; color: white; }
            h1, h2, h3 { color: #CBC3E3; }
        </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
            .stApp { background-color: #CBC3E3; }
            .css-1r6slb0, .css-12oz5g7 { background-color: white !important; }
            h1, h2, h3 { color: #4B0082; }
        </style>
        """, unsafe_allow_html=True)

# ---------- SIDEBAR (AUTH & SETTINGS) ----------
with st.sidebar:
    st.image("https://via.placeholder.com/150x50?text=CoinCast+Pro", use_column_width=True)
    st.title("🔐 Account")

    if not st.session_state.logged_in:
        tab1, tab2 = st.tabs(["Login", "Sign Up"])
        with tab1:
            login_user = st.text_input("Username", key="login_user")
            login_pass = st.text_input("Password", type="password", key="login_pass")
            if st.button("Login"):
                if login_user and login_pass and login_user(login_user, login_pass):
                    st.session_state.logged_in = True
                    st.session_state.username = login_user
                    st.rerun()
                else:
                    st.error("Invalid credentials")
        with tab2:
            reg_user = st.text_input("Username", key="reg_user")
            reg_pass = st.text_input("Password", type="password", key="reg_pass")
            if st.button("Register"):
                if reg_user and reg_pass:
                    if register_user(reg_user, reg_pass):
                        st.success("Registered! Please log in.")
                    else:
                        st.error("Username exists")
    else:
        st.success(f"Logged in as {st.session_state.username}")
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.username = None
            st.rerun()

    st.markdown("---")
    st.header("⚙️ Settings")
    currency = st.selectbox("Currency", ["GBP", "USD", "EUR"], index=0)
    if currency != st.session_state.currency:
        st.session_state.currency = currency
        st.session_state.forex_rates = get_forex_rates(currency)
        st.rerun()

    theme = st.radio("Theme", ["light", "dark"], index=0 if st.session_state.theme=='light' else 1)
    if theme != st.session_state.theme:
        st.session_state.theme = theme
        set_theme(theme)
        st.rerun()

    st.markdown("---")
    st.caption(f"Last updated: {st.session_state.last_update.strftime('%H:%M:%S')}")

# Apply theme
set_theme(st.session_state.theme)

# ---------- MAIN CONTENT (Multi‑page via radio) ----------
page = st.sidebar.radio("Navigation", ["Leaderboard", "Portfolio", "Alerts", "Search", "Settings"])

# ---------- LEADERBOARD PAGE ----------
if page == "Leaderboard":
    st.title("🏆 Market Leaderboard")
    st.markdown("Top 90 cryptocurrencies by market cap")

    currency_lower = st.session_state.currency.lower()
    coins = get_top_coins(currency_lower, 90)

    if coins:
        # Create DataFrame
        df = pd.DataFrame(coins)
        df['name'] = df['name'].astype(str)
        df['symbol'] = df['symbol'].astype(str).str.upper()
        df['current_price'] = df['current_price'].round(2)
        df['market_cap'] = df['market_cap'].apply(lambda x: f"{x:,.0f}")
        df['price_change_24h'] = df['price_change_percentage_24h'].round(2)
        df['price_change_7d'] = df['price_change_percentage_7d_in_currency'].round(2)
        df['price_change_30d'] = df['price_change_percentage_30d_in_currency'].round(2)
        df['price_change_1y'] = df['price_change_percentage_1y_in_currency'].round(2)

        # Show as interactive table with selection
        display_df = df[['name', 'symbol', 'current_price', 'market_cap', 'price_change_24h']]
        display_df.columns = ['Name', 'Symbol', f'Price ({st.session_state.currency})', 'Market Cap', '24h %']

        # Add select buttons
        for i, row in df.iterrows():
            col1, col2, col3, col4, col5, col6 = st.columns([2,1,1,1,1,1])
            col1.write(f"{row['name']} ({row['symbol']})")
            col2.write(f"{row['current_price']:,.2f}")
            col3.write(f"{row['market_cap']}")
            col4.write(f"{row['price_change_24h']:+.2f}%")
            if col5.button("Select", key=f"select_{row['id']}"):
                st.session_state.selected_coin = row['id']
                st.rerun()
            if col6.button("Alert", key=f"alert_{row['id']}"):
                st.session_state['alert_coin'] = row['id']
                st.session_state['alert_name'] = row['name']
                st.rerun()

        # Quick stats
        st.markdown("---")
        st.subheader("📊 Market Summary")
        total_mcap = sum([c['market_cap'] for c in coins if c['market_cap']])
        avg_price = np.mean([c['current_price'] for c in coins])
        st.metric("Total Market Cap", f"{total_mcap:,.0f} {st.session_state.currency}")
        st.metric("Average Price", f"{avg_price:,.2f} {st.session_state.currency}")

    else:
        st.error("Failed to load leaderboard. Check your internet connection.")

# ---------- COIN DETAIL PAGE (shown after selecting a coin) ----------
# This will appear as an overlay or separate section when a coin is selected
if st.session_state.selected_coin:
    st.markdown("---")
    st.header(f"📈 {st.session_state.selected_coin.capitalize()} Details")
    coin_data = get_coin_data(st.session_state.selected_coin, st.session_state.currency.lower())

    if coin_data:
        # Main price
        st.subheader(f"{coin_data['name']} ({coin_data['symbol']})")
        st.markdown(f"### 💰 {coin_data['current_price']:,.2f} {st.session_state.currency}")

        # Price changes (mockup: 24h, 72h, 14d, 1y) – we'll map 72h to 3d (not directly available)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("24h", f"{coin_data['price_change_24h']:+.2f}%")
        col2.metric("72h", f"{coin_data.get('price_change_7d', 0)/7*3:+.2f}%")  # approx
        col3.metric("14d", f"{coin_data.get('price_change_30d', 0)/30*14:+.2f}%")
        col4.metric("1y", f"{coin_data['price_change_1y']:+.2f}%")

        # Market Stats (matching mockup)
        st.markdown("#### 📊 Market Stats (GBP)")
        mc1, mc2, mc3, mc4, mc5, mc6 = st.columns(6)
        mc1.metric("Currency Price", f"{coin_data['current_price']:,.2f}")
        mc2.metric("Median Cap", "N/A")  # mock
        mc3.metric("24h Volume", f"{coin_data['total_volume']:,.0f}")
        mc4.metric("24h High", f"{coin_data['high_24h']:,.2f}")
        mc5.metric("24h Low", f"{coin_data['low_24h']:,.2f}")
        mc6.metric("Closing Today", f"{coin_data['current_price']:,.2f}")  # approximate

        # Total Users (mock)
        st.metric("Total Users", f"{coin_data.get('total_holders', 0):,}")

        # Historical chart
        st.markdown("#### 📈 Price History & Prediction")
        timeframe = st.selectbox("Timeframe", ["1D", "7D", "1M", "3M", "1Y"], index=2)
        days_map = {"1D":1, "7D":7, "1M":30, "3M":90, "1Y":365}
        hist_df = get_historical_data(st.session_state.selected_coin, days_map[timeframe], st.session_state.currency.lower())
        hist_df = preprocess_data(hist_df)

        # Technical indicators
        hist_df['RSI'] = calculate_rsi(hist_df['price'])
        hist_df['MACD'], hist_df['Signal'], hist_df['Histogram'] = calculate_macd(hist_df['price'])
        hist_df['BB_upper'], hist_df['BB_mid'], hist_df['BB_lower'] = calculate_bollinger(hist_df['price'])

        # Plot with subplots
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                            vertical_spacing=0.05, row_heights=[0.5,0.25,0.25])
        # Price & BB
        fig.add_trace(go.Scatter(x=hist_df['date'], y=hist_df['price'], mode='lines', name='Price'), row=1, col=1)
        fig.add_trace(go.Scatter(x=hist_df['date'], y=hist_df['BB_upper'], mode='lines', line=dict(dash='dash'), name='BB Upper'), row=1, col=1)
        fig.add_trace(go.Scatter(x=hist_df['date'], y=hist_df['BB_lower'], mode='lines', line=dict(dash='dash'), name='BB Lower'), row=1, col=1)
        # RSI
        fig.add_trace(go.Scatter(x=hist_df['date'], y=hist_df['RSI'], mode='lines', name='RSI'), row=2, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="green", row=2, col=1)
        # MACD
        fig.add_trace(go.Scatter(x=hist_df['date'], y=hist_df['MACD'], mode='lines', name='MACD'), row=3, col=1)
        fig.add_trace(go.Scatter(x=hist_df['date'], y=hist_df['Signal'], mode='lines', name='Signal'), row=3, col=1)
        fig.add_trace(go.Bar(x=hist_df['date'], y=hist_df['Histogram'], name='Histogram'), row=3, col=1)

        fig.update_layout(height=600, showlegend=False, title=f"{coin_data['name']} Price & Indicators")
        st.plotly_chart(fig, use_container_width=True)

        # Prediction
        st.markdown("#### 🔮 Price Prediction")
        try:
            pred_price, curr_price = predict_with_lstm(hist_df)
            if pred_price is None:
                pred_price, curr_price = predict_with_linear(hist_df)
        except:
            pred_price, curr_price = predict_with_linear(hist_df)

        if pred_price:
            st.write(f"**Predicted next day price:** {pred_price:,.2f} {st.session_state.currency}")
            if pred_price > curr_price:
                st.success("📈 **Bullish** (predicted > current)")
            elif pred_price < curr_price:
                st.error("📉 **Bearish** (predicted < current)")
            else:
                st.info("➡️ **Sideways**")
        else:
            st.warning("Not enough data for prediction")

        # Sentiment
        st.markdown("#### 📰 News Sentiment")
        sentiment = get_news_sentiment(coin_data['name'])
        if sentiment == 'Bullish':
            st.success(f"Sentiment: {sentiment} 😊")
        elif sentiment == 'Bearish':
            st.error(f"Sentiment: {sentiment} 😞")
        else:
            st.info(f"Sentiment: {sentiment} 😐")

        # Products (mock buttons)
        st.markdown("#### 🛒 Products for " + coin_data['name'])
        cols = st.columns(5)
        cols[0].button("Buy", key="buy", disabled=True)
        cols[1].button("Sell", key="sell", disabled=True)
        cols[2].button("Trade", key="trade", disabled=True)
        cols[3].button("Deposit", key="deposit", disabled=True)
        cols[4].button("Withdraw", key="withdraw", disabled=True)

        # Add to Portfolio from this page
        if st.session_state.logged_in:
            st.markdown("#### ➕ Add to Portfolio")
            with st.form("add_portfolio_form"):
                amount_invested = st.number_input("Amount invested (GBP)", min_value=0.0, step=10.0)
                entry_price = st.number_input("Entry price (GBP)", min_value=0.0, step=0.01, value=coin_data['current_price'])
                submitted = st.form_submit_button("Add to Portfolio")
                if submitted and amount_invested > 0:
                    add_to_portfolio(st.session_state.username, st.session_state.selected_coin,
                                     amount_invested, entry_price, st.session_state.currency)
                    st.success("Added!")
        else:
            st.info("Login to add to portfolio")

    else:
        st.error("Could not load coin data")

# ---------- PORTFOLIO PAGE ----------
if page == "Portfolio":
    st.title("📁 My Portfolio")
    if not st.session_state.logged_in:
        st.warning("Please login to view your portfolio")
    else:
        portfolio_df = get_portfolio(st.session_state.username)
        if portfolio_df.empty:
            st.info("Your portfolio is empty. Add coins from the leaderboard.")
        else:
            total_invested = 0
            total_current = 0
            portfolio_list = []
            for _, row in portfolio_df.iterrows():
                coin_data = get_coin_data(row['coin_id'], st.session_state.currency.lower())
                if coin_data:
                    current_price = coin_data['current_price']
                    # Convert entry price if currency differs? Assume same currency for now
                    quantity = row['amount_invested'] / row['entry_price']
                    current_value = quantity * current_price
                    profit = current_value - row['amount_invested']
                    profit_pct = (profit / row['amount_invested']) * 100
                    portfolio_list.append({
                        'id': row['id'],
                        'coin': row['coin_id'].capitalize(),
                        'invested': row['amount_invested'],
                        'entry': row['entry_price'],
                        'current': current_price,
                        'value': current_value,
                        'profit': profit,
                        'profit_pct': profit_pct
                    })
                    total_invested += row['amount_invested']
                    total_current += current_value

            df_port = pd.DataFrame(portfolio_list)
            st.dataframe(df_port[['coin', 'invested', 'entry', 'current', 'value', 'profit', 'profit_pct']])

            st.metric("Total Invested", f"{total_invested:,.2f} {st.session_state.currency}")
            st.metric("Current Value", f"{total_current:,.2f} {st.session_state.currency}")
            st.metric("Total P&L", f"{total_current - total_invested:+,.2f} {st.session_state.currency}",
                      delta=f"{((total_current-total_invested)/total_invested*100):+.2f}%" if total_invested else "")

            # Option to remove
            remove_id = st.selectbox("Remove entry", df_port['id'].tolist(), format_func=lambda x: f"ID {x}")
            if st.button("Remove"):
                remove_from_portfolio(remove_id)
                st.rerun()

            # Export
            if st.button("Export to CSV"):
                csv = df_port.to_csv(index=False)
                st.download_button("Download CSV", csv, "portfolio.csv", "text/csv")

# ---------- ALERTS PAGE ----------
if page == "Alerts":
    st.title("🔔 Price Alerts")
    if not st.session_state.logged_in:
        st.warning("Login to manage alerts")
    else:
        alerts_df = get_alerts(st.session_state.username)
        if not alerts_df.empty:
            st.dataframe(alerts_df[['coin_id', 'target_price', 'currency']])
            # Check for triggered alerts (background thread marks them triggered)
            # Show if any triggered
            conn = sqlite3.connect('coincast.db')
            triggered = pd.read_sql_query("SELECT * FROM alerts WHERE username=? AND triggered=1", conn, params=(st.session_state.username,))
            conn.close()
            if not triggered.empty:
                st.balloons()
                for _, row in triggered.iterrows():
                    st.success(f"🚨 {row['coin_id']} reached {row['target_price']} {row['currency']}!")
                # Option to clear triggered
                if st.button("Clear triggered alerts"):
                    conn = sqlite3.connect('coincast.db')
                    c = conn.cursor()
                    c.execute("DELETE FROM alerts WHERE username=? AND triggered=1", (st.session_state.username,))
                    conn.commit()
                    conn.close()
                    st.rerun()
        else:
            st.info("No active alerts. Set one from the leaderboard.")

        # Quick add alert
        st.subheader("Set new alert")
        coin_id = st.text_input("Coin ID (e.g., bitcoin)")
        target = st.number_input("Target price", min_value=0.0, step=0.01)
        if st.button("Set Alert") and coin_id and target:
            add_alert(st.session_state.username, coin_id.lower(), target, st.session_state.currency)
            st.success("Alert set!")

# ---------- SEARCH PAGE ----------
if page == "Search":
    st.title("🔍 Search Coins")
    search_term = st.text_input("Enter coin name or symbol")
    if search_term:
        search_term = search_term.lower().strip()
        # Use CoinGecko search
        url = f"https://api.coingecko.com/api/v3/search?query={search_term}"
        try:
            r = requests.get(url)
            data = r.json()
            coins = data.get('coins', [])
            if coins:
                for coin in coins[:10]:  # show top 10 matches
                    st.write(f"**{coin['name']}** ({coin['symbol']})")
                    if st.button("Select", key=f"search_{coin['id']}"):
                        st.session_state.selected_coin = coin['id']
                        st.rerun()
            else:
                st.error("❌ Coin not found")
        except:
            st.error("Search failed")

# ---------- SETTINGS PAGE ----------
if page == "Settings":
    st.title("⚙️ Settings")
    st.write("Configure your preferences")
    # already in sidebar, but here for completeness
    st.write(f"Current currency: {st.session_state.currency}")
    st.write(f"Theme: {st.session_state.theme}")
    if st.button("Clear cache"):
        st.cache_data.clear()
        st.success("Cache cleared")
