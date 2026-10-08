
import os
import time
import math
import logging
from datetime import datetime, timezone

import requests
import pandas as pd
import numpy as np


# =========================================================
# CONFIG
# =========================================================

TWELVE_DATA_API_KEY = os.getenv("TWELVE_DATA_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

SYMBOL = "XAU/USD"

# We poll every 5 minutes.
POLL_SECONDS = 300

# User's XAU/USD pip convention:
# 0.10 price movement = 1 pip
PIP_SIZE = 0.10

# Minimum score required before sending a trade.
MIN_SCORE = 82

# Risk/reward targets
TP1_RR = 1.0
TP2_RR = 2.0

# Maximum spread filter cannot be obtained from Twelve Data,
# so we do not fake a broker spread.
# This bot uses market-data price only.

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger("gold-ai-bot")

last_signal_key = None
active_trade = None


# =========================================================
# VALIDATION
# =========================================================

def validate_config():
    missing = []

    if not TWELVE_DATA_API_KEY:
        missing.append("TWELVE_DATA_API_KEY")

    if not TELEGRAM_BOT_TOKEN:
        missing.append("TELEGRAM_BOT_TOKEN")

    if not TELEGRAM_CHAT_ID:
        missing.append("TELEGRAM_CHAT_ID")

    if missing:
        raise RuntimeError(
            "Missing environment variables: " + ", ".join(missing)
        )


# =========================================================
# TELEGRAM
# =========================================================

def telegram_send(message):
    url = (
        f"https://api.telegram.org/bot"
        f"{TELEGRAM_BOT_TOKEN}/sendMessage"
    )

    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message
    }

    try:
        response = requests.post(
            url,
            json=payload,
            timeout=20
        )

        if response.status_code != 200:
            logger.error(
                "Telegram error: %s",
                response.text
            )
            return False

        return True

    except Exception as e:
        logger.exception("Telegram exception: %s", e)
        return False


# =========================================================
# TWELVE DATA
# =========================================================

def get_candles(interval="5min", outputsize=200):
    url = "https://api.twelvedata.com/time_series"

    params = {
        "symbol": SYMBOL,
        "interval": interval,
        "outputsize": outputsize,
        "apikey": TWELVE_DATA_API_KEY,
        "timezone": "UTC",
        "format": "JSON"
    }

    response = requests.get(
        url,
        params=params,
        timeout=30
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"Twelve Data HTTP {response.status_code}: "
            f"{response.text[:500]}"
        )

    data = response.json()

    if data.get("status") == "error":
        raise RuntimeError(
            data.get("message", "Twelve Data error")
        )

    values = data.get("values")

    if not values:
        raise RuntimeError("No market data returned.")

    df = pd.DataFrame(values)

    required = [
        "datetime",
        "open",
        "high",
        "low",
        "close"
    ]

    for col in required:
        if col not in df.columns:
            raise RuntimeError(
                f"Missing column from Twelve Data: {col}"
            )

    df["datetime"] = pd.to_datetime(
        df["datetime"],
        utc=True
    )

    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    df = df.dropna(
        subset=[
            "datetime",
            "open",
            "high",
            "low",
            "close"
        ]
    )

    df = df.sort_values("datetime")
    df = df.drop_duplicates(
        subset=["datetime"]
    )

    df = df.set_index("datetime")

    return df


# =========================================================
# BUILD M15 FROM M5
# =========================================================

def build_m15(m5):
    m15 = pd.DataFrame()

    m15["open"] = m5["open"].resample(
        "15min"
    ).first()

    m15["high"] = m5["high"].resample(
        "15min"
    ).max()

    m15["low"] = m5["low"].resample(
        "15min"
    ).min()

    m15["close"] = m5["close"].resample(
        "15min"
    ).last()

    m15 = m15.dropna()

    return m15


# =========================================================
# INDICATORS
# =========================================================

def ema(series, period):
    return series.ewm(
        span=period,
        adjust=False
    ).mean()


def atr(df, period=14):
    previous_close = df["close"].shift(1)

    tr1 = df["high"] - df["low"]

    tr2 = (
        df["high"] -
        previous_close
    ).abs()

    tr3 = (
        df["low"] -
        previous_close
    ).abs()

    true_range = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    return true_range.rolling(
        period
    ).mean()


def rsi(series, period=14):
    delta = series.diff()

    gain = delta.clip(
        lower=0
    )

    loss = -delta.clip(
        upper=0
    )

    avg_gain = gain.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(
        0,
        np.nan
    )

    return 100 - (
        100 / (1 + rs)
    )


def add_indicators(df):
    df = df.copy()

    df["ema20"] = ema(
        df["close"],
        20
    )

    df["ema50"] = ema(
        df["close"],
        50
    )

    df["ema200"] = ema(
        df["close"],
        200
    )

    df["atr"] = atr(
        df,
        14
    )

    df["rsi"] = rsi(
        df["close"],
        14
    )

    return df


# =========================================================
# MARKET STRUCTURE
# =========================================================

def structure(df):
    if len(df) < 30:
        return "UNKNOWN"

    recent = df.tail(30)

    old_high = recent["high"].iloc[:-10].max()
    old_low = recent["low"].iloc[:-10].min()

    last_close = recent["close"].iloc[-1]

    if last_close > old_high:
        return "BULLISH_BOS"

    if last_close < old_low:
        return "BEARISH_BOS"

    ema20 = recent["ema20"].iloc[-1]
    ema50 = recent["ema50"].iloc[-1]

    if ema20 > ema50:
        return "BULLISH"

    if ema20 < ema50:
        return "BEARISH"

    return "RANGE"


# =========================================================
# SUPPORT / RESISTANCE
# =========================================================

def support_resistance(df, lookback=50):
    data = df.tail(lookback)

    support = float(
        data["low"].min()
    )

    resistance = float(
        data["high"].max()
    )

    return support, resistance


# =========================================================
# SUPPLY / DEMAND APPROXIMATION
# =========================================================

def supply_demand(df, lookback=30):
    data = df.tail(lookback)

    demand = float(
        data["low"].rolling(5).min().min()
    )

    supply = float(
        data["high"].rolling(5).max().max()
    )

    return demand, supply


# =========================================================
# LIQUIDITY SWEEP
# =========================================================

def liquidity_sweep(df):
    if len(df) < 10:
        return None

    last = df.iloc[-1]
    previous = df.iloc[-6:-1]

    previous_high = previous["high"].max()
    previous_low = previous["low"].min()

    # Bearish liquidity sweep:
    # price takes previous high and closes back below it.
    if (
        last["high"] > previous_high
        and last["close"] < previous_high
    ):
        return "BEARISH_SWEEP"

    # Bullish liquidity sweep:
    # price takes previous low and closes back above it.
    if (
        last["low"] < previous_low
        and last["close"] > previous_low
    ):
        return "BULLISH_SWEEP"

    return None


# =========================================================
# FVG
# =========================================================

def detect_fvg(df):
    if len(df) < 5:
        return None

    a = df.iloc[-3]
    b = df.iloc[-2]
    c = df.iloc[-1]

    # Bullish FVG
    if c["low"] > a["high"]:
        return {
            "type": "BULLISH_FVG",
            "low": float(a["high"]),
            "high": float(c["low"])
        }

    # Bearish FVG
    if c["high"] < a["low"]:
        return {
            "type": "BEARISH_FVG",
            "low": float(c["high"]),
            "high": float(a["low"])
        }

    return None


# =========================================================
# CANDLE CONFIRMATION
# =========================================================

def candle_confirmation(df):
    if len(df) < 3:
        return None

    last = df.iloc[-1]

    body = abs(
        last["close"] - last["open"]
    )

    candle_range = (
        last["high"] -
        last["low"]
    )

    if candle_range <= 0:
        return None

    upper_wick = (
        last["high"] -
        max(
            last["open"],
            last["close"]
        )
    )

    lower_wick = (
        min(
            last["open"],
            last["close"]
        ) -
        last["low"]
    )

    body_ratio = (
        body / candle_range
    )

    # Strong bullish candle
    if (
        last["close"] > last["open"]
        and body_ratio >= 0.55
    ):
        return "BULLISH_CONFIRMATION"

    # Strong bearish candle
    if (
        last["close"] < last["open"]
        and body_ratio >= 0.55
    ):
        return "BEARISH_CONFIRMATION"

    # Rejection from below
    if (
        lower_wick > body * 1.5
        and last["close"] > last["open"]
    ):
        return "BULLISH_REJECTION"

    # Rejection from above
    if (
        upper_wick > body * 1.5
        and last["close"] < last["open"]
    ):
        return "BEARISH_REJECTION"

    return None


# =========================================================
# MOMENTUM
# =========================================================

def momentum_state(df):
    last = df.iloc[-1]

    if (
        last["close"] > last["ema20"]
        and last["ema20"] > last["ema50"]
        and last["rsi"] >= 52
    ):
        return "BULLISH"

    if (
        last["close"] < last["ema20"]
        and last["ema20"] < last["ema50"]
        and last["rsi"] <= 48
    ):
        return "BEARISH"

    return "NEUTRAL"


# =========================================================
# DISTANCE TO ZONE
# =========================================================

def near_price(price, level, atr_value, multiplier=0.7):
    if not np.isfinite(atr_value):
        return False

    return abs(
        price - level
    ) <= atr_value * multiplier


# =========================================================
# SIGNAL ENGINE
# =========================================================

def analyze_market(m5, m15):
    m5 = add_indicators(m5)
    m15 = add_indicators(m15)

    # Ignore incomplete latest candle.
    # We use the latest completed 5m candle.
    if len(m5) < 60 or len(m15) < 60:
        return None

    m5c = m5.iloc[:-1].copy()
    m15c = m15.iloc[:-1].copy()

    if len(m5c) < 60 or len(m15c) < 60:
        return None

    last = m5c.iloc[-1]
    price = float(last["close"])

    atr_value = float(last["atr"])

    if not np.isfinite(atr_value):
        return None

    # -------------------------------
    # M15 CONTEXT
    # -------------------------------

    m15_trend = structure(m15c)
    m15_momentum = momentum_state(m15c)

    m15_support, m15_resistance = (
        support_resistance(m15c)
    )

    m15_demand, m15_supply = (
        supply_demand(m15c)
    )

    # -------------------------------
    # M5 ENTRY
    # -------------------------------

    m5_structure = structure(m5c)
    m5_momentum = momentum_state(m5c)

    m5_support, m5_resistance = (
        support_resistance(m5c)
    )

    m5_demand, m5_supply = (
        supply_demand(m5c)
    )

    sweep = liquidity_sweep(m5c)
    fvg = detect_fvg(m5c)
    confirmation = candle_confirmation(m5c)

    buy_score = 0
    sell_score = 0

    buy_reasons = []
    sell_reasons = []

    # =====================================================
    # BUY
    # =====================================================

    # M15 trend
    if m15_trend in [
        "BULLISH",
        "BULLISH_BOS"
    ]:
        buy_score += 18
        buy_reasons.append(
            "M15 bullish structure"
        )

    # M5 trend
    if m5_structure in [
        "BULLISH",
        "BULLISH_BOS"
    ]:
        buy_score += 12
        buy_reasons.append(
            "M5 bullish structure"
        )

    # Momentum
    if m15_momentum == "BULLISH":
        buy_score += 10
        buy_reasons.append(
            "M15 bullish momentum"
        )

    if m5_momentum == "BULLISH":
        buy_score += 10
        buy_reasons.append(
            "M5 bullish momentum"
        )

    # Liquidity sweep
    if sweep == "BULLISH_SWEEP":
        buy_score += 18
        buy_reasons.append(
            "Bullish liquidity sweep"
        )

    # Candle confirmation
    if confirmation in [
        "BULLISH_CONFIRMATION",
        "BULLISH_REJECTION"
    ]:
        buy_score += 10
        buy_reasons.append(
            "Bullish candle confirmation"
        )

    # FVG
    if fvg and fvg["type"] == "BULLISH_FVG":
        buy_score += 8
        buy_reasons.append(
            "Bullish FVG"
        )

    # Support / demand proximity
    if near_price(
        price,
        m5_support,
        atr_value
    ):
        buy_score += 7
        buy_reasons.append(
            "Near M5 support"
        )

    if near_price(
        price,
        m5_demand,
        atr_value
    ):
        buy_score += 7
        buy_reasons.append(
            "Near M5 demand"
        )

    # =====================================================
    # SELL
    # =====================================================

    if m15_trend in [
        "BEARISH",
        "BEARISH_BOS"
    ]:
        sell_score += 18
        sell_reasons.append(
            "M15 bearish structure"
        )

    if m5_structure in [
        "BEARISH",
        "BEARISH_BOS"
    ]:
        sell_score += 12
        sell_reasons.append(
            "M5 bearish structure"
        )

    if m15_momentum == "BEARISH":
        sell_score += 10
        sell_reasons.append(
            "M15 bearish momentum"
        )

    if m5_momentum == "BEARISH":
        sell_score += 10
        sell_reasons.append(
            "M5 bearish momentum"
        )

    if sweep == "BEARISH_SWEEP":
        sell_score += 18
        sell_reasons.append(
            "Bearish liquidity sweep"
        )

    if confirmation in [
        "BEARISH_CONFIRMATION",
        "BEARISH_REJECTION"
    ]:
        sell_score += 10
        sell_reasons.append(
            "Bearish candle confirmation"
        )

    if fvg and fvg["type"] == "BEARISH_FVG":
        sell_score += 8
        sell_reasons.append(
            "Bearish FVG"
        )

    if near_price(
        price,
        m5_resistance,
        atr_value
    ):
        sell_score += 7
        sell_reasons.append(
            "Near M5 resistance"
        )

    if near_price(
        price,
        m5_supply,
        atr_value
    ):
        sell_score += 7
        sell_reasons.append(
            "Near M5 supply"
        )

    # =====================================================
    # CHOOSE SIGNAL
    # =====================================================

    if (
        buy_score >= MIN_SCORE
        and buy_score > sell_score
    ):
        direction = "BUY"
        score = min(
            buy_score,
            100
        )
        reasons = buy_reasons

    elif (
        sell_score >= MIN_SCORE
        and sell_score > buy_score
    ):
        direction = "SELL"
        score = min(
            sell_score,
            100
        )
        reasons = sell_reasons

    else:
        return None

    # =====================================================
    # SL / TP
    # =====================================================

    # ATR based stop.
    # This avoids arbitrary fixed SL distances.
    stop_distance = max(
        atr_value * 1.15,
        PIP_SIZE * 8
    )

    if direction == "BUY":

        entry = price

        sl = entry - stop_distance

        risk = entry - sl

        tp1 = entry + (
            risk * TP1_RR
        )

        tp2 = entry + (
            risk * TP2_RR
        )

    else:

        entry = price

        sl = entry + stop_distance

        risk = sl - entry

        tp1 = entry - (
            risk * TP1_RR
        )

        tp2 = entry - (
            risk * TP2_RR
        )

    # =====================================================
    # PIPS
    # =====================================================

    sl_pips = abs(
        entry - sl
    ) / PIP_SIZE

    tp1_pips = abs(
        tp1 - entry
    ) / PIP_SIZE

    tp2_pips = abs(
        tp2 - entry
    ) / PIP_SIZE

    signal_time = m5c.index[-1]

    return {
        "direction": direction,
        "score": int(score),
        "entry": entry,
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,
        "sl_pips": sl_pips,
        "tp1_pips": tp1_pips,
        "tp2_pips": tp2_pips,
        "time": signal_time,
        "reasons": reasons,
        "m15_trend": m15_trend,
        "m5_structure": m5_structure,
        "m15_momentum": m15_momentum,
        "m5_momentum": m5_momentum,
        "sweep": sweep,
        "fvg": fvg
    }


# =========================================================
# FORMAT SIGNAL
# =========================================================

def format_signal(signal):

    direction = signal["direction"]

    emoji = "🟢" if direction == "BUY" else "🔴"

    reasons = "\n".join(
        f"• {reason}"
        for reason in signal["reasons"]
    )

    return (
        f"{emoji} GOLD AI SIGNAL\n"
        f"━━━━━━━━━━━━━━\n"
        f"XAU/USD\n"
        f"Type: SCALPING\n"
        f"Signal: {direction}\n"
        f"Score: {signal['score']}/100\n"
        f"━━━━━━━━━━━━━━\n"
        f"Entry: {signal['entry']:.2f}\n"
        f"SL: {signal['sl']:.2f} "
        f"(-{signal['sl_pips']:.1f} pips)\n"
        f"TP1: {signal['tp1']:.2f} "
        f"(+{signal['tp1_pips']:.1f} pips)\n"
        f"TP2: {signal['tp2']:.2f} "
        f"(+{signal['tp2_pips']:.1f} pips)\n"
        f"━━━━━━━━━━━━━━\n"
        f"M15: {signal['m15_trend']}\n"
        f"M5: {signal['m5_structure']}\n"
        f"Sweep: {signal['sweep'] or 'None'}\n"
        f"━━━━━━━━━━━━━━\n"
        f"Reasons:\n"
        f"{reasons}\n"
        f"━━━━━━━━━━━━━━\n"
        f"Status: WAITING FOR ENTRY\n"
        f"Data: Twelve Data\n"
        f"⚠️ Signal is algorithmic, not a guarantee."
    )


# =========================================================
# TRADE MONITOR
# =========================================================

def monitor_trade(price):

    global active_trade

    if not active_trade:
        return

    direction = active_trade["direction"]

    entry = active_trade["entry"]
    sl = active_trade["sl"]
    tp1 = active_trade["tp1"]
    tp2 = active_trade["tp2"]

    # BUY
    if direction == "BUY":

        # TP2
        if price >= tp2:

            send_trade_result(
                active_trade,
                "TP2 HIT",
                price
            )

            active_trade = None
            return

        # TP1
        if (
            not active_trade["tp1_hit"]
            and price >= tp1
        ):

            active_trade["tp1_hit"] = True

            telegram_send(
                "🟢 GOLD UPDATE\n"
                "━━━━━━━━━━━━━━\n"
                "BUY\n"
                "✅ TP1 HIT\n"
                f"Price: {price:.2f}\n"
                f"Profit: +"
                f"{abs(price-entry)/PIP_SIZE:.1f} pips\n"
                "SL moved to BREAKEVEN."
            )

            # Move SL to entry
            active_trade["sl"] = entry

        # SL
        if price <= active_trade["sl"]:

            send_trade_result(
                active_trade,
                "SL HIT",
                price
            )

            active_trade = None
            return

    # SELL
    else:

        # TP2
        if price <= tp2:

            send_trade_result(
                active_trade,
                "TP2 HIT",
                price
            )

            active_trade = None
            return

        # TP1
        if (
            not active_trade["tp1_hit"]
            and price <= tp1
        ):

            active_trade["tp1_hit"] = True

            telegram_send(
                "🔴 GOLD UPDATE\n"
                "━━━━━━━━━━━━━━\n"
                "SELL\n"
                "✅ TP1 HIT\n"
                f"Price: {price:.2f}\n"
                f"Profit: +"
                f"{abs(entry-price)/PIP_SIZE:.1f} pips\n"
                "SL moved to BREAKEVEN."
            )

            active_trade["sl"] = entry

        # SL
        if price >= active_trade["sl"]:

            send_trade_result(
                active_trade,
                "SL HIT",
                price
            )

            active_trade = None
            return


def send_trade_result(
    trade,
    status,
    price
):

    entry = trade["entry"]

    if trade["direction"] == "BUY":
        pips = (
            price - entry
        ) / PIP_SIZE
    else:
        pips = (
            entry - price
        ) / PIP_SIZE

    emoji = (
        "🟢"
        if pips >= 0
        else "🔴"
    )

    telegram_send(
        f"{emoji} GOLD TRADE UPDATE\n"
        f"━━━━━━━━━━━━━━\n"
        f"{trade['direction']}\n"
        f"{status}\n"
        f"Entry: {entry:.2f}\n"
        f"Exit: {price:.2f}\n"
        f"Result: {pips:+.1f} pips\n"
        f"━━━━━━━━━━━━━━\n"
        f"XAU/USD"
    )


# =========================================================
# STARTUP TEST
# =========================================================

def startup_message():

    telegram_send(
        "🤖 GOLD AI SIGNALS\n"
        "━━━━━━━━━━━━━━\n"
        "Bot connected successfully.\n"
        "Market monitor: READY\n"
        "XAU/USD: READY\n"
        "M5 analysis: READY\n"
        "M15 analysis: READY\n"
        "Liquidity: READY\n"
        "Structure: READY\n"
        "FVG: READY\n"
        "Risk engine: READY\n"
        "━━━━━━━━━━━━━━\n"
        "Waiting for a strong setup..."
    )


# =========================================================
# MAIN LOOP
# =========================================================

def main():

    global last_signal_key
    global active_trade

    validate_config()

    logger.info(
        "GOLD AI BOT STARTING..."
    )

    startup_message()

    while True:

        try:

            logger.info(
                "Fetching XAU/USD M5 data..."
            )

            # One API call every 5 minutes.
            # M15 is created locally from M5 data.
            m5 = get_candles(
                interval="5min",
                outputsize=250
            )

            m15 = build_m15(m5)

            # Current market price
            current_price = float(
                m5["close"].iloc[0]
            )

            # Monitor existing signal first
            monitor_trade(
                current_price
            )

            # Only search for a new trade
            # if no active trade exists.
            if active_trade is None:

                signal = analyze_market(
                    m5,
                    m15
                )

                if signal:

                    # Unique ID based on
                    # direction + candle time.
                    signal_key = (
                        signal["direction"],
                        str(signal["time"])
                    )

                    if signal_key != last_signal_key:

                        message = format_signal(
                            signal
                        )

                        sent = telegram_send(
                            message
                        )

                        if sent:

                            active_trade = {
                                **signal,
                                "tp1_hit": False
                            }

                            last_signal_key = (
                                signal_key
                            )

                            logger.info(
                                "NEW %s SIGNAL | "
                                "Score=%s",
                                signal["direction"],
                                signal["score"]
                            )

            logger.info(
                "Monitor cycle complete. "
                "Sleeping %s seconds.",
                POLL_SECONDS
            )

            time.sleep(
                POLL_SECONDS
            )

        except KeyboardInterrupt:

            logger.info(
                "Bot stopped."
            )

            break

        except Exception as e:

            logger.exception(
                "Main loop error: %s",
                e
            )

            # Do not crash permanently.
            time.sleep(60)


if __name__ == "__main__":
    main()
