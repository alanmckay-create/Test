"""
Market simulator — generates realistic stock price movements using
geometric Brownian motion with sector correlations and news events.
"""

import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional


@dataclass
class StockInfo:
    symbol: str
    name: str
    sector: str
    current_price: float
    volatility: float          # annualised σ
    drift: float               # annualised μ (expected return)
    price_history: list[float] = field(default_factory=list)


@dataclass
class NewsEvent:
    headline: str
    symbol: Optional[str]      # None  → market-wide
    sentiment: float           # -1.0 … +1.0
    impact: float              # price-move magnitude


class MarketSimulator:
    """Simulates a small equity market with ~20 tickers."""

    STOCKS: dict[str, dict] = {
        "AAPL": {"name": "Apple Inc.",             "sector": "Technology",    "price": 185.0, "vol": 0.25, "drift": 0.12},
        "MSFT": {"name": "Microsoft Corp.",         "sector": "Technology",    "price": 415.0, "vol": 0.22, "drift": 0.14},
        "GOOGL":{"name": "Alphabet Inc.",           "sector": "Technology",    "price": 175.0, "vol": 0.28, "drift": 0.11},
        "AMZN": {"name": "Amazon.com Inc.",         "sector": "Technology",    "price": 195.0, "vol": 0.30, "drift": 0.13},
        "NVDA": {"name": "NVIDIA Corp.",            "sector": "Technology",    "price": 875.0, "vol": 0.45, "drift": 0.25},
        "META": {"name": "Meta Platforms Inc.",     "sector": "Technology",    "price": 550.0, "vol": 0.35, "drift": 0.18},
        "JPM":  {"name": "JPMorgan Chase & Co.",   "sector": "Financials",    "price": 205.0, "vol": 0.20, "drift": 0.10},
        "BAC":  {"name": "Bank of America Corp.",  "sector": "Financials",    "price": 40.0,  "vol": 0.22, "drift": 0.09},
        "GS":   {"name": "Goldman Sachs Group",    "sector": "Financials",    "price": 490.0, "vol": 0.24, "drift": 0.11},
        "JNJ":  {"name": "Johnson & Johnson",      "sector": "Healthcare",    "price": 155.0, "vol": 0.15, "drift": 0.08},
        "UNH":  {"name": "UnitedHealth Group",     "sector": "Healthcare",    "price": 530.0, "vol": 0.18, "drift": 0.12},
        "PFE":  {"name": "Pfizer Inc.",            "sector": "Healthcare",    "price": 28.0,  "vol": 0.20, "drift": 0.06},
        "XOM":  {"name": "Exxon Mobil Corp.",      "sector": "Energy",        "price": 115.0, "vol": 0.22, "drift": 0.09},
        "CVX":  {"name": "Chevron Corp.",          "sector": "Energy",        "price": 155.0, "vol": 0.21, "drift": 0.08},
        "WMT":  {"name": "Walmart Inc.",           "sector": "Consumer Staples","price": 68.0, "vol": 0.14, "drift": 0.08},
        "PG":   {"name": "Procter & Gamble Co.",   "sector": "Consumer Staples","price": 165.0, "vol": 0.13, "drift": 0.07},
        "TSLA": {"name": "Tesla Inc.",             "sector": "Consumer Discretionary","price": 175.0, "vol": 0.55, "drift": 0.15},
        "HD":   {"name": "Home Depot Inc.",        "sector": "Consumer Discretionary","price": 395.0, "vol": 0.20, "drift": 0.10},
        "SPY":  {"name": "S&P 500 ETF",            "sector": "ETF",           "price": 520.0, "vol": 0.16, "drift": 0.10},
        "QQQ":  {"name": "Nasdaq-100 ETF",         "sector": "ETF",           "price": 445.0, "vol": 0.22, "drift": 0.12},
    }

    NEWS_TEMPLATES: list[dict] = [
        {"tpl": "{} beats earnings by 15%, guidance raised",        "sentiment":  0.8, "impact": 0.04},
        {"tpl": "{} misses revenue estimates, shares under pressure","sentiment": -0.7, "impact": 0.05},
        {"tpl": "{} announces $10B share buyback programme",         "sentiment":  0.6, "impact": 0.03},
        {"tpl": "{} CEO unexpectedly resigns amid investigation",    "sentiment": -0.9, "impact": 0.08},
        {"tpl": "{} launches breakthrough product, analysts bullish","sentiment":  0.7, "impact": 0.05},
        {"tpl": "{} faces antitrust lawsuit from regulators",        "sentiment": -0.6, "impact": 0.04},
        {"tpl": "{} wins $2B government contract",                   "sentiment":  0.7, "impact": 0.04},
        {"tpl": "{} upgrades full-year outlook on strong demand",    "sentiment":  0.5, "impact": 0.02},
        {"tpl": "{} cuts dividend, cites cash-flow concerns",        "sentiment": -0.8, "impact": 0.06},
        {"tpl": "{} acquires rival in all-stock deal",               "sentiment":  0.4, "impact": 0.03},
        # market-wide events (symbol = None)
        {"tpl": "Fed signals interest-rate pause — rally in equities","sentiment": 0.6, "impact": 0.02, "market": True},
        {"tpl": "Inflation data hotter than expected — risk-off mood","sentiment":-0.5, "impact": 0.02, "market": True},
        {"tpl": "Strong jobs report lifts market sentiment",          "sentiment": 0.4, "impact": 0.015,"market": True},
        {"tpl": "Trade-war fears resurface on new tariff threats",    "sentiment":-0.6, "impact": 0.025,"market": True},
        {"tpl": "Global growth upgrade from IMF boosts equities",     "sentiment": 0.5, "impact": 0.018,"market": True},
    ]

    def __init__(self, seed: int = 42):
        random.seed(seed)
        self.day = 0
        self.date = datetime(2025, 1, 2)
        self.stocks: dict[str, StockInfo] = {}
        self._pending_events: list[NewsEvent] = []

        for sym, cfg in self.STOCKS.items():
            s = StockInfo(
                symbol=sym,
                name=cfg["name"],
                sector=cfg["sector"],
                current_price=cfg["price"],
                volatility=cfg["vol"],
                drift=cfg["drift"],
            )
            s.price_history.append(s.current_price)
            self.stocks[sym] = s

    # ------------------------------------------------------------------ #
    #  Public helpers                                                       #
    # ------------------------------------------------------------------ #

    def get_price(self, symbol: str) -> float:
        sym = symbol.upper()
        if sym not in self.stocks:
            raise ValueError(f"Unknown symbol: {symbol}")
        return round(self.stocks[sym].current_price, 2)

    def get_all_prices(self) -> dict[str, float]:
        return {sym: round(s.current_price, 2) for sym, s in self.stocks.items()}

    def get_stock_info(self, symbol: str) -> dict:
        sym = symbol.upper()
        s = self.stocks[sym]
        hist = s.price_history[-10:]          # last 10 days
        chg = (hist[-1] - hist[0]) / hist[0] * 100 if len(hist) > 1 else 0.0
        return {
            "symbol": sym,
            "name": s.name,
            "sector": s.sector,
            "price": round(s.current_price, 2),
            "10d_change_pct": round(chg, 2),
            "volatility_annual": round(s.volatility, 3),
            "price_history_10d": [round(p, 2) for p in hist],
        }

    def advance_day(self) -> list[NewsEvent]:
        """Simulate one trading day. Returns any news events generated."""
        self.day += 1
        self.date += timedelta(days=1)
        # skip weekends
        while self.date.weekday() >= 5:
            self.date += timedelta(days=1)

        events: list[NewsEvent] = []

        # Possibly generate news
        if random.random() < 0.40:
            events.append(self._generate_news())

        # Apply market-wide drift from pending events
        market_drift = sum(
            e.impact * e.sentiment for e in events if e.symbol is None
        )

        dt = 1 / 252          # one trading day
        for s in self.stocks.values():
            # sector correlation: 40 % beta to market
            sector_shock = random.gauss(0, 1)
            idio_shock    = random.gauss(0, 1)
            corr = 0.4
            combined = math.sqrt(corr) * random.gauss(0, 1) + math.sqrt(1 - corr) * idio_shock
            # GBM step
            log_return = (s.drift - 0.5 * s.volatility ** 2) * dt \
                        + s.volatility * math.sqrt(dt) * combined \
                        + market_drift * dt
            # Company-specific news impact
            for ev in events:
                if ev.symbol == s.symbol:
                    log_return += ev.sentiment * ev.impact
            s.current_price *= math.exp(log_return)
            s.current_price = max(s.current_price, 0.01)  # floor
            s.price_history.append(round(s.current_price, 2))

        return events

    def get_market_summary(self) -> dict:
        """Return aggregate market stats (simple equal-weight index)."""
        prices = list(self.get_all_prices().values())
        avg_price = sum(prices) / len(prices)
        # 1-day change
        day_changes = []
        for s in self.stocks.values():
            if len(s.price_history) >= 2:
                d = (s.price_history[-1] - s.price_history[-2]) / s.price_history[-2] * 100
                day_changes.append(d)
        avg_day_change = sum(day_changes) / len(day_changes) if day_changes else 0.0
        return {
            "date": self.date.strftime("%Y-%m-%d"),
            "trading_day": self.day,
            "average_price": round(avg_price, 2),
            "avg_1d_change_pct": round(avg_day_change, 2),
            "market_sentiment": "bullish" if avg_day_change > 0.3 else
                                "bearish" if avg_day_change < -0.3 else "neutral",
        }

    # ------------------------------------------------------------------ #
    #  Private                                                              #
    # ------------------------------------------------------------------ #

    def _generate_news(self) -> NewsEvent:
        tpl = random.choice(self.NEWS_TEMPLATES)
        if tpl.get("market"):
            return NewsEvent(
                headline=tpl["tpl"],
                symbol=None,
                sentiment=tpl["sentiment"] * random.uniform(0.8, 1.2),
                impact=tpl["impact"],
            )
        sym = random.choice(list(self.stocks.keys()))
        s = self.stocks[sym]
        return NewsEvent(
            headline=tpl["tpl"].format(s.name),
            symbol=sym,
            sentiment=tpl["sentiment"] * random.uniform(0.8, 1.2),
            impact=tpl["impact"],
        )
