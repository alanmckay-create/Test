"""
Portfolio manager — tracks cash, positions, trade history, and performance.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class Position:
    symbol: str
    shares: float
    avg_cost: float           # weighted average cost per share
    open_date: str


@dataclass
class Trade:
    date: str
    action: str               # BUY | SELL
    symbol: str
    shares: float
    price: float
    total: float              # positive = spent, negative = received
    reason: str


class Portfolio:
    def __init__(self, initial_cash: float = 100_000.0):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions: dict[str, Position] = {}
        self.trade_history: list[Trade] = []
        self._commission_rate = 0.001    # 0.1 % per trade

    # ------------------------------------------------------------------ #
    #  Trading                                                              #
    # ------------------------------------------------------------------ #

    def buy(self, symbol: str, shares: float, price: float,
            date: str, reason: str = "") -> dict:
        if shares <= 0:
            return {"ok": False, "error": "shares must be positive"}
        cost = shares * price
        commission = max(1.0, cost * self._commission_rate)
        total = cost + commission
        if total > self.cash:
            return {
                "ok": False,
                "error": f"Insufficient cash. Need ${total:.2f}, have ${self.cash:.2f}",
            }
        self.cash -= total
        if symbol in self.positions:
            pos = self.positions[symbol]
            new_shares = pos.shares + shares
            pos.avg_cost = (pos.shares * pos.avg_cost + cost) / new_shares
            pos.shares = new_shares
        else:
            self.positions[symbol] = Position(symbol, shares, price, date)
        self.trade_history.append(
            Trade(date, "BUY", symbol, shares, price, total, reason)
        )
        return {
            "ok": True,
            "symbol": symbol,
            "shares": shares,
            "price": price,
            "commission": round(commission, 2),
            "total_spent": round(total, 2),
            "cash_remaining": round(self.cash, 2),
        }

    def sell(self, symbol: str, shares: float, price: float,
             date: str, reason: str = "") -> dict:
        if symbol not in self.positions:
            return {"ok": False, "error": f"No position in {symbol}"}
        pos = self.positions[symbol]
        if shares > pos.shares:
            return {
                "ok": False,
                "error": f"Only {pos.shares:.2f} shares available, tried to sell {shares:.2f}",
            }
        proceeds = shares * price
        commission = max(1.0, proceeds * self._commission_rate)
        net = proceeds - commission
        profit = (price - pos.avg_cost) * shares - commission
        self.cash += net
        pos.shares -= shares
        if pos.shares < 1e-6:
            del self.positions[symbol]
        self.trade_history.append(
            Trade(date, "SELL", symbol, shares, price, -net, reason)
        )
        return {
            "ok": True,
            "symbol": symbol,
            "shares": shares,
            "price": price,
            "commission": round(commission, 2),
            "net_proceeds": round(net, 2),
            "trade_profit": round(profit, 2),
            "cash_now": round(self.cash, 2),
        }

    def sell_all(self, symbol: str, price: float, date: str,
                 reason: str = "") -> dict:
        if symbol not in self.positions:
            return {"ok": False, "error": f"No position in {symbol}"}
        return self.sell(symbol, self.positions[symbol].shares, price, date, reason)

    # ------------------------------------------------------------------ #
    #  Reporting                                                            #
    # ------------------------------------------------------------------ #

    def snapshot(self, current_prices: dict[str, float]) -> dict:
        equity = self.cash
        positions_out = []
        for sym, pos in self.positions.items():
            price = current_prices.get(sym, pos.avg_cost)
            mkt_val = pos.shares * price
            unrealised = (price - pos.avg_cost) * pos.shares
            unrealised_pct = (price - pos.avg_cost) / pos.avg_cost * 100
            equity += mkt_val
            positions_out.append({
                "symbol": sym,
                "shares": round(pos.shares, 4),
                "avg_cost": round(pos.avg_cost, 2),
                "current_price": round(price, 2),
                "market_value": round(mkt_val, 2),
                "unrealised_pnl": round(unrealised, 2),
                "unrealised_pct": round(unrealised_pct, 2),
                "weight_pct": 0,       # filled below
            })
        for p in positions_out:
            p["weight_pct"] = round(p["market_value"] / equity * 100, 2) if equity > 0 else 0

        total_return = (equity - self.initial_cash) / self.initial_cash * 100
        realised_pnl = sum(
            t.total for t in self.trade_history if t.action == "SELL"
        )
        realised_pnl = -realised_pnl  # stored as negative for cash received
        return {
            "cash": round(self.cash, 2),
            "equity": round(equity, 2),
            "initial_capital": self.initial_cash,
            "total_return_pct": round(total_return, 2),
            "total_trades": len(self.trade_history),
            "positions": sorted(positions_out, key=lambda x: -x["market_value"]),
        }

    def recent_trades(self, n: int = 10) -> list[dict]:
        trades = self.trade_history[-n:]
        return [
            {
                "date": t.date,
                "action": t.action,
                "symbol": t.symbol,
                "shares": round(t.shares, 4),
                "price": round(t.price, 2),
                "amount": round(abs(t.total), 2),
                "reason": t.reason,
            }
            for t in reversed(trades)
        ]
