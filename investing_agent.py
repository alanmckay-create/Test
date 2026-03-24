#!/usr/bin/env python3
"""
Investing Agent — uses Claude Opus 4.6 with adaptive thinking to simulate
real-world equity trading and maximise return on investment.

Usage:
    python investing_agent.py [--days 30] [--capital 100000] [--seed 42]
"""

import argparse
import json
import os
import sys
from typing import Any

import anthropic

from market_simulator import MarketSimulator
from portfolio import Portfolio


# ─────────────────────────────────────────────────────────────────────────────
#  Tool implementations
# ─────────────────────────────────────────────────────────────────────────────

def make_tools(market: MarketSimulator, portfolio: Portfolio) -> list[dict]:
    return [
        {
            "name": "get_market_summary",
            "description": (
                "Returns the current date, trading day number, average market "
                "price, 1-day change, and overall market sentiment."
            ),
            "input_schema": {"type": "object", "properties": {}, "required": []},
        },
        {
            "name": "get_available_stocks",
            "description": (
                "Returns all tradeable symbols with their current prices and "
                "sectors. Use this to scan for opportunities."
            ),
            "input_schema": {"type": "object", "properties": {}, "required": []},
        },
        {
            "name": "get_stock_info",
            "description": (
                "Returns detailed information for a stock: price, 10-day "
                "price history, sector, and annualised volatility."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Ticker symbol, e.g. AAPL",
                    }
                },
                "required": ["symbol"],
            },
        },
        {
            "name": "get_portfolio_snapshot",
            "description": (
                "Returns your current portfolio: cash, equity value, total "
                "return %, open positions with unrealised P&L."
            ),
            "input_schema": {"type": "object", "properties": {}, "required": []},
        },
        {
            "name": "get_recent_trades",
            "description": "Returns the last N trades you have executed.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "n": {
                        "type": "integer",
                        "description": "Number of trades to return (default 10)",
                    }
                },
                "required": [],
            },
        },
        {
            "name": "buy_stock",
            "description": (
                "Buy shares of a stock. Specify either `shares` OR "
                "`dollar_amount` (not both). Commission is 0.1 %."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Ticker symbol"},
                    "shares": {
                        "type": "number",
                        "description": "Number of shares to buy",
                    },
                    "dollar_amount": {
                        "type": "number",
                        "description": "Dollar amount to invest (shares derived from current price)",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Brief investment thesis for this trade",
                    },
                },
                "required": ["symbol", "reason"],
            },
        },
        {
            "name": "sell_stock",
            "description": (
                "Sell shares of a stock you hold. Set `sell_all` to true to "
                "close the entire position, or provide a `shares` count."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Ticker symbol"},
                    "shares": {
                        "type": "number",
                        "description": "Number of shares to sell",
                    },
                    "sell_all": {
                        "type": "boolean",
                        "description": "Set true to close entire position",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Reason for selling",
                    },
                },
                "required": ["symbol", "reason"],
            },
        },
        {
            "name": "advance_trading_day",
            "description": (
                "Advance the simulation by one trading day. Prices move "
                "according to the market model and news events may occur. "
                "Returns any news headlines generated."
            ),
            "input_schema": {"type": "object", "properties": {}, "required": []},
        },
    ]


def dispatch_tool(
    name: str,
    inp: dict[str, Any],
    market: MarketSimulator,
    portfolio: Portfolio,
    date: str,
) -> Any:
    if name == "get_market_summary":
        return market.get_market_summary()

    if name == "get_available_stocks":
        prices = market.get_all_prices()
        rows = []
        for sym, price in sorted(prices.items()):
            s = market.stocks[sym]
            in_portfolio = sym in portfolio.positions
            rows.append({
                "symbol": sym,
                "name": s.name,
                "sector": s.sector,
                "price": price,
                "in_portfolio": in_portfolio,
            })
        return rows

    if name == "get_stock_info":
        symbol = inp["symbol"].upper()
        return market.get_stock_info(symbol)

    if name == "get_portfolio_snapshot":
        prices = market.get_all_prices()
        return portfolio.snapshot(prices)

    if name == "get_recent_trades":
        n = inp.get("n", 10)
        return portfolio.recent_trades(n)

    if name == "buy_stock":
        symbol = inp["symbol"].upper()
        price = market.get_price(symbol)
        if "dollar_amount" in inp and inp["dollar_amount"]:
            shares = inp["dollar_amount"] / price
        elif "shares" in inp and inp["shares"]:
            shares = float(inp["shares"])
        else:
            return {"ok": False, "error": "Provide either shares or dollar_amount"}
        return portfolio.buy(symbol, shares, price, date, inp.get("reason", ""))

    if name == "sell_stock":
        symbol = inp["symbol"].upper()
        price = market.get_price(symbol)
        if inp.get("sell_all"):
            return portfolio.sell_all(symbol, price, date, inp.get("reason", ""))
        elif "shares" in inp and inp["shares"]:
            return portfolio.sell(symbol, float(inp["shares"]), price, date,
                                  inp.get("reason", ""))
        else:
            return {"ok": False, "error": "Provide either shares or sell_all=true"}

    if name == "advance_trading_day":
        events = market.advance_day()
        news = [
            {"headline": e.headline, "symbol": e.symbol, "sentiment": round(e.sentiment, 2)}
            for e in events
        ]
        return {
            "new_date": market.date.strftime("%Y-%m-%d"),
            "trading_day": market.day,
            "news": news,
        }

    return {"error": f"Unknown tool: {name}"}


# ─────────────────────────────────────────────────────────────────────────────
#  Agent loop
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert quantitative portfolio manager running a
simulated equity account. Your single goal is to maximise the total return on
the portfolio over the trading period.

## Environment
- You start with $100,000 in cash.
- You have access to ~20 US equities and two ETFs across Technology,
  Financials, Healthcare, Energy, and Consumer sectors.
- Commission is 0.1 % per trade (minimum $1).
- Prices move via geometric Brownian motion with realistic volatility.
  News events occur randomly and can cause sudden price moves.

## Strategy guidance
1. **Scan the market** each session with `get_available_stocks` and
   `get_stock_info` for individual stocks you're watching.
2. **Diversify sensibly** — don't put more than 25 % of your equity in
   one position. Sector diversification reduces correlation risk.
3. **Use momentum and mean-reversion** — stocks with strong recent
   performance may continue; extreme moves often revert.
4. **React to news** — positive catalysts are buying opportunities;
   negative news often warrants a stop-loss exit.
5. **Manage risk** — keep some cash as a buffer and consider selling
   losing positions to reallocate capital.
6. **Advance the day** after each decision round so prices evolve.
7. **Track your thesis** — always supply a `reason` when trading so
   you can review performance against your investment rationale.

## Per-session workflow
1. Check the date and market summary.
2. Review your portfolio.
3. Read any news.
4. Scan stocks and pick 2-5 to investigate further.
5. Execute trades.
6. Advance the day.
7. Repeat until the session is complete.

Make bold, informed decisions. Maximise ROI!
"""


def run_agent(trading_days: int, initial_capital: float, seed: int,
              verbose: bool = True) -> dict:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit("ERROR: ANTHROPIC_API_KEY environment variable is not set.")

    client = anthropic.Anthropic(api_key=api_key)
    market = MarketSimulator(seed=seed)
    portfolio = Portfolio(initial_cash=initial_capital)
    tools = make_tools(market, portfolio)

    user_prompt = (
        f"You are managing a ${initial_capital:,.0f} portfolio over {trading_days} "
        f"trading days (seed={seed}). Start immediately — analyse the market, "
        f"build a diversified portfolio, actively trade to maximise ROI, and "
        f"advance through all {trading_days} trading days. When you have "
        f"completed all {trading_days} trading days, provide a comprehensive "
        f"performance summary including: total return %, best/worst trades, "
        f"sector allocation, and key lessons learned."
    )

    messages: list[dict] = [{"role": "user", "content": user_prompt}]

    if verbose:
        print(f"\n{'='*70}")
        print(f"  INVESTING AGENT  |  Capital: ${initial_capital:,.0f}  "
              f"|  Days: {trading_days}  |  Seed: {seed}")
        print(f"{'='*70}\n")

    iteration = 0
    max_iterations = trading_days * 15   # safety ceiling

    while iteration < max_iterations:
        iteration += 1
        response = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=8192,
            thinking={"type": "adaptive"},
            system=SYSTEM_PROMPT,
            tools=tools,
            messages=messages,
        )

        # Collect the full assistant response to append to messages
        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in response.content:
            if block.type == "thinking" and verbose:
                # Show a brief excerpt of the thinking
                excerpt = block.thinking[:300].replace("\n", " ")
                print(f"[THINKING] {excerpt}{'...' if len(block.thinking) > 300 else ''}\n")
            elif block.type == "text" and block.text.strip() and verbose:
                print(f"[AGENT] {block.text}\n")
            elif block.type == "tool_use":
                if verbose:
                    print(f"[TOOL] {block.name}({json.dumps(block.input, separators=(',', ':'))})")
                result = dispatch_tool(
                    block.name, block.input, market, portfolio,
                    market.date.strftime("%Y-%m-%d"),
                )
                if verbose:
                    result_str = json.dumps(result)
                    excerpt = result_str[:200]
                    print(f"       → {excerpt}{'...' if len(result_str) > 200 else ''}\n")
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result),
                })

        if tool_results:
            messages.append({"role": "user", "content": tool_results})

        # Stop when Claude has issued its final text with no tool calls
        if response.stop_reason == "end_turn" and not tool_results:
            break

    # ── Final performance report ──────────────────────────────────────── #
    prices = market.get_all_prices()
    final = portfolio.snapshot(prices)

    if verbose:
        print(f"\n{'='*70}")
        print("  FINAL PERFORMANCE REPORT")
        print(f"{'='*70}")
        print(f"  Initial capital  : ${final['initial_capital']:>12,.2f}")
        print(f"  Final equity     : ${final['equity']:>12,.2f}")
        print(f"  Total return     : {final['total_return_pct']:>+.2f} %")
        print(f"  Cash             : ${final['cash']:>12,.2f}")
        print(f"  Total trades     : {final['total_trades']}")
        print(f"  Trading days     : {market.day}")
        if final["positions"]:
            print(f"\n  Open positions:")
            for p in final["positions"]:
                print(f"    {p['symbol']:<6} {p['shares']:>8.2f} shares  "
                      f"cost ${p['avg_cost']:>8.2f}  "
                      f"now ${p['current_price']:>8.2f}  "
                      f"P&L {p['unrealised_pct']:>+.1f}%  "
                      f"({p['weight_pct']:.1f}% of portfolio)")
        print(f"{'='*70}\n")

    return final


# ─────────────────────────────────────────────────────────────────────────────
#  CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Investing agent that simulates equity trading with Claude."
    )
    parser.add_argument("--days", type=int, default=20,
                        help="Number of trading days to simulate (default: 20)")
    parser.add_argument("--capital", type=float, default=100_000.0,
                        help="Starting capital in USD (default: 100000)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress verbose output; only show final summary")
    args = parser.parse_args()

    result = run_agent(
        trading_days=args.days,
        initial_capital=args.capital,
        seed=args.seed,
        verbose=not args.quiet,
    )

    if args.quiet:
        print(f"Final equity: ${result['equity']:,.2f}  "
              f"Return: {result['total_return_pct']:+.2f}%  "
              f"Trades: {result['total_trades']}")


if __name__ == "__main__":
    main()
