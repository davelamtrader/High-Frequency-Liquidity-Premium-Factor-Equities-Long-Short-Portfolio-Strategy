# High-Frequency-Liquidity-Premium-Factor-Equities-Long-Short-Portfolio-Strategy

## Overview

This repository presents a trading strategy developed to capitalize on the liquidity premium factor within the equities market. The strategy focuses on identifying and exploiting a "liquidity premium" by analyzing the impact of simulated trading on order book data.

## Strategy Components

### 1. Data Acquisition and Preprocessing

-   **Data Source:**  The strategy utilizes high-frequency market data sourced from the Polygon.io API, including Level 1 (L1) quotes (bid/ask) and trade data for a set of NASDAQ 100 stocks.
-   **API Client:**  A  PolygonDataClient  class is implemented to fetch intraday quotes, trade, and daily OHLCV data using the Polygon.io API.
-   **Data Loading:**  The code includes a function  get_nasdaq100_tickers()  to obtain a list of NASDAQ 100 constituent tickers, though it uses a simplified list for demonstration purposes. The code can be adapted to analyze other markets by modifying the ticker list.

### 2. Liquidity Premium Factor Calculation

-   **Liquidity Premium Factor:**  The core of the strategy is the "Liquidity Premium Factor," which quantifies the market impact cost of trading.
-   **simulate_order_matching():**  This function simulates order matching using bid data, providing an estimate of the market impact. It uses bid price and size information to estimate trading costs.
-   **calculate_daily_factor():**  This function calculates the daily factor value based on simulated trading  . It allocates a trading amount based on market capitalization and frequency and then simulates trading at different time intervals to estimate liquidity costs. The trading frequency is set by the  frequency_minutes  parameter.
-   **calculate_factor():**  This function calculates the final liquidity premium factor by taking the relative difference between the simulated trading costs  . The function uses a lookback period of  period_days.

### 3. Strategy Implementation

-   **LiquidityPremiumStrategy  Class:**  This class encapsulates the strategy logic, including loading data, calculating factors, and running backtests.
-   **Backtesting:**  The strategy framework is implemented to backtest the liquidity premium factor strategy.
-   **load_data():**  This method loads historical data, including daily price data and intraday quote and trade data, for a set of tickers.
-   **calculate_factors():**  This method calculates the rolling factors and implements a time-weighted and volatility-weighted factor for a more robust strategy.
-   **rebalance_portfolio():**  This method rebalances the portfolio based on the calculated factors, selecting stocks for inclusion.
-   **run_backtest():**  The  run_backtest()  function executes the backtest and computes various performance metrics.

### 4. Performance Evaluation

-   **calculate_liquidity_premium_factor():**  Calculates liquidity premium factor based on order book data.
-   **Performance Metrics:**  The backtesting process computes key performance metrics, including total return, annualized return, volatility, Sharpe ratio, and maximum drawdown. The  quantstats  library is used for detailed performance reporting.

## Code Structure

The repository is organized into modular classes and functions:

-   PolygonDataClient: Handles data fetching from the Polygon.io API.
-   LiquidityPremiumFactor: Calculates the liquidity premium factor.
-   LiquidityPremiumStrategy: Implements the overall strategy, including data loading, factor calculation, backtesting, and performance reporting.
