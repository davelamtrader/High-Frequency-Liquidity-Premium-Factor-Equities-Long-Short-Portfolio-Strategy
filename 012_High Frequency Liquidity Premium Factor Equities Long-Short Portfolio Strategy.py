#20190408-長江證劵-基礎因子研究（六）：高頻因子（一），流動性溢價因子
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
import warnings
import quantstats as qs
warnings.filterwarnings('ignore')

# Required libraries
import quantstats as qs
from scipy import stats
from typing import Dict, List, Tuple, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PolygonDataClient:
    """Client for fetching data from Polygon.io API"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.polygon.io"
        
    def get_nasdaq100_tickers(self) -> List[str]:
        """Get NASDAQ 100 constituent tickers"""
        # Note: In practice, you would fetch this from a reliable source
        # This is a simplified list for demonstration
        nasdaq100_tickers = [
            'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'GOOG', 'META', 'TSLA',
            'AVGO', 'COST', 'NFLX', 'TMUS', 'ASML', 'ADBE', 'PEP', 'CSCO',
            'AMD', 'LIN', 'INTU', 'TXN', 'QCOM', 'ISRG', 'CMCSA', 'AMGN',
            'HON', 'BKNG', 'VRTX', 'AMAT', 'ADP', 'SBUX', 'GILD', 'ADI',
            'LRCX', 'MDLZ', 'PYPL', 'REGN', 'MELI', 'KLAC', 'SNPS', 'CDNS',
            'MAR', 'MRVL', 'ORLY', 'CSX', 'FTNT', 'DASH', 'ADSK', 'ASML',
            'CHTR', 'PCAR', 'NXPI', 'MNST', 'ABNB', 'WDAY', 'FANG', 'AEP',
            'PAYX', 'FAST', 'ODFL', 'ROST', 'KDP', 'VRSK', 'EA', 'LULU',
            'CTSH', 'EXC', 'GEHC', 'DDOG', 'TEAM', 'CSGP', 'XEL', 'KHC',
            'IDXX', 'ZS', 'CCEP', 'TTWO', 'ON', 'ANSS', 'CDW', 'CRWD',
            'WBD', 'GFS', 'ILMN', 'BIIB', 'ARM', 'MDB', 'DLTR'
        ]
        return nasdaq100_tickers[:20]  # Use first 20 for demonstration
        
    def get_quotes_data(self, ticker: str, date: str) -> pd.DataFrame:
        """Fetch L1 quotes data using Polygon API"""
        url = f"{self.base_url}/v3/quotes/{ticker}"
        params = {
            'timestamp': date,
            'order': 'asc',
            'limit': 50000,
            'sort': 'timestamp',
            'apikey': self.api_key
        }
        
        try:
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                if 'results' in data and data['results']:
                    quotes_df = pd.DataFrame(data['results'])
                    quotes_df['timestamp'] = pd.to_datetime(quotes_df['sip_timestamp'], unit='ns')
                    return quotes_df[['timestamp', 'bid_price', 'bid_size', 'ask_price', 'ask_size']].dropna()
            time.sleep(0.1)  # Rate limiting
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error fetching quotes for {ticker}: {e}")
            return pd.DataFrame()
    
    def get_trades_data(self, ticker: str, date: str) -> pd.DataFrame:
        """Fetch tick trades data using Polygon API"""
        url = f"{self.base_url}/v3/trades/{ticker}"
        params = {
            'timestamp': date,
            'order': 'asc',
            'limit': 50000,
            'sort': 'timestamp',
            'apikey': self.api_key
        }
        
        try:
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                if 'results' in data and data['results']:
                    trades_df = pd.DataFrame(data['results'])
                    trades_df['timestamp'] = pd.to_datetime(trades_df['sip_timestamp'], unit='ns')
                    return trades_df[['timestamp', 'price', 'size']].dropna()
            time.sleep(0.1)  # Rate limiting
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error fetching trades for {ticker}: {e}")
            return pd.DataFrame()
    
    def get_daily_bars(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch daily OHLCV data using Polygon API"""
        url = f"{self.base_url}/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}"
        params = {
            'adjusted': 'true',
            'sort': 'asc',
            'limit': 50000,
            'apikey': self.api_key
        }
        
        try:
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                if 'results' in data and data['results']:
                    bars_df = pd.DataFrame(data['results'])
                    bars_df['date'] = pd.to_datetime(bars_df['t'], unit='ms')
                    bars_df = bars_df.rename(columns={
                        'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'
                    })
                    return bars_df[['date', 'open', 'high', 'low', 'close', 'volume']].set_index('date')
            time.sleep(0.1)  # Rate limiting
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error fetching daily bars for {ticker}: {e}")
            return pd.DataFrame()

class LiquidityPremiumFactor:
    """
    Implementation of the liquidity premium factor based on the Chinese research report.
    The factor constructs a liquidity premium using order book data and simulated trading.
    """
    
    def __init__(self, trading_amount: float = 1e8, frequency_minutes: int = 30, period_days: int = 21):
        """
        Initialize the liquidity premium factor calculator
        
        Args:
            trading_amount: Trading amount in dollars (A parameter from report)
            frequency_minutes: Trading frequency in minutes (n parameter from report)  
            period_days: Lookback period in days (T parameter from report)
        """
        self.trading_amount = trading_amount  # A parameter
        self.frequency_minutes = frequency_minutes  # n parameter
        self.period_days = period_days  # T parameter
        
    def simulate_order_matching(self, quotes_df: pd.DataFrame, allocation: float) -> Tuple[float, float]:
        """
        Simulate order matching using bid data with virtual orders through interpolation
        Based on algorithm described in the research report
        """
        if quotes_df.empty:
            return 0.0, 0.0
            
        # Get the latest quote data
        latest_quote = quotes_df.iloc[-1]
        
        # Extract bid information (bid1 to bid5 equivalent)
        bid_price = latest_quote.get('bid_price', 0)
        bid_size = latest_quote.get('bid_size', 0)
        ask_price = latest_quote.get('ask_price', 0)
        
        if bid_price <= 0 or bid_size <= 0:
            return 0.0, 0.0
            
        # Simulate multiple bid levels using interpolation
        # Create virtual bid levels as described in the algorithm
        bid_levels = []
        
        # Add the actual bid level
        bid_levels.append({'price': bid_price, 'size': bid_size})
        
        # Create virtual levels by reducing price by 0.01 each level
        virtual_size = max(bid_size / 4, 100)  # Average size for virtual levels
        for i in range(1, 6):  # Create 5 additional virtual levels
            virtual_price = bid_price - (i * 0.01)
            if virtual_price > 0:
                bid_levels.append({'price': virtual_price, 'size': virtual_size})
        
        # Calculate shares that can be purchased with allocation
        total_shares_demand = 0
        remaining_allocation = allocation
        
        for level in bid_levels:
            if remaining_allocation <= 0:
                break
                
            level_value = level['price'] * level['size']
            if remaining_allocation >= level_value:
                total_shares_demand += level['size']
                remaining_allocation -= level_value
            else:
                partial_shares = remaining_allocation / level['price']
                total_shares_demand += partial_shares
                remaining_allocation = 0
        
        # Calculate market value using demand-side pricing
        vol_need = total_shares_demand
        
        # Calculate actual trading volume using average price
        # Use mid-price as proxy for average trading price
        avg_price = (bid_price + ask_price) / 2 if ask_price > 0 else bid_price
        vol_actual = allocation / avg_price if avg_price > 0 else 0
        
        return vol_need, vol_actual
    
    def calculate_daily_factor(self, ticker: str, date: str, quotes_df: pd.DataFrame, 
                             trades_df: pd.DataFrame, market_cap: float) -> Tuple[float, float]:
        """
        Calculate daily factor values based on simulated trading
        """
        if quotes_df.empty:
            return 0.0, 0.0
            
        # Step 2: Allocate trading amount based on market cap
        allocation = market_cap * self.trading_amount / 1e12  # Normalize by total market
        
        # Step 3: Divide allocation by frequency (daily trading frequency)
        trades_per_day = int(480 / self.frequency_minutes)  # 480 minutes in trading day
        allocation_per_trade = allocation / trades_per_day
        
        daily_cap_need = 0
        daily_cap_actual = 0
        
        # Step 4-5: Simulate trading at different time intervals
        time_intervals = pd.date_range(
            start=date + ' 09:30:00',
            end=date + ' 16:00:00', 
            freq=f'{self.frequency_minutes}min'
        )
        
        for interval_time in time_intervals:
            # Get quotes near this time interval
            interval_quotes = quotes_df[
                (quotes_df['timestamp'] >= interval_time - pd.Timedelta(minutes=5)) &
                (quotes_df['timestamp'] <= interval_time + pd.Timedelta(minutes=5))
            ]
            
            if not interval_quotes.empty:
                # Get closing price for this interval (use last available trade price or quote)
                close_price = interval_quotes['bid_price'].iloc[-1]
                
                # Simulate order matching
                vol_need, vol_actual = self.simulate_order_matching(interval_quotes, allocation_per_trade)
                
                # Calculate market values using closing price
                cap_need = vol_need * close_price
                cap_actual = vol_actual * close_price
                
                daily_cap_need += cap_need
                daily_cap_actual += cap_actual
        
        return daily_cap_need, daily_cap_actual
    
    def calculate_factor(self, ticker: str, daily_data: Dict[str, Tuple[float, float]]) -> float:
        """
        Calculate the final liquidity premium factor
        """
        if len(daily_data) < self.period_days:
            return np.nan
            
        # Get the last T days of data
        recent_dates = sorted(daily_data.keys())[-self.period_days:]
        
        sum_cap_need = sum(daily_data[date][0] for date in recent_dates)
        sum_cap_actual = sum(daily_data[date][1] for date in recent_dates)
        
        if sum_cap_actual == 0:
            return np.nan
            
        # Calculate relative difference as liquidity premium factor
        factor_value = (sum_cap_need - sum_cap_actual) / sum_cap_actual
        
        return factor_value


class LiquidityPremiumStrategy:
    """High-frequency liquidity premium factor strategy implementation"""
    
    def __init__(self, api_key: str):
        self.client = PolygonDataClient(api_key)
        self.factor_data = {}
        self.price_data = {}
        self.positions = {}
        
    def calculate_liquidity_premium_factor(self, quotes_df: pd.DataFrame, trades_df: pd.DataFrame, 
                                         amount: float = 100000, time_window: int = 10) -> float:
        """
        Calculate liquidity premium factor based on order book data
        
        Parameters:
        - quotes_df: DataFrame with bid/ask quotes
        - trades_df: DataFrame with trade data
        - amount: Trading amount in currency units
        - time_window: Time window in minutes
        
        Returns:
        - Liquidity premium factor value
        """
        if quotes_df.empty or trades_df.empty:
            return np.nan
            
        # Filter data for the specified time window
        end_time = quotes_df['timestamp'].max()
        start_time = end_time - pd.Timedelta(minutes=time_window)
        
        quotes_filtered = quotes_df[(quotes_df['timestamp'] >= start_time) & 
                                   (quotes_df['timestamp'] <= end_time)]
        
        if len(quotes_filtered) < 2:
            return np.nan
            
        # Calculate average price during the period
        avg_price = trades_df['price'].mean()
        
        # Simulate market impact for given amount
        simulated_cost = 0
        remaining_amount = amount
        
        for _, quote in quotes_filtered.iterrows():
            if remaining_amount <= 0:
                break
                
            # Calculate available liquidity at current quote
            available_value = quote['ask_price'] * quote['ask_size']
            
            if available_value > 0:
                trade_amount = min(remaining_amount, available_value)
                simulated_cost += trade_amount
                remaining_amount -= trade_amount
        
        # Calculate liquidity premium
        if simulated_cost > 0:
            avg_cost = amount * avg_price
            liquidity_premium = (simulated_cost - avg_cost) / avg_cost
            return liquidity_premium
        
        return np.nan
    
    def load_data(self, start_date: str, end_date: str):
        """Load historical data for all stocks"""
        tickers = self.client.get_nasdaq100_tickers()
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        logger.info(f"Loading data for {len(tickers)} stocks from {start_date} to {end_date}")
        
        for ticker in tickers:
            logger.info(f"Processing {ticker}")
            
            # Load daily price data
            daily_data = self.client.get_daily_bars(ticker, start_date, end_date)
            if not daily_data.empty:
                self.price_data[ticker] = daily_data
            
            # Load intraday data for factor calculation
            ticker_factors = {}
            
            for date in date_range:
                date_str = date.strftime('%Y-%m-%d')
                
                # Get quotes and trades data
                quotes_df = self.client.get_quotes_data(ticker, date_str)
                trades_df = self.client.get_trades_data(ticker, date_str)
                
                if not quotes_df.empty and not trades_df.empty:
                    # Calculate factors with different parameters
                    factors = {}
                    
                    # Different amount and time window combinations
                    params = [
                        (10000, 10), (10000, 30),
                        (100000, 10), (100000, 30),
                        (1000000, 10), (1000000, 30)
                    ]
                    
                    for amount, time_window in params:
                        factor_key = f"{amount}_{time_window}"
                        factor_value = self.calculate_liquidity_premium_factor(
                            quotes_df, trades_df, amount, time_window
                        )
                        factors[factor_key] = factor_value
                    
                    ticker_factors[date_str] = factors
                
                time.sleep(0.1)  # Rate limiting
            
            self.factor_data[ticker] = ticker_factors
    
    def calculate_factors(self, start_date: str, end_date: str):
        """Calculate rolling factors and improvements"""
        logger.info("Calculating rolling factors and improvements")
        
        for ticker in self.factor_data:
            ticker_data = self.factor_data[ticker]
            
            # Calculate 21-day rolling factors
            for date_str in ticker_data:
                date = pd.to_datetime(date_str)
                
                # Get 21-day window
                window_start = date - pd.Timedelta(days=30)
                window_dates = []
                
                for check_date in pd.date_range(window_start, date, freq='D'):
                    check_str = check_date.strftime('%Y-%m-%d')
                    if check_str in ticker_data:
                        window_dates.append(check_str)
                
                if len(window_dates) >= 10:  # Minimum data requirement
                    # Calculate rolling averages
                    for param_key in ticker_data[date_str]:
                        values = [ticker_data[d][param_key] for d in window_dates[-21:] 
                                 if not pd.isna(ticker_data[d][param_key])]
                        
                        if len(values) >= 5:
                            # Basic factor
                            ticker_data[date_str][f"{param_key}_rolling"] = np.mean(values)
                            
                            # Time-weighted factor
                            weights = np.exp(np.linspace(-1, 0, len(values)))
                            weights = weights / weights.sum()
                            ticker_data[date_str][f"{param_key}_time_weighted"] = np.average(values, weights=weights)
                            
                            # Volatility-weighted factor (if price data available)
                            if ticker in self.price_data:
                                price_data = self.price_data[ticker]
                                volatility_weights = []
                                
                                for d in window_dates[-len(values):]:
                                    price_date = pd.to_datetime(d)
                                    if price_date in price_data.index:
                                        daily_return = price_data.loc[price_date, 'close'] / price_data.loc[price_date, 'open'] - 1
                                        volatility_weights.append(abs(daily_return))
                                
                                if len(volatility_weights) == len(values):
                                    vol_weights = np.array(volatility_weights)
                                    vol_weights = vol_weights / vol_weights.sum()
                                    ticker_data[date_str][f"{param_key}_vol_weighted"] = np.average(values, weights=vol_weights)
    
    def run_backtest(self, start_date: str, end_date: str, 
                    factor_key: str = "100000_30_rolling", 
                    rebalance_freq: str = "M") -> Dict:
        """
        Run backtest for the liquidity premium factor strategy
        
        Parameters:
        - start_date: Start date for backtest
        - end_date: End date for backtest
        - factor_key: Which factor variant to use
        - rebalance_freq: Rebalancing frequency ('D', 'W', 'M')
        
        Returns:
        - Dictionary with backtest results
        """
        logger.info(f"Running backtest from {start_date} to {end_date}")
        
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        rebalance_dates = pd.date_range(start=start_date, end=end_date, freq=rebalance_freq)
        
        portfolio_values = []
        returns = []
        positions_history = []
        
        for i, current_date in enumerate(date_range):
            date_str = current_date.strftime('%Y-%m-%d')
            
            # Rebalance portfolio
            if current_date in rebalance_dates:
                self.rebalance_portfolio(date_str, factor_key)
            
            # Calculate daily return
            daily_return = 0
            if self.positions:
                for ticker, weight in self.positions.items():
                    if ticker in self.price_data and date_str in self.price_data[ticker].index:
                        price_data = self.price_data[ticker]
                        if i > 0:
                            prev_date = date_range[i-1].strftime('%Y-%m-%d')
                            if prev_date in price_data.index:
                                stock_return = (price_data.loc[date_str, 'close'] / 
                                              price_data.loc[prev_date, 'close']) - 1
                                daily_return += weight * stock_return
            
            returns.append(daily_return)
            
            # Calculate portfolio value
            if i == 0:
                portfolio_values.append(1.0)
            else:
                portfolio_values.append(portfolio_values[-1] * (1 + daily_return))
            
            positions_history.append(self.positions.copy())
        
        # Calculate performance metrics
        returns_series = pd.Series(returns, index=date_range)
        
        results = {
            'returns': returns_series,
            'portfolio_values': pd.Series(portfolio_values, index=date_range),
            'positions_history': positions_history,
            'total_return': portfolio_values[-1] - 1,
            'annualized_return': (portfolio_values[-1] ** (252 / len(date_range))) - 1,
            'volatility': returns_series.std() * np.sqrt(252),
            'sharpe_ratio': (returns_series.mean() * 252) / (returns_series.std() * np.sqrt(252)),
            'max_drawdown': self.calculate_max_drawdown(portfolio_values)
        }
        
        return results
    
    def rebalance_portfolio(self, date_str: str, factor_key: str, top_n: int = 20):
        """Rebalance portfolio based on factor scores"""
        factor_scores = {}
        
        # Calculate factor scores for all stocks
        for ticker in self.factor_data:
            if date_str in self.factor_data[ticker]:
                factor_value = self.factor_data[ticker][date_str].get(factor_key, np.nan)
                if not pd.isna(factor_value):
                    factor_scores[ticker] = factor_value
        
        if len(factor_scores) < top_n:
            logger.warning(f"Only {len(factor_scores)} stocks available for {date_str}")
            return
        
        # Select top stocks (lowest liquidity premium = most liquid)
        sorted_stocks = sorted(factor_scores.items(), key=lambda x: x[1])
        selected_stocks = [stock[0] for stock in sorted_stocks[:top_n]]
        
        # Equal weight allocation
        weight = 1.0 / len(selected_stocks)
        self.positions = {ticker: weight for ticker in selected_stocks}
    
    def calculate_max_drawdown(self, portfolio_values: List[float]) -> float:
        """Calculate maximum drawdown"""
        peak = portfolio_values[0]
        max_dd = 0
        
        for value in portfolio_values:
            if value > peak:
                peak = value
            
            drawdown = (peak - value) / peak
            if drawdown > max_dd:
                max_dd = drawdown
        
        return max_dd


# Example usage and main execution
if __name__ == "__main__":
    
    # Initialize strategy
    API_KEY = "YOUR_POLYGON_API_KEY"
    strategy = LiquidityPremiumStrategy(API_KEY)
    
    # Define backtest period
    start_date = "2020-01-01"
    end_date = "2025-06-30"
    
    try:
        # Load data
        logger.info("Loading market data...")
        strategy.load_data(start_date, end_date)
        
        # Calculate factors
        logger.info("Calculating liquidity premium factors...")
        strategy.calculate_factors(start_date, end_date)
        
        # Run backtests for different factor variants
        factor_variants = [
            "100000_30_rolling",
            "100000_30_time_weighted", 
            "100000_30_vol_weighted"
        ]
        
        results = {}
        for variant in factor_variants:
            logger.info(f"Running backtest for {variant}")
            results[variant] = strategy.run_backtest(
                start_date, end_date, 
                factor_key=variant,
                rebalance_freq="M"
            )
        
        # Display results
        print("\n" + "="*50)
        print("BACKTEST RESULTS SUMMARY")
        print("="*50)
        
        for variant, result in results.items():
            print(f"\n{variant.upper()}:")
            print(f"Total Return: {result['total_return']:.2%}")
            print(f"Annualized Return: {result['annualized_return']:.2%}")
            print(f"Volatility: {result['volatility']:.2%}")
            print(f"Sharpe Ratio: {result['sharpe_ratio']:.2f}")
            print(f"Max Drawdown: {result['max_drawdown']:.2%}")
        
        # Generate performance report using quantstats
        try:
            
            for variant, result in results.items():
                print(f"\n{variant} Performance Report:")
                qs.reports.html(result['returns'], output=f"{variant}_performance.html")
                
        except ImportError:
            logger.warning("quantstats not available for detailed reporting")
            
    except Exception as e:
        logger.error(f"Error during execution: {e}")
        raise
