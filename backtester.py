"""
Backtester module for intraday trading strategy evaluation.

This module provides functionality to backtest trading strategies using
historical OHLC (Open, High, Low, Close) data from CSV files.
"""

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict


@dataclass
class OHLCBar:
    """Represents a single OHLC candlestick bar."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: Optional[float] = None

    def __str__(self) -> str:
        return (f"{self.timestamp} | O: {self.open} H: {self.high} "
                f"L: {self.low} C: {self.close}")


@dataclass
class Trade:
    """Represents a single trade transaction."""
    entry_time: datetime
    entry_price: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    quantity: float = 1.0
    trade_type: str = "long"  # 'long' or 'short'

    @property
    def is_closed(self) -> bool:
        """Check if trade is closed."""
        return self.exit_time is not None and self.exit_price is not None

    @property
    def pnl(self) -> Optional[float]:
        """Calculate profit/loss for closed trades."""
        if not self.is_closed:
            return None
        if self.trade_type == "long":
            return (self.exit_price - self.entry_price) * self.quantity
        else:  # short
            return (self.entry_price - self.exit_price) * self.quantity

    @property
    def pnl_pct(self) -> Optional[float]:
        """Calculate profit/loss percentage for closed trades."""
        if not self.is_closed:
            return None
        return (self.pnl / (self.entry_price * self.quantity)) * 100

    @property
    def duration_minutes(self) -> Optional[int]:
        """Calculate trade duration in minutes."""
        if not self.is_closed:
            return None
        return int((self.exit_time - self.entry_time).total_seconds() / 60)


@dataclass
class BacktestResults:
    """Container for backtest statistics and results."""
    total_trades: int
    winning_trades: int
    losing_trades: int
    total_pnl: float
    total_pnl_pct: float
    win_rate: float
    average_win: Optional[float]
    average_loss: Optional[float]
    profit_factor: Optional[float]
    largest_win: Optional[float]
    largest_loss: Optional[float]
    average_trade_duration: Optional[float]
    max_drawdown: Optional[float]
    max_drawdown_pct: Optional[float]
    trades: List[Trade]
    start_date: datetime
    end_date: datetime
    data_bars: int

    def to_dict(self) -> Dict:
        """Convert results to dictionary."""
        result_dict = {
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "total_pnl": round(self.total_pnl, 2),
            "total_pnl_pct": round(self.total_pnl_pct, 2),
            "win_rate": round(self.win_rate, 2),
            "average_win": round(self.average_win, 2) if self.average_win else None,
            "average_loss": round(self.average_loss, 2) if self.average_loss else None,
            "profit_factor": round(self.profit_factor, 2) if self.profit_factor else None,
            "largest_win": round(self.largest_win, 2) if self.largest_win else None,
            "largest_loss": round(self.largest_loss, 2) if self.largest_loss else None,
            "average_trade_duration_minutes": round(self.average_trade_duration, 0) if self.average_trade_duration else None,
            "max_drawdown": round(self.max_drawdown, 2) if self.max_drawdown else None,
            "max_drawdown_pct": round(self.max_drawdown_pct, 2) if self.max_drawdown_pct else None,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "data_bars": self.data_bars,
        }
        return result_dict

    def print_summary(self) -> None:
        """Print a formatted summary of backtest results."""
        print("\n" + "="*60)
        print("BACKTEST RESULTS SUMMARY")
        print("="*60)
        print(f"Test Period: {self.start_date.date()} to {self.end_date.date()}")
        print(f"Data Bars: {self.data_bars}")
        print(f"\nTrade Statistics:")
        print(f"  Total Trades: {self.total_trades}")
        print(f"  Winning Trades: {self.winning_trades}")
        print(f"  Losing Trades: {self.losing_trades}")
        print(f"  Win Rate: {self.win_rate:.2f}%")
        print(f"\nProfit & Loss:")
        print(f"  Total P&L: ${self.total_pnl:,.2f}")
        print(f"  Total P&L %: {self.total_pnl_pct:.2f}%")
        print(f"  Average Win: ${self.average_win:.2f}" if self.average_win else "  Average Win: N/A")
        print(f"  Average Loss: ${self.average_loss:.2f}" if self.average_loss else "  Average Loss: N/A")
        print(f"  Largest Win: ${self.largest_win:,.2f}" if self.largest_win else "  Largest Win: N/A")
        print(f"  Largest Loss: ${self.largest_loss:,.2f}" if self.largest_loss else "  Largest Loss: N/A")
        if self.profit_factor:
            print(f"  Profit Factor: {self.profit_factor:.2f}")
        print(f"\nRisk Metrics:")
        print(f"  Max Drawdown: ${self.max_drawdown:,.2f}" if self.max_drawdown else "  Max Drawdown: N/A")
        print(f"  Max Drawdown %: {self.max_drawdown_pct:.2f}%" if self.max_drawdown_pct else "  Max Drawdown %: N/A")
        if self.average_trade_duration:
            print(f"  Avg Trade Duration: {int(self.average_trade_duration)} minutes")
        print("="*60 + "\n")


class CSVDataLoader:
    """Loads OHLC data from CSV files."""

    @staticmethod
    def load_from_csv(filepath: Path, 
                     date_format: str = "%Y-%m-%d %H:%M:%S",
                     date_column: str = "timestamp",
                     open_column: str = "open",
                     high_column: str = "high",
                     low_column: str = "low",
                     close_column: str = "close",
                     volume_column: Optional[str] = "volume") -> List[OHLCBar]:
        """
        Load OHLC data from a CSV file.

        Args:
            filepath: Path to the CSV file
            date_format: Format string for timestamp parsing
            date_column: Name of the date/time column
            open_column: Name of the open price column
            high_column: Name of the high price column
            low_column: Name of the low price column
            close_column: Name of the close price column
            volume_column: Name of the volume column (optional)

        Returns:
            List of OHLCBar objects

        Raises:
            FileNotFoundError: If the CSV file doesn't exist
            ValueError: If required columns are missing or data is invalid
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"CSV file not found: {filepath}")

        bars = []
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                if not reader.fieldnames:
                    raise ValueError("CSV file is empty or has no headers")

                # Validate required columns
                required_columns = {date_column, open_column, high_column, 
                                  low_column, close_column}
                missing_columns = required_columns - set(reader.fieldnames)
                if missing_columns:
                    raise ValueError(f"Missing required columns: {missing_columns}")

                for row_num, row in enumerate(reader, start=2):
                    try:
                        timestamp = datetime.strptime(row[date_column].strip(), date_format)
                        open_price = float(row[open_column])
                        high_price = float(row[high_column])
                        low_price = float(row[low_column])
                        close_price = float(row[close_column])
                        volume = float(row[volume_column]) if volume_column and row.get(volume_column) else None

                        bar = OHLCBar(
                            timestamp=timestamp,
                            open=open_price,
                            high=high_price,
                            low=low_price,
                            close=close_price,
                            volume=volume
                        )
                        bars.append(bar)
                    except (ValueError, KeyError) as e:
                        raise ValueError(f"Error parsing row {row_num}: {e}")

        except csv.Error as e:
            raise ValueError(f"CSV parsing error: {e}")

        if not bars:
            raise ValueError("No valid data found in CSV file")

        return bars


class Backtester:
    """Main backtester class for evaluating trading strategies."""

    def __init__(self, initial_capital: float = 10000.0):
        """
        Initialize the backtester.

        Args:
            initial_capital: Starting capital for the backtest
        """
        self.initial_capital = initial_capital
        self.trades: List[Trade] = []
        self.bars: List[OHLCBar] = []

    def load_data(self, filepath: Path, **kwargs) -> None:
        """
        Load OHLC data from a CSV file.

        Args:
            filepath: Path to the CSV file
            **kwargs: Additional arguments to pass to CSVDataLoader.load_from_csv
        """
        self.bars = CSVDataLoader.load_from_csv(filepath, **kwargs)

    def add_trade(self, trade: Trade) -> None:
        """Add a trade to the backtest."""
        self.trades.append(trade)

    def add_trades(self, trades: List[Trade]) -> None:
        """Add multiple trades to the backtest."""
        self.trades.extend(trades)

    def _calculate_drawdown(self) -> Tuple[Optional[float], Optional[float]]:
        """
        Calculate maximum drawdown and drawdown percentage.

        Returns:
            Tuple of (max_drawdown_value, max_drawdown_percentage)
        """
        if not self.trades:
            return None, None

        closed_trades = [t for t in self.trades if t.is_closed]
        if not closed_trades:
            return None, None

        cumulative_pnl = 0
        peak_pnl = 0
        max_drawdown = 0
        max_drawdown_pct = 0

        for trade in closed_trades:
            cumulative_pnl += trade.pnl
            if cumulative_pnl > peak_pnl:
                peak_pnl = cumulative_pnl
            drawdown = peak_pnl - cumulative_pnl
            if drawdown > max_drawdown:
                max_drawdown = drawdown
                max_drawdown_pct = (drawdown / peak_pnl * 100) if peak_pnl != 0 else 0

        return max_drawdown if max_drawdown > 0 else None, max_drawdown_pct if max_drawdown_pct > 0 else None

    def calculate_results(self) -> BacktestResults:
        """
        Calculate backtest statistics.

        Returns:
            BacktestResults object with all metrics
        """
        closed_trades = [t for t in self.trades if t.is_closed]
        winning_trades = [t for t in closed_trades if t.pnl > 0]
        losing_trades = [t for t in closed_trades if t.pnl < 0]

        total_trades = len(closed_trades)
        total_pnl = sum(t.pnl for t in closed_trades)
        total_pnl_pct = (total_pnl / self.initial_capital * 100) if self.initial_capital > 0 else 0
        win_rate = (len(winning_trades) / total_trades * 100) if total_trades > 0 else 0

        average_win = (sum(t.pnl for t in winning_trades) / len(winning_trades)) if winning_trades else None
        average_loss = (sum(t.pnl for t in losing_trades) / len(losing_trades)) if losing_trades else None

        largest_win = max((t.pnl for t in winning_trades), default=None)
        largest_loss = min((t.pnl for t in losing_trades), default=None)

        # Profit factor (gross profit / gross loss)
        gross_profit = sum(t.pnl for t in winning_trades) if winning_trades else 0
        gross_loss = abs(sum(t.pnl for t in losing_trades)) if losing_trades else 0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else None

        # Average trade duration
        trade_durations = [t.duration_minutes for t in closed_trades if t.duration_minutes]
        average_trade_duration = (sum(trade_durations) / len(trade_durations)) if trade_durations else None

        # Drawdown calculations
        max_drawdown, max_drawdown_pct = self._calculate_drawdown()

        # Date range
        start_date = self.bars[0].timestamp if self.bars else datetime.now()
        end_date = self.bars[-1].timestamp if self.bars else datetime.now()

        return BacktestResults(
            total_trades=total_trades,
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            total_pnl=total_pnl,
            total_pnl_pct=total_pnl_pct,
            win_rate=win_rate,
            average_win=average_win,
            average_loss=average_loss,
            profit_factor=profit_factor,
            largest_win=largest_win,
            largest_loss=largest_loss,
            average_trade_duration=average_trade_duration,
            max_drawdown=max_drawdown,
            max_drawdown_pct=max_drawdown_pct,
            trades=closed_trades,
            start_date=start_date,
            end_date=end_date,
            data_bars=len(self.bars)
        )

    def save_results_to_json(self, filepath: Path, results: BacktestResults) -> None:
        """
        Save backtest results to a JSON file.

        Args:
            filepath: Path where to save the JSON file
            results: BacktestResults object to save
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        results_dict = results.to_dict()
        results_dict["trades"] = [
            {
                "entry_time": t.entry_time.isoformat(),
                "entry_price": t.entry_price,
                "exit_time": t.exit_time.isoformat() if t.exit_time else None,
                "exit_price": t.exit_price,
                "quantity": t.quantity,
                "trade_type": t.trade_type,
                "pnl": round(t.pnl, 2) if t.pnl else None,
                "pnl_pct": round(t.pnl_pct, 2) if t.pnl_pct else None,
                "duration_minutes": t.duration_minutes
            }
            for t in results.trades
        ]

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, indent=2)

    def reset(self) -> None:
        """Reset the backtester to initial state."""
        self.trades = []
        self.bars = []


# Example usage
if __name__ == "__main__":
    # Example: Load data and run backtest
    try:
        backtester = Backtester(initial_capital=10000.0)
        
        # Load OHLC data from CSV
        # backtester.load_data(
        #     Path("data/AAPL_1min.csv"),
        #     date_format="%Y-%m-%d %H:%M:%S",
        #     date_column="timestamp",
        #     open_column="open",
        #     high_column="high",
        #     low_column="low",
        #     close_column="close",
        #     volume_column="volume"
        # )
        
        # Example trades (replace with actual strategy logic)
        # sample_trade = Trade(
        #     entry_time=datetime(2025, 1, 1, 10, 0, 0),
        #     entry_price=150.0,
        #     exit_time=datetime(2025, 1, 1, 11, 30, 0),
        #     exit_price=152.5,
        #     quantity=10,
        #     trade_type="long"
        # )
        # backtester.add_trade(sample_trade)
        
        # Calculate and display results
        # results = backtester.calculate_results()
        # results.print_summary()
        
        print("Backtester module loaded successfully!")
        print("Use backtester.load_data() to load CSV files and implement your strategy.")
        
    except Exception as e:
        print(f"Error: {e}")
