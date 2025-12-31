"""
Dynamic lot sizing module based on account risk management.
Implements position sizing strategies that limit risk to a percentage of account equity.
"""

from typing import Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class LotSizingStrategy(Enum):
    """Enumeration of available lot sizing strategies."""
    FIXED_PERCENTAGE = "fixed_percentage"
    KELLY_CRITERION = "kelly_criterion"
    VOLATILITY_ADJUSTED = "volatility_adjusted"


@dataclass
class PositionSize:
    """Data class to hold position sizing results."""
    lot_size: float
    position_quantity: float
    risk_amount: float
    entry_price: float
    stop_loss_price: float
    potential_loss: float
    potential_gain: Optional[float] = None


class DynamicLotSizer:
    """
    Dynamic lot sizing calculator for intraday trading.
    
    This class implements position sizing based on account risk parameters,
    allowing traders to maintain consistent risk management across all trades.
    """
    
    def __init__(self, account_equity: float, risk_percentage: float = 5.0):
        """
        Initialize the lot sizer with account parameters.
        
        Args:
            account_equity: Current account equity in base currency
            risk_percentage: Maximum risk per trade as percentage of account (default: 5%)
        """
        if account_equity <= 0:
            raise ValueError("Account equity must be positive")
        if not (0 < risk_percentage <= 100):
            raise ValueError("Risk percentage must be between 0 and 100")
        
        self.account_equity = account_equity
        self.risk_percentage = risk_percentage
    
    def calculate_risk_amount(self) -> float:
        """
        Calculate the maximum amount that can be risked in a single trade.
        
        Returns:
            Maximum risk amount in base currency
        """
        return self.account_equity * (self.risk_percentage / 100)
    
    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss_price: float,
        instrument: str = "unknown",
        lot_size: float = 1.0,
        take_profit_price: Optional[float] = None,
    ) -> PositionSize:
        """
        Calculate the position size based on entry and stop loss prices.
        
        This implements the core 5% risk rule: risk amount is fixed at 5% of
        account equity, and position size is calculated based on the distance
        to the stop loss level.
        
        Args:
            entry_price: Entry price for the trade
            stop_loss_price: Stop loss price (exit if hit)
            instrument: Trade instrument name (for logging/reference)
            lot_size: Lot size unit (default: 1.0, can be set to 0.01 for micro lots)
            take_profit_price: Optional take profit price for reward calculation
        
        Returns:
            PositionSize object containing position details
        
        Raises:
            ValueError: If prices are invalid or entry equals stop loss
        """
        if entry_price <= 0 or stop_loss_price <= 0:
            raise ValueError("Entry and stop loss prices must be positive")
        
        if entry_price == stop_loss_price:
            raise ValueError("Entry price cannot equal stop loss price")
        
        # Determine if this is a long or short position
        is_long = entry_price > stop_loss_price
        
        # Calculate risk per unit
        price_difference = abs(entry_price - stop_loss_price)
        
        # Calculate maximum risk amount (5% of account)
        max_risk_amount = self.calculate_risk_amount()
        
        # Calculate position quantity based on risk per pip/point
        position_quantity = max_risk_amount / price_difference
        
        # Round to nearest lot size
        position_quantity = (position_quantity // lot_size) * lot_size
        
        if position_quantity <= 0:
            raise ValueError(
                f"Position size too small for current account and risk parameters. "
                f"Required minimum risk per pip: {max_risk_amount / position_quantity if position_quantity else 'infinite'}"
            )
        
        # Calculate actual potential loss
        potential_loss = position_quantity * price_difference
        
        # Calculate potential gain if take profit is provided
        potential_gain = None
        if take_profit_price is not None:
            if take_profit_price <= 0:
                raise ValueError("Take profit price must be positive")
            tp_difference = abs(take_profit_price - entry_price)
            potential_gain = position_quantity * tp_difference
        
        return PositionSize(
            lot_size=lot_size,
            position_quantity=position_quantity,
            risk_amount=potential_loss,
            entry_price=entry_price,
            stop_loss_price=stop_loss_price,
            potential_loss=potential_loss,
            potential_gain=potential_gain,
        )
    
    def calculate_kelly_criterion_size(
        self,
        entry_price: float,
        stop_loss_price: float,
        win_rate: float,
        take_profit_price: Optional[float] = None,
        lot_size: float = 1.0,
    ) -> PositionSize:
        """
        Calculate position size using Kelly Criterion formula.
        
        Kelly Criterion: f* = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        
        Args:
            entry_price: Entry price for the trade
            stop_loss_price: Stop loss price
            win_rate: Historical win rate as decimal (e.g., 0.55 for 55%)
            take_profit_price: Target profit price
            lot_size: Lot size unit
        
        Returns:
            PositionSize object with Kelly-adjusted sizing
        """
        if not (0 < win_rate < 1):
            raise ValueError("Win rate must be between 0 and 1")
        
        # Get base position size
        base_position = self.calculate_position_size(
            entry_price, stop_loss_price, lot_size=lot_size,
            take_profit_price=take_profit_price
        )
        
        # Calculate win/loss amounts per unit
        loss_per_unit = abs(entry_price - stop_loss_price)
        
        if take_profit_price is None:
            raise ValueError("Kelly criterion requires take_profit_price")
        
        win_per_unit = abs(take_profit_price - entry_price)
        
        # Kelly formula
        kelly_fraction = (
            (win_rate * win_per_unit) - ((1 - win_rate) * loss_per_unit)
        ) / win_per_unit
        
        # Apply Kelly fraction to position size (commonly use half-Kelly for safety)
        half_kelly_fraction = kelly_fraction / 2
        
        if half_kelly_fraction <= 0:
            raise ValueError("Calculated Kelly fraction is non-positive; system is not profitable")
        
        kelly_adjusted_quantity = base_position.position_quantity * half_kelly_fraction
        kelly_adjusted_quantity = (kelly_adjusted_quantity // lot_size) * lot_size
        
        return PositionSize(
            lot_size=lot_size,
            position_quantity=kelly_adjusted_quantity,
            risk_amount=kelly_adjusted_quantity * loss_per_unit,
            entry_price=entry_price,
            stop_loss_price=stop_loss_price,
            potential_loss=kelly_adjusted_quantity * loss_per_unit,
            potential_gain=kelly_adjusted_quantity * win_per_unit if take_profit_price else None,
        )
    
    def calculate_volatility_adjusted_size(
        self,
        entry_price: float,
        stop_loss_price: float,
        volatility_ratio: float = 1.0,
        lot_size: float = 1.0,
        take_profit_price: Optional[float] = None,
    ) -> PositionSize:
        """
        Calculate position size adjusted for market volatility.
        
        Higher volatility reduces position size to maintain consistent risk.
        Lower volatility increases position size when conditions are stable.
        
        Args:
            entry_price: Entry price for the trade
            stop_loss_price: Stop loss price
            volatility_ratio: Volatility ratio (1.0 = baseline, >1.0 = high volatility)
            lot_size: Lot size unit
            take_profit_price: Optional take profit price
        
        Returns:
            PositionSize object with volatility-adjusted sizing
        """
        if volatility_ratio <= 0:
            raise ValueError("Volatility ratio must be positive")
        
        # Get base position size
        base_position = self.calculate_position_size(
            entry_price, stop_loss_price, lot_size=lot_size,
            take_profit_price=take_profit_price
        )
        
        # Adjust position size inversely with volatility
        volatility_adjusted_quantity = base_position.position_quantity / volatility_ratio
        volatility_adjusted_quantity = (volatility_adjusted_quantity // lot_size) * lot_size
        
        loss_per_unit = abs(entry_price - stop_loss_price)
        
        return PositionSize(
            lot_size=lot_size,
            position_quantity=volatility_adjusted_quantity,
            risk_amount=volatility_adjusted_quantity * loss_per_unit,
            entry_price=entry_price,
            stop_loss_price=stop_loss_price,
            potential_loss=volatility_adjusted_quantity * loss_per_unit,
            potential_gain=(
                volatility_adjusted_quantity * abs(take_profit_price - entry_price)
                if take_profit_price else None
            ),
        )
    
    def update_account_equity(self, new_equity: float):
        """
        Update the account equity for dynamic risk calculations.
        
        Call this method after trades to recalculate position sizes based
        on the updated account balance.
        
        Args:
            new_equity: Updated account equity
        """
        if new_equity <= 0:
            raise ValueError("Account equity must be positive")
        self.account_equity = new_equity
    
    def get_summary(self) -> dict:
        """
        Get a summary of current lot sizing parameters.
        
        Returns:
            Dictionary containing risk parameters and calculations
        """
        max_risk = self.calculate_risk_amount()
        return {
            "account_equity": self.account_equity,
            "risk_percentage": self.risk_percentage,
            "max_risk_per_trade": max_risk,
            "account_currency": "BASE",  # Can be extended with currency parameter
        }


def calculate_position_size_simple(
    account_equity: float,
    entry_price: float,
    stop_loss_price: float,
    risk_percentage: float = 5.0,
    lot_size: float = 1.0,
) -> Tuple[float, float, float]:
    """
    Simple utility function to calculate position size.
    
    Args:
        account_equity: Current account balance
        entry_price: Trade entry price
        stop_loss_price: Stop loss price
        risk_percentage: Risk percentage (default 5%)
        lot_size: Lot size unit
    
    Returns:
        Tuple of (position_quantity, risk_amount, entry_to_sl_distance)
    """
    sizer = DynamicLotSizer(account_equity, risk_percentage)
    position = sizer.calculate_position_size(
        entry_price, stop_loss_price, lot_size=lot_size
    )
    return position.position_quantity, position.risk_amount, position.potential_loss
