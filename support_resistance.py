"""
Support and Resistance Level Detection Module

This module detects support and resistance levels on M1 (1-minute) timeframe
using pivot points and local extremes.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class SupportResistanceLevel:
    """Represents a support or resistance level."""
    price: float
    level_type: str  # 'support' or 'resistance'
    strength: int  # Number of times the level was touched
    pivot_based: bool  # True if derived from pivot points
    local_extreme_based: bool  # True if derived from local extremes
    first_touch: int  # Bar index of first touch
    last_touch: int  # Bar index of last touch
    touches: List[int]  # List of bar indices where level was touched


class SupportResistanceDetector:
    """
    Detects support and resistance levels using pivot points and local extremes
    on M1 timeframe data.
    """

    def __init__(
        self,
        tolerance_pips: float = 0.0005,
        min_strength: int = 2,
        local_extreme_period: int = 5,
        pivot_lookback: int = 20
    ):
        """
        Initialize the support and resistance detector.

        Args:
            tolerance_pips: Price tolerance in pips for grouping levels
            min_strength: Minimum number of touches to consider a level valid
            local_extreme_period: Period for detecting local highs/lows
            pivot_lookback: Lookback period for pivot point calculation
        """
        self.tolerance_pips = tolerance_pips
        self.min_strength = min_strength
        self.local_extreme_period = local_extreme_period
        self.pivot_lookback = pivot_lookback
        self.levels: Dict[str, List[SupportResistanceLevel]] = {
            'support': [],
            'resistance': []
        }

    def detect_pivot_points(
        self,
        df: pd.DataFrame
    ) -> Tuple[List[float], List[float]]:
        """
        Detect pivot points based on daily/previous bar pivots.

        Args:
            df: DataFrame with OHLC data (columns: 'open', 'high', 'low', 'close')

        Returns:
            Tuple of (pivot_levels, pivot_resistances_and_supports)
        """
        pivots = []
        resistances = []
        supports = []

        for i in range(self.pivot_lookback, len(df)):
            high = df['high'].iloc[i - self.pivot_lookback:i].max()
            low = df['low'].iloc[i - self.pivot_lookback:i].min()
            close = df['close'].iloc[i - self.pivot_lookback]

            # Standard pivot formula
            pivot = (high + low + close) / 3
            resistance = (2 * pivot) - low
            support = (2 * pivot) - high

            pivots.append(pivot)
            resistances.append(resistance)
            supports.append(support)

        return pivots, resistances, supports

    def detect_local_extremes(
        self,
        df: pd.DataFrame,
        period: Optional[int] = None
    ) -> Tuple[List[float], List[float], List[int], List[int]]:
        """
        Detect local highs and lows based on a rolling window.

        Args:
            df: DataFrame with OHLC data
            period: Period for local extreme detection (uses instance default if None)

        Returns:
            Tuple of (local_highs, local_lows, high_indices, low_indices)
        """
        if period is None:
            period = self.local_extreme_period

        highs = df['high'].values
        lows = df['low'].values
        local_highs = []
        local_lows = []
        high_indices = []
        low_indices = []

        for i in range(period, len(df) - period):
            # Check if it's a local high
            if highs[i] == highs[i - period:i + period + 1].max():
                local_highs.append(highs[i])
                high_indices.append(i)

            # Check if it's a local low
            if lows[i] == lows[i - period:i + period + 1].min():
                local_lows.append(lows[i])
                low_indices.append(i)

        return local_highs, local_lows, high_indices, low_indices

    def group_levels(
        self,
        prices: List[float],
        level_type: str,
        source: str
    ) -> Dict[float, List[int]]:
        """
        Group prices within tolerance range and assign touch indices.

        Args:
            prices: List of price levels
            level_type: 'support' or 'resistance'
            source: Source of the levels ('pivot' or 'local_extreme')

        Returns:
            Dictionary of grouped levels with their touch counts
        """
        if not prices:
            return {}

        grouped = defaultdict(list)
        sorted_prices = sorted(enumerate(prices), key=lambda x: x[1])

        used = set()
        for idx, price in sorted_prices:
            if idx in used:
                continue

            # Find all prices within tolerance
            group = [idx]
            used.add(idx)

            for other_idx, other_price in sorted_prices:
                if other_idx not in used:
                    if abs(other_price - price) <= self.tolerance_pips * price:
                        group.append(other_idx)
                        used.add(other_idx)

            # Use average of grouped prices
            avg_price = np.mean([prices[i] for i in group])
            grouped[avg_price] = group

        return grouped

    def calculate_level_strength(
        self,
        df: pd.DataFrame,
        level_price: float,
        touch_indices: List[int]
    ) -> Tuple[int, List[int]]:
        """
        Calculate the strength of a support/resistance level based on touches.

        Args:
            df: OHLC DataFrame
            level_price: Price level to check
            touch_indices: Initial indices of touches

        Returns:
            Tuple of (strength, validated_touch_indices)
        """
        touches = []
        tolerance = self.tolerance_pips * level_price

        for i in range(len(df)):
            high = df['high'].iloc[i]
            low = df['low'].iloc[i]

            # Check if bar touches the level (high >= level >= low or low <= level <= high)
            if low <= level_price <= high or abs(low - level_price) <= tolerance or abs(high - level_price) <= tolerance:
                touches.append(i)

        return len(touches), touches

    def detect(self, df: pd.DataFrame) -> Dict[str, List[SupportResistanceLevel]]:
        """
        Detect support and resistance levels using both methods.

        Args:
            df: DataFrame with OHLC data (columns: 'open', 'high', 'low', 'close')

        Returns:
            Dictionary with 'support' and 'resistance' keys containing detected levels
        """
        if len(df) < max(self.pivot_lookback, self.local_extreme_period * 2 + 1):
            raise ValueError(
                f"Insufficient data. Need at least {max(self.pivot_lookback, self.local_extreme_period * 2 + 1)} bars"
            )

        all_levels: Dict[str, List[Tuple[float, bool, bool]]] = {
            'support': [],
            'resistance': []
        }

        # Detect pivot-based levels
        pivots, resistances, supports = self.detect_pivot_points(df)
        all_levels['resistance'].extend([(r, True, False) for r in resistances])
        all_levels['support'].extend([(s, True, False) for s in supports])

        # Detect local extreme-based levels
        local_highs, local_lows, high_indices, low_indices = self.detect_local_extremes(df)
        all_levels['resistance'].extend([(h, False, True) for h in local_highs])
        all_levels['support'].extend([(l, False, True) for l in local_lows])

        # Process each level type
        detected_levels = {'support': [], 'resistance': []}

        for level_type in ['support', 'resistance']:
            # Group prices within tolerance
            prices = [p for p, _, _ in all_levels[level_type]]
            sources = [(pb, le) for _, pb, le in all_levels[level_type]]

            grouped = self.group_levels(prices, level_type, 'combined')

            # Create SupportResistanceLevel objects
            for avg_price, group_indices in grouped.items():
                # Verify actual touches in price data
                strength, touch_indices = self.calculate_level_strength(
                    df, avg_price, group_indices
                )

                if strength >= self.min_strength:
                    # Determine if this level has pivot-based or local extreme touches
                    pivot_based = any(sources[i][0] for i in group_indices if i < len(sources))
                    local_extreme_based = any(sources[i][1] for i in group_indices if i < len(sources))

                    level = SupportResistanceLevel(
                        price=avg_price,
                        level_type=level_type,
                        strength=strength,
                        pivot_based=pivot_based,
                        local_extreme_based=local_extreme_based,
                        first_touch=min(touch_indices) if touch_indices else 0,
                        last_touch=max(touch_indices) if touch_indices else 0,
                        touches=touch_indices
                    )
                    detected_levels[level_type].append(level)

            # Sort by strength (descending)
            detected_levels[level_type].sort(
                key=lambda x: x.strength, reverse=True
            )

        self.levels = detected_levels
        return detected_levels

    def get_nearest_support(
        self,
        current_price: float
    ) -> Optional[SupportResistanceLevel]:
        """
        Get the nearest support level below the current price.

        Args:
            current_price: Current market price

        Returns:
            Nearest support level or None
        """
        supports = [s for s in self.levels['support'] if s.price < current_price]
        if supports:
            return max(supports, key=lambda x: x.price)
        return None

    def get_nearest_resistance(
        self,
        current_price: float
    ) -> Optional[SupportResistanceLevel]:
        """
        Get the nearest resistance level above the current price.

        Args:
            current_price: Current market price

        Returns:
            Nearest resistance level or None
        """
        resistances = [r for r in self.levels['resistance'] if r.price > current_price]
        if resistances:
            return min(resistances, key=lambda x: x.price)
        return None

    def get_all_levels(self) -> Dict[str, List[SupportResistanceLevel]]:
        """
        Get all detected support and resistance levels.

        Returns:
            Dictionary with 'support' and 'resistance' keys
        """
        return self.levels

    def summary(self) -> str:
        """
        Generate a summary of detected levels.

        Returns:
            Formatted string summary
        """
        summary_text = "Support and Resistance Levels Summary\n"
        summary_text += "=" * 50 + "\n\n"

        for level_type in ['resistance', 'support']:
            summary_text += f"{level_type.upper()} LEVELS:\n"
            summary_text += "-" * 50 + "\n"

            if not self.levels[level_type]:
                summary_text += f"No {level_type} levels detected.\n"
            else:
                for i, level in enumerate(self.levels[level_type], 1):
                    sources = []
                    if level.pivot_based:
                        sources.append("pivot")
                    if level.local_extreme_based:
                        sources.append("local extreme")

                    summary_text += (
                        f"{i}. Price: {level.price:.5f} | "
                        f"Strength: {level.strength} | "
                        f"Source: {', '.join(sources)}\n"
                    )

            summary_text += "\n"

        return summary_text


# Example usage
if __name__ == "__main__":
    # Create sample M1 OHLC data
    dates = pd.date_range(start='2025-01-01', periods=100, freq='1min')
    sample_data = {
        'timestamp': dates,
        'open': np.random.uniform(1.0500, 1.0510, 100),
        'high': np.random.uniform(1.0510, 1.0520, 100),
        'low': np.random.uniform(1.0490, 1.0500, 100),
        'close': np.random.uniform(1.0500, 1.0510, 100)
    }

    df = pd.DataFrame(sample_data)

    # Detect support and resistance levels
    detector = SupportResistanceDetector(
        tolerance_pips=0.0005,
        min_strength=2,
        local_extreme_period=5,
        pivot_lookback=20
    )

    levels = detector.detect(df)

    # Print summary
    print(detector.summary())

    # Get nearest levels to current price
    current_price = 1.0505
    nearest_support = detector.get_nearest_support(current_price)
    nearest_resistance = detector.get_nearest_resistance(current_price)

    print(f"Current Price: {current_price:.5f}")
    if nearest_support:
        print(f"Nearest Support: {nearest_support.price:.5f} (Strength: {nearest_support.strength})")
    if nearest_resistance:
        print(f"Nearest Resistance: {nearest_resistance.price:.5f} (Strength: {nearest_resistance.strength})")
