"""
Market-agnostic trading primitives.

Every concept here is broker- and market-independent. A Zerodha Kite order
and an Interactive Brokers order are both `Order` instances. A Mumbai NSE
stock and a NYSE stock are both `Instrument` instances with a different
`Exchange` value.

Currency is INR everywhere in Kestrel today. When we add non-INR markets, add
a `currency` field to Instrument and `currency` to monetary fields.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional


class Side(Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    SL = "SL"       # Stop-loss limit
    SLM = "SL-M"    # Stop-loss market


class Product(Enum):
    CNC = "CNC"      # Cash & Carry — delivery
    MIS = "MIS"      # Margin Intraday Square-off
    NRML = "NRML"    # Normal — overnight F&O


class Exchange(Enum):
    NSE = "NSE"
    BSE = "BSE"
    # Future: NYSE, NASDAQ, LSE, SGX...


@dataclass(frozen=True)
class Instrument:
    symbol: str
    exchange: Exchange = Exchange.NSE
    token: Optional[int] = None      # Broker-specific instrument token
    tick_size: float = 0.05
    lot_size: int = 1

    @property
    def display(self) -> str:
        return f"{self.exchange.value}:{self.symbol}"


@dataclass(frozen=True)
class Order:
    instrument: Instrument
    side: Side
    quantity: int
    order_type: OrderType = OrderType.MARKET
    product: Product = Product.CNC
    price: Optional[float] = None           # For LIMIT / SL
    trigger_price: Optional[float] = None   # For SL / SL-M
    tag: Optional[str] = None               # Strategy attribution tag


@dataclass(frozen=True)
class Fill:
    instrument: Instrument
    side: Side
    quantity: int
    price: float
    timestamp: datetime
    order_id: str
    tag: Optional[str] = None


@dataclass
class Position:
    instrument: Instrument
    quantity: int          # +ve long, -ve short
    avg_price: float
    entry_time: datetime
    tag: Optional[str] = None

    @property
    def is_long(self) -> bool:
        return self.quantity > 0

    def unrealized_pnl_inr(self, ltp: float) -> float:
        return (ltp - self.avg_price) * self.quantity


@dataclass
class Trade:
    """A completed round-trip: entry → exit. One row per trade in the journal."""
    strategy: str
    instrument: Instrument
    side: Side
    quantity: int
    entry_time: datetime
    entry_price: float
    exit_time: datetime
    exit_price: float
    gross_pnl_inr: float
    net_pnl_inr: float      # After charges
    return_pct: float       # (exit - entry) / entry, signed
    charges_inr: float
    tag: Optional[str] = None   # Gate condition, scorecard result, exit reason
