import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict
from enum import Enum
from collections import deque
import warnings
import logging
import json
import os
from datetime import datetime

# PostgreSQL
import psycopg2
from psycopg2.extras import execute_values
from psycopg2 import pool

warnings.filterwarnings('ignore')

# Set professional plotting style
sns.set_style("darkgrid")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = '#f8f8f8'
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'sans-serif'

logging.basicConfig(level=logging.WARNING, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)
_log_handlers = logger.handlers

# ============================================================================
# POSTGRESQL CONFIGURATION AND TELEMETRY DB
# ============================================================================

@dataclass
class PostgreSQLConfig:
    """PostgreSQL connection configuration (reads from environment variables)"""
    host:     str = os.getenv("PG_HOST",     "localhost")
    port:     int = int(os.getenv("PG_PORT", "5432"))
    dbname:   str = os.getenv("PG_DBNAME",  "silver_trading")
    user:     str = os.getenv("PG_USER",     "trading_user")
    password: str = os.getenv("PG_PASSWORD", "your_password_here")
    min_conn: int = 1
    max_conn: int = 10


class TelemetryDB:
    """PostgreSQL telemetry backend with connection pooling"""
    
    def __init__(self, pg_config: PostgreSQLConfig):
        self.cfg = pg_config
        self._pool: Optional[pool.SimpleConnectionPool] = None
        self._connected = False
    
    def connect(self) -> bool:
        """Open connection pool. Returns True on success, False on failure."""
        try:
            self._pool = pool.SimpleConnectionPool(
                self.cfg.min_conn,
                self.cfg.max_conn,
                host=self.cfg.host,
                port=self.cfg.port,
                dbname=self.cfg.dbname,
                user=self.cfg.user,
                password=self.cfg.password,
            )
            self._connected = True
            logger.info("PostgreSQL connection pool created")
            return True
        except psycopg2.OperationalError as e:
            logger.warning(f"PostgreSQL unavailable – falling back to in-memory mode: {e}")
            self._connected = False
            return False
    
    def _get_conn(self):
        if not self._connected or self._pool is None:
            raise RuntimeError("Not connected to PostgreSQL")
        return self._pool.getconn()
    
    def _put_conn(self, conn):
        if self._pool:
            self._pool.putconn(conn)
    
    def disconnect(self):
        if self._pool:
            self._pool.closeall()
            self._connected = False
    
    @property
    def connected(self) -> bool:
        return self._connected
    
    def create_schema(self):
        """Idempotent schema creation with IF NOT EXISTS"""
        ddl = """
        CREATE TABLE IF NOT EXISTS simulation_runs (
            id               SERIAL PRIMARY KEY,
            run_id           UUID DEFAULT gen_random_uuid(),
            started_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            n_ticks          INTEGER,
            initial_capital  NUMERIC(14,4),
            final_capital    NUMERIC(14,4),
            total_return_pct NUMERIC(10,4),
            sharpe_ratio     NUMERIC(10,4),
            sortino_ratio    NUMERIC(10,4),
            max_drawdown_pct NUMERIC(10,4),
            win_rate_pct     NUMERIC(10,4),
            profit_factor    NUMERIC(10,4),
            total_trades     INTEGER,
            avg_win          NUMERIC(14,4),
            avg_loss         NUMERIC(14,4),
            latency_mean     NUMERIC(10,2),
            fill_prob_base   NUMERIC(6,4),
            slippage_base    NUMERIC(6,4),
            inventory_kappa  NUMERIC(6,4),
            config_json      JSONB
        );

        CREATE TABLE IF NOT EXISTS telemetry_events (
            id            BIGSERIAL PRIMARY KEY,
            simulation_id INTEGER REFERENCES simulation_runs(id) ON DELETE CASCADE,
            ts            TIMESTAMPTZ NOT NULL,
            price         NUMERIC(14,5),
            z_score       NUMERIC(10,4),
            p_bull        NUMERIC(6,4),
            rv            NUMERIC(10,4),
            regime        VARCHAR(10),
            event_type    VARCHAR(30),
            reason        VARCHAR(50),
            pnl           NUMERIC(14,4)
        );

        CREATE TABLE IF NOT EXISTS trade_log (
            id              BIGSERIAL PRIMARY KEY,
            simulation_id   INTEGER REFERENCES simulation_runs(id) ON DELETE CASCADE,
            opened_at       TIMESTAMPTZ NOT NULL,
            closed_at       TIMESTAMPTZ,
            side            VARCHAR(5),
            entry_price     NUMERIC(14,5),
            exit_price      NUMERIC(14,5),
            size            NUMERIC(10,4),
            entry_z         NUMERIC(10,4),
            exit_reason     VARCHAR(50),
            pnl             NUMERIC(14,4),
            holding_secs    NUMERIC(10,1),
            regime          VARCHAR(10),
            latency_ms      NUMERIC(8,2),
            fill_prob       NUMERIC(6,4),
            slippage_pips   NUMERIC(8,4),
            adverse_select  BOOLEAN,
            stop_loss       NUMERIC(14,5),
            rv_at_entry     NUMERIC(10,4)
        );

        CREATE INDEX IF NOT EXISTS idx_tel_sim_ts
            ON telemetry_events (simulation_id, ts);
        CREATE INDEX IF NOT EXISTS idx_tel_event_type
            ON telemetry_events (event_type);
        CREATE INDEX IF NOT EXISTS idx_trade_sim
            ON trade_log (simulation_id, opened_at);
        CREATE INDEX IF NOT EXISTS idx_runs_started
            ON simulation_runs (started_at);
        """
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(ddl)
            conn.commit()
            logger.info("Schema created / verified")
        finally:
            self._put_conn(conn)
    
    def insert_simulation_run(self, metrics: dict, config: 'SimulatorConfig', n_ticks: int) -> int:
        """Insert simulation summary row. Returns the new simulation id."""
        sql = """
        INSERT INTO simulation_runs (
            n_ticks, initial_capital, final_capital,
            total_return_pct, sharpe_ratio, sortino_ratio,
            max_drawdown_pct, win_rate_pct, profit_factor,
            total_trades, avg_win, avg_loss,
            latency_mean, fill_prob_base, slippage_base, inventory_kappa,
            config_json
        ) VALUES (
            %(n_ticks)s, %(initial_capital)s, %(final_capital)s,
            %(total_return_pct)s, %(sharpe_ratio)s, %(sortino_ratio)s,
            %(max_drawdown_pct)s, %(win_rate_pct)s, %(profit_factor)s,
            %(total_trades)s, %(avg_win)s, %(avg_loss)s,
            %(latency_mean)s, %(fill_prob_base)s, %(slippage_base)s, %(inventory_kappa)s,
            %(config_json)s
        ) RETURNING id
        """
        params = {
            "n_ticks":          n_ticks,
            "initial_capital":  10000,
            "final_capital":    metrics.get("final_capital", 0),
            "total_return_pct": round(metrics.get("total_return", 0) * 100, 4),
            "sharpe_ratio":     metrics.get("sharpe_ratio", 0),
            "sortino_ratio":    metrics.get("sortino_ratio", 0),
            "max_drawdown_pct": round(metrics.get("max_drawdown", 0) * 100, 4),
            "win_rate_pct":     round(metrics.get("win_rate", 0) * 100, 4),
            "profit_factor":    metrics.get("profit_factor", 0),
            "total_trades":     metrics.get("total_trades", 0),
            "avg_win":          metrics.get("avg_win", 0),
            "avg_loss":         metrics.get("avg_loss", 0),
            "latency_mean":     config.latency_mean,
            "fill_prob_base":   config.fill_prob_base,
            "slippage_base":    config.slippage_base,
            "inventory_kappa":  config.inventory_decay_kappa,
            "config_json":      json.dumps({
                "hurst_threshold_low":  config.hurst_threshold_low,
                "z_threshold":          config.z_threshold,
                "rv_circuit_breaker":   config.rv_circuit_breaker,
                "stop_loss_sigma":      config.stop_loss_sigma,
                "profit_target":        config.profit_target,
            }),
        }
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                sim_id = cur.fetchone()[0]
            conn.commit()
            return sim_id
        finally:
            self._put_conn(conn)
    
    def bulk_insert_telemetry(self, simulation_id: int, events: List[dict]):
        """Bulk-insert telemetry events for one simulation (fast path)."""
        if not events:
            return
        rows = [
            (
                simulation_id,
                e["timestamp"],
                e["price"],
                e["z_score"],
                e["p_bull"],
                e["rv"],
                e["regime"],
                e["event"],
                e["reason"],
                e["pnl"],
            )
            for e in events
        ]
        sql = """
        INSERT INTO telemetry_events
            (simulation_id, ts, price, z_score, p_bull, rv, regime, event_type, reason, pnl)
        VALUES %s
        """
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                execute_values(cur, sql, rows, page_size=500)
            conn.commit()
        finally:
            self._put_conn(conn)
    
    def bulk_insert_trades(self, simulation_id: int, trades: List[dict]):
        """Bulk-insert the closed-trade log for one simulation."""
        closed = [t for t in trades if t.get("action") == "CLOSE"]
        if not closed:
            return

        rows = []
        for t in closed:
            rows.append((
                simulation_id,
                t.get("timestamp"),
                t.get("timestamp"),
                None,
                None,
                t.get("exit_price"),
                None,
                None,
                t.get("reason"),
                t.get("pnl"),
                t.get("holding_time"),
                t.get("regime"),
                None, None, None, None, None, None,
            ))

        sql = """
        INSERT INTO trade_log (
            simulation_id, closed_at, opened_at, side,
            entry_price, exit_price, size, entry_z,
            exit_reason, pnl, holding_secs, regime,
            latency_ms, fill_prob, slippage_pips, adverse_select,
            stop_loss, rv_at_entry
        ) VALUES %s
        """
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                execute_values(cur, sql, rows, page_size=200)
            conn.commit()
        finally:
            self._put_conn(conn)
    
    def update_simulation_metrics(self, simulation_id: int, metrics: dict):
        """Update simulation summary with final metrics"""
        sql = """
        UPDATE simulation_runs SET
            final_capital    = %(final_capital)s,
            total_return_pct = %(total_return_pct)s,
            sharpe_ratio     = %(sharpe_ratio)s,
            sortino_ratio    = %(sortino_ratio)s,
            max_drawdown_pct = %(max_drawdown_pct)s,
            win_rate_pct     = %(win_rate_pct)s,
            profit_factor    = %(profit_factor)s,
            total_trades     = %(total_trades)s,
            avg_win          = %(avg_win)s,
            avg_loss         = %(avg_loss)s
        WHERE id = %(id)s
        """
        params = {
            "final_capital":    metrics.get("final_capital", 0),
            "total_return_pct": round(metrics.get("total_return", 0) * 100, 4),
            "sharpe_ratio":     metrics.get("sharpe_ratio", 0),
            "sortino_ratio":    metrics.get("sortino_ratio", 0),
            "max_drawdown_pct": round(metrics.get("max_drawdown", 0) * 100, 4),
            "win_rate_pct":     round(metrics.get("win_rate", 0) * 100, 4),
            "profit_factor":    metrics.get("profit_factor", 0),
            "total_trades":     metrics.get("total_trades", 0),
            "avg_win":          metrics.get("avg_win", 0),
            "avg_loss":         metrics.get("avg_loss", 0),
            "id":               simulation_id,
        }
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(sql, params)
            conn.commit()
        finally:
            self._put_conn(conn)

# ============================================================================
# CONFIGURATION AND CONSTANTS
# ============================================================================

class RegimeState(Enum):
    BULL = "BULL"
    NEUTRAL = "NEUTRAL"
    BEAR = "BEAR"
    
class OrderSide(Enum):
    LONG = 1
    SHORT = -1

@dataclass
class SimulatorConfig:
    """Configuration parameters for realistic execution simulation"""
    # Latency parameters (milliseconds)
    latency_mean: float = 150.0
    latency_std: float = 30.0
    latency_min: float = 50.0
    latency_max: float = 300.0
    
    # Fill probability parameters
    fill_prob_base: float = 0.85
    fill_prob_decay: float = 0.15
    
    # Slippage parameters (pips)
    slippage_base: float = 0.5
    slippage_volatility_factor: float = 0.3
    
    # Market parameters (XAGUSD - Silver 5000oz lot)
    tick_size: float = 0.01  # Silver price granularity (not 0.001 forex)
    min_spread_pips: float = 0.05  # Silver typical spread in USD
    
    # Inventory decay
    inventory_decay_kappa: float = 0.65
    
    # Execution costs
    commission_per_lot: float = 5.0
    lot_size: float = 5000.0  # Silver: 5000 oz per lot (not 100k forex)
    
    # ─────────────────────────────────────────────────────────────
    # METATRADER5 CONFIGURATION
    # ─────────────────────────────────────────────────────────────
    mt5_symbol: str = "XAGUSD"  # V6.3: Silver (not EURUSD)
    mt5_timeframe: str = "1min"
    mt5_n_bars: int = 50000
    use_mt5: bool = False
    
    # ─────────────────────────────────────────────────────────────
    # V6.3 SIGNAL GENERATION PARAMETERS
    # ─────────────────────────────────────────────────────────────
    hurst_threshold_low: float = 0.45
    hurst_threshold_high: float = 0.65
    hurst_windows: List[int] = field(default_factory=lambda: [60, 120, 240, 480, 960])
    hurst_update_freq: int = 60
    
    pmig_threshold: float = 0.45
    migration_decay: float = 0.95
    w_velocity: float = 0.5
    eta_coupling: float = 0.3
    vwap_lookback: int = 60
    velocity_threshold: float = 0.3
    
    regime_bull_threshold: float = 0.65
    regime_bear_threshold: float = 0.35
    regime_temperature: float = 2.0
    regime_decay: float = 0.97
    
    z_threshold: float = 2.8
    rv_circuit_breaker: float = 2.5
    
    # Risk Management
    stop_loss_sigma: float = 3.0
    time_limit_base: float = 900.0
    profit_target: float = 0.95  # V6.3 SPEC: 95% reversion target
    
    # Position Sizing
    base_size: float = 0.1
    max_positions: int = 1
    size_gamma: float = 1.5  # Power law for (1-P_mig)^size_gamma
    
    # ─────────────────────────────────────────────────────────────
    # V7.1 BREATHING ANCHOR PROTOCOL ENHANCEMENTS (Optional)
    # ─────────────────────────────────────────────────────────────
    enable_drift_detection: bool = True
    enable_variance_ratio: bool = True
    enable_breathing_stops: bool = True
    
    # Drift Detection Parameters
    drift_lookback: int = 240  # 4 hours at 1-min
    drift_lambda: float = 0.5  # Reversion penalty weight
    drift_threshold: float = 1.5  # Significant drift threshold
    
    # Variance Ratio Parameters
    vr_short_window: int = 60  # 1 hour
    vr_long_window: int = 300  # 5 hours
    vr_trend_threshold: float = 1.15  # VR > 1.15 indicates trending
    vr_mr_threshold: float = 0.85  # VR < 0.85 indicates mean-reverting
    
    # Enhanced Migration Detection
    migration_lookback: int = 120  # 2 hours
    migration_velocity_threshold: float = 0.25
    migration_persistence_threshold: float = 0.70
    
    # Breathing Stops
    stop_breathing_expand_factor: float = 1.3  # Expand in trending
    stop_breathing_tighten_factor: float = 0.85  # Tighten in MR

@dataclass
class MarketState:
    """Current state of the market"""
    timestamp: pd.Timestamp
    price: float
    bid: float
    ask: float
    volume: float
    vwap_60: float
    sigma_60: float
    RV_1min: float
    regime: RegimeState
    P_bull: float
    hurst: float = 0.5  # V6.3: Hurst exponent (default 0.5 for neutral)
    P_mig: float = 0.5  # V6.3: VWAP migration probability (default 0.5 for neutral)

@dataclass
class Order:
    """Limit order representation"""
    timestamp: pd.Timestamp
    side: OrderSide
    size: float
    limit_price: float
    entry_Z: float
    entry_RV: float
    regime: RegimeState
    
@dataclass
class Position:
    """Open position tracking"""
    entry_time: pd.Timestamp
    entry_price: float
    size: float
    side: OrderSide
    entry_Z: float
    stop_loss: float
    time_limit: pd.Timestamp
    last_Z: float = 0.0  # Track Z-score velocity for V4.7 Elasticity

# ============================================================================
# MARKET MICROSTRUCTURE MODELS
# ============================================================================

class LatencyModel:
    """Models realistic order routing and execution latency"""
    
    def __init__(self, config: SimulatorConfig):
        self.config = config
        
    def sample_latency(self) -> float:
        """Sample latency from truncated normal distribution"""
        latency = np.random.normal(
            self.config.latency_mean,
            self.config.latency_std
        )
        return np.clip(latency, self.config.latency_min, self.config.latency_max)
    
    def apply_latency(
        self,
        original_price: float,
        latency_ms: float,
        volatility: float
    ) -> float:
        """Model price movement during latency period"""
        latency_minutes = latency_ms / 60000.0
        price_std = volatility * np.sqrt(latency_minutes)
        price_change = np.random.normal(0, price_std)
        return original_price + price_change


class FillProbabilityModel:
    """Models realistic limit order fill probability"""
    
    def __init__(self, config: SimulatorConfig):
        self.config = config
        
    def calculate_fill_probability(
        self,
        Z: float,
        RV: float,
        spread_pips: float,
        is_aggressive: bool = False
    ) -> float:
        """Calculate probability of limit order being filled"""
        Z_abs = abs(Z)
        p_base = self.config.fill_prob_base * np.exp(
            -self.config.fill_prob_decay * max(0, Z_abs - 2.5)
        )
        
        # Penalty for high volatility
        if RV > 2.5:
            volatility_penalty = 0.3
        elif RV > 1.5:
            volatility_penalty = 0.15
        else:
            volatility_penalty = 0.0
            
        p_adjusted = p_base * (1 - volatility_penalty)
        
        # Penalty for wide spreads
        spread_penalty = min(0.2, max(0, (spread_pips - 2.0) * 0.05))
        p_adjusted *= (1 - spread_penalty)
        
        if is_aggressive:
            p_adjusted = min(0.95, p_adjusted * 1.5)
        
        return np.clip(p_adjusted, 0.05, 0.95)
    
    def is_filled(self, fill_probability: float) -> bool:
        """Bernoulli trial for order fill"""
        return np.random.random() < fill_probability


class SlippageModel:
    """Models realistic execution slippage"""
    
    def __init__(self, config: SimulatorConfig):
        self.config = config
        
    def calculate_slippage(
        self,
        order_size: float,
        RV: float,
        spread_pips: float,
        is_adverse: bool = False
    ) -> float:
        """Calculate execution slippage in pips"""
        slippage = self.config.slippage_base
        vol_slippage = self.config.slippage_volatility_factor * RV
        slippage += vol_slippage
        impact = 0.5 * np.sqrt(order_size)
        slippage += impact
        
        if is_adverse:
            slippage += spread_pips * 0.5
        
        return slippage


class AdverseSelectionDetector:
    """Detects when limit order fill is due to adverse information"""
    
    def __init__(self, lookback: int = 10):
        self.lookback = lookback
        self.recent_returns = deque(maxlen=lookback)
        
    def update(self, return_value: float):
        """Update with latest return"""
        self.recent_returns.append(return_value)
    
    def is_adverse_selection(
        self,
        fill_price: float,
        current_price: float,
        side: OrderSide,
        RV: float
    ) -> bool:
        """Detect if fill was due to adverse selection"""
        if len(self.recent_returns) < self.lookback:
            return False
        
        price_move = current_price - fill_price
        
        if side == OrderSide.LONG:
            adverse = price_move < 0
        else:
            adverse = price_move > 0
        
        recent_array = np.array(self.recent_returns)
        directional_pct = np.sum(recent_array > 0) / len(recent_array)
        is_directional = directional_pct > 0.70 or directional_pct < 0.30
        
        if RV > 2.0 and is_directional and adverse:
            return True
        
        return False

# ============================================================================
# V7.1 BREATHING ANCHOR PROTOCOL COMPONENTS (Enhanced Detectors)
# ============================================================================

class DriftDetector:
    """
    V7.1: Detects persistent price drift away from mean-reverting behavior
    D_t = Σ(P_t - VWAP_t) - λ * Σ|P_t - VWAP_t|
    """
    
    def __init__(self, config: 'SimulatorConfig'):
        self.config = config
        self.price_history = deque(maxlen=config.drift_lookback)
        self.vwap_history = deque(maxlen=config.drift_lookback)
        self.D_t = 0.0
        
    def update(self, price: float, vwap: float, sigma: float) -> Tuple[float, bool, float]:
        """
        Update drift calculation
        Returns: (D_t, is_drifting, drift_direction)
        """
        self.price_history.append(price)
        self.vwap_history.append(vwap)
        
        if len(self.price_history) < 10:
            return 0.0, False, 0.0
        
        prices = np.array(self.price_history)
        vwaps = np.array(self.vwap_history)
        
        deviations = prices - vwaps
        cumulative_deviation = np.sum(deviations)
        reversion_penalty = self.config.drift_lambda * np.sum(np.abs(deviations))
        
        D_raw = cumulative_deviation - reversion_penalty
        
        if sigma > 0:
            self.D_t = D_raw / (sigma * np.sqrt(len(self.price_history)))
        else:
            self.D_t = 0.0
        
        is_drifting = abs(self.D_t) > self.config.drift_threshold
        drift_direction = np.sign(self.D_t) if is_drifting else 0.0
        
        return self.D_t, is_drifting, drift_direction
    
    def reset(self):
        """Reset drift detector"""
        self.price_history.clear()
        self.vwap_history.clear()
        self.D_t = 0.0


class VarianceRatioCalculator:
    """
    V7.1: Calculate variance ratio to detect trending vs mean-reverting regimes
    VR = Var(R_long) / (k * Var(R_short))
    VR > 1.15: Trending, VR < 0.85: Mean-reverting
    """
    
    def __init__(self, config: 'SimulatorConfig'):
        self.config = config
        self.price_history = deque(maxlen=config.vr_long_window)
        
    def update(self, price: float) -> Tuple[float, str]:
        """
        Calculate variance ratio
        Returns: (VR, regime_type: "TRENDING" | "MEAN_REVERTING" | "RANDOM_WALK")
        """
        self.price_history.append(price)
        
        if len(self.price_history) < self.config.vr_long_window:
            return 1.0, "RANDOM_WALK"
        
        prices = np.array(self.price_history)
        returns_short = np.diff(prices[-self.config.vr_short_window:])
        
        k = self.config.vr_long_window // self.config.vr_short_window
        returns_long = np.diff(prices[::k])
        
        var_short = np.var(returns_short) if len(returns_short) > 0 else 1e-10
        var_long = np.var(returns_long) if len(returns_long) > 0 else 1e-10
        
        VR = var_long / (k * var_short) if var_short > 0 else 1.0
        
        if VR > self.config.vr_trend_threshold:
            regime_type = "TRENDING"
        elif VR < self.config.vr_mr_threshold:
            regime_type = "MEAN_REVERTING"
        else:
            regime_type = "RANDOM_WALK"
        
        return VR, regime_type
    
    def reset(self):
        """Reset calculator"""
        self.price_history.clear()


class EnhancedBreathingMigrationDetector:
    """
    V7.1: Enhanced VWAP migration detector using drift + velocity + VR alignment
    Detects when VWAP anchor is "breathing" (moving persistently)
    """
    
    def __init__(self, config: 'SimulatorConfig'):
        self.config = config
        self.vwap_history = deque(maxlen=config.migration_lookback)
        self.time_history = deque(maxlen=config.migration_lookback)
        self.L_migration = 0.0  # Bayesian log-odds
        
    def update(
        self,
        vwap: float,
        timestamp: pd.Timestamp,
        sigma: float,
        D_t: float,
        VR: float
    ) -> Tuple[float, bool, float, float]:
        """
        Update migration detection
        Returns: (P_migration, is_stable, velocity_norm, persistence_score)
        """
        self.vwap_history.append(vwap)
        self.time_history.append(timestamp)
        
        if len(self.vwap_history) < 2:
            return 0.0, True, 0.0, 0.0
        
        vwap_delta = self.vwap_history[-1] - self.vwap_history[0]
        time_delta = (self.time_history[-1] - self.time_history[0]).total_seconds()
        
        if time_delta == 0 or sigma == 0:
            return 0.0, True, 0.0, 0.0
        
        velocity = vwap_delta / time_delta
        velocity_norm = abs(velocity) / sigma
        
        self.L_migration *= self.config.migration_decay
        
        # Evidence 1: VWAP velocity
        if velocity_norm > self.config.migration_velocity_threshold:
            self.L_migration += 1.0 * np.sign(velocity)
        
        # Evidence 2: Drift alignment
        drift_vwap_alignment = np.sign(D_t) * np.sign(vwap_delta)
        if abs(D_t) > 1.0 and drift_vwap_alignment > 0:
            self.L_migration += 0.8 * drift_vwap_alignment
        
        # Evidence 3: Variance ratio alignment
        if VR > self.config.vr_trend_threshold:
            self.L_migration += 0.5 * np.sign(vwap_delta)
        
        P_migration = 1.0 / (1.0 + np.exp(-self.L_migration))
        is_stable = P_migration < self.config.migration_persistence_threshold
        persistence_score = P_migration if velocity_norm > self.config.migration_velocity_threshold else 0.0
        
        return P_migration, is_stable, velocity_norm, persistence_score
    
    def reset(self):
        """Reset detector"""
        self.vwap_history.clear()
        self.time_history.clear()
        self.L_migration = 0.0


# ============================================================================
# STRATEGY COMPONENTS
# ============================================================================

class RiskManager:
    """Risk management and position sizing"""
    
    def __init__(self, config: SimulatorConfig):
        self.config = config
    
    def check_exit_condition(
        self,
        position: Position,
        current_price: float,
        timestamp: pd.Timestamp,
        vwap: float,
        sigma: float
    ) -> Tuple[bool, str]:
        """Check if position should be exited"""
        # Stop loss check
        if position.side == OrderSide.LONG:
            if current_price < position.stop_loss:
                return True, "STOP_LOSS"
        else:
            if current_price > position.stop_loss:
                return True, "STOP_LOSS"
        
        # Time limit check
        if timestamp >= position.time_limit:
            return True, "TIME_LIMIT"
        
        # Profit target check
        Z = (current_price - vwap) / sigma if sigma > 0 else 0
        if position.side == OrderSide.LONG and Z < self.config.profit_target * -1:
            return True, "PROFIT_TARGET"
        elif position.side == OrderSide.SHORT and Z > self.config.profit_target:
            return True, "PROFIT_TARGET"
        
        return False, ""
    
    def calculate_position_size(
        self,
        base_size: float,
        confidence: float,
        inventory_count: int,
        hurst: float = 0.5,
        P_mig: float = 0.5
    ) -> float:
        """Calculate position size based on confidence and market conditions"""
        if inventory_count >= 1:
            return 0
        
        # Hurst adjustment (mean-reversion favors larger sizes)
        hurst_factor = 1.0 + (0.5 - hurst) * 0.5
        
        # Migration adjustment (stable anchor favors larger sizes)
        mig_factor = 1.0 + (0.5 - abs(P_mig - 0.5)) * 0.3
        
        size = base_size * confidence * hurst_factor * mig_factor
        return max(0.01, min(size, base_size * 2.0))


class ExecutionSimulator:
    """Full execution simulation engine"""
    
    def __init__(self, config: SimulatorConfig, db: Optional[TelemetryDB] = None):
        self.config = config
        self.db = db
        self.capital = 10000.0
        self.initial_capital = 10000.0
        self.equity_curve = [self.capital]
        self.price_history = deque(maxlen=10000)
        self.trades = []
        self.telemetry = []
        self.current_positions: List[Position] = []
        self.simulation_id: Optional[int] = None
        self.n_ticks = 0
        
        # Initialize components
        self.micro_engine = MicroEngine(config)
        self.risk_manager = RiskManager(config)
        self.adverse_detector = AdverseSelectionDetector()
        self.migration_detector = EnhancedBreathingMigrationDetector(config)
        
        # Optional V7.1 detectors
        self.drift_detector = DriftDetector(config) if config.enable_drift_detection else None
        self.vr_calculator = VarianceRatioCalculator(config) if config.enable_variance_ratio else None
        
        self.base_position_size = config.base_size
        self.latency_model = LatencyModel(config)
        self.fill_model = FillProbabilityModel(config)
        self.slippage_model = SlippageModel(config)
    
    def register_simulation(self, n_ticks: int):
        """Register simulation in database"""
        self.n_ticks = n_ticks
        if self.db and self.db.connected:
            self.simulation_id = self.db.insert_simulation_run(
                {},
                self.config,
                n_ticks
            )
    
    def log_telemetry(
        self,
        market_state: MarketState,
        event_type: str,
        reason: str,
        pnl: float = 0
    ):
        """Log telemetry event"""
        event = {
            "timestamp": market_state.timestamp,
            "price": market_state.price,
            "z_score": self.micro_engine.calculate_Z_score(
                market_state.price,
                market_state.vwap_60,
                market_state.sigma_60
            ),
            "p_bull": market_state.P_bull,
            "rv": market_state.RV_1min,
            "regime": market_state.regime.name,
            "event": event_type,
            "reason": reason,
            "pnl": pnl,
        }
        self.telemetry.append(event)
    
    def export_telemetry_df(self) -> pd.DataFrame:
        """Export telemetry as DataFrame"""
        if not self.telemetry:
            return pd.DataFrame()
        return pd.DataFrame(self.telemetry)
    
    def simulate_trade_execution(
        self,
        market_state: MarketState,
        side: OrderSide,
        size: float,
        entry_Z: float
    ) -> Optional[Position]:
        """Simulate trade execution with latency and slippage"""
        latency = self.latency_model.sample_latency()
        execution_price = self.latency_model.apply_latency(
            market_state.price,
            latency,
            market_state.sigma_60
        )
        
        fill_prob = self.fill_model.calculate_fill_probability(
            entry_Z,
            market_state.RV_1min,
            self.config.min_spread_pips
        )
        
        if not self.fill_model.is_filled(fill_prob):
            self.trades.append({
                'timestamp': market_state.timestamp,
                'action': 'REJECTED',
                'reason': 'NO_FILL',
                'price': market_state.price,
                'size': size,
                'side': side.name
            })
            return None
        
        slippage = self.slippage_model.calculate_slippage(
            size,
            market_state.RV_1min,
            self.config.min_spread_pips
        )
        
        if side == OrderSide.LONG:
            execution_price += slippage / 10000
        else:
            execution_price -= slippage / 10000
        
        stop_loss = execution_price - (market_state.sigma_60 * self.config.stop_loss_sigma) \
            if side == OrderSide.LONG else execution_price + (market_state.sigma_60 * self.config.stop_loss_sigma)
        
        time_limit = market_state.timestamp + pd.Timedelta(seconds=self.config.time_limit_base)
        
        position = Position(
            entry_time=market_state.timestamp,
            entry_price=execution_price,
            size=size,
            side=side,
            entry_Z=entry_Z,
            stop_loss=stop_loss,
            time_limit=time_limit
        )
        
        self.trades.append({
            'timestamp': market_state.timestamp,
            'action': 'OPEN',
            'reason': 'SIGNAL',
            'price': execution_price,
            'size': size,
            'side': side.name
        })
        
        self.log_telemetry(market_state, 'TRADE_OPEN', 'SIGNAL', pnl=0)
        return position
    
    def process_tick(self, market_state: MarketState):
        """Process single market tick"""
        self.price_history.append(market_state.price)
        
        if len(self.equity_curve) > 0:
            last_equity = self.equity_curve[-1]
            if last_equity > 0:
                return_val = (self.capital - last_equity) / last_equity
                self.adverse_detector.update(return_val)
        
        positions_to_close = []
        for position in self.current_positions:
            should_exit, reason = self.risk_manager.check_exit_condition(
                position,
                market_state.price,
                market_state.timestamp,
                market_state.vwap_60,
                market_state.sigma_60
            )
            if should_exit:
                positions_to_close.append((position, reason))
        
        for position, reason in positions_to_close:
            self.close_position(position, market_state, reason)
        
        D_t = 0.0
        VR = 1.0
        P_migration = 0.5
        
        if self.drift_detector is not None:
            D_t, _, _ = self.drift_detector.update(
                market_state.price,
                market_state.vwap_60,
                market_state.sigma_60
            )
        
        if self.vr_calculator is not None:
            VR, _ = self.vr_calculator.update(market_state.price)
        
        P_migration, _, _, _ = self.migration_detector.update(
            market_state.vwap_60,
            market_state.timestamp,
            market_state.sigma_60,
            D_t,
            VR
        )
        
        market_state.hurst = 0.5
        market_state.P_mig = P_migration
        
        signal = self.micro_engine.generate_signal(
            market_state,
            market_state.regime,
            market_state.P_bull
        )
        
        if signal is None:
            self.log_telemetry(market_state, 'SIGNAL_REJECTED', 'GATE_BLOCKED', pnl=0)
            self.equity_curve.append(self.capital)
            return
        
        side, confidence = signal
        
        if len(self.current_positions) >= 1:
            self.log_telemetry(market_state, 'SIGNAL_REJECTED', 'INVENTORY_CAP', pnl=0)
            self.equity_curve.append(self.capital)
            return
        
        inventory_count = len([p for p in self.current_positions if p.side == side])
        
        position_size = self.risk_manager.calculate_position_size(
            self.base_position_size,
            confidence,
            inventory_count,
            hurst=market_state.hurst,
            P_mig=P_migration
        )
        
        if position_size == 0:
            self.log_telemetry(market_state, 'SIGNAL_REJECTED', 'ZERO_SIZE', pnl=0)
            self.equity_curve.append(self.capital)
            return
        
        Z = self.micro_engine.calculate_Z_score(
            market_state.price,
            market_state.vwap_60,
            market_state.sigma_60
        )
        
        position = self.simulate_trade_execution(
            market_state,
            side,
            position_size,
            Z
        )
        
        if position is not None:
            self.current_positions.append(position)
        
        self.equity_curve.append(self.capital)
    
    def close_position(self, position: Position, market_state: MarketState, reason: str):
        """Close a position and update capital"""
        risk_per_position = self.initial_capital * 0.005
        
        if position.side == OrderSide.LONG:
            pnl = (market_state.price - position.entry_price) * position.size * (risk_per_position / 0.01)
        else:
            pnl = (position.entry_price - market_state.price) * position.size * (risk_per_position / 0.01)
        
        commission = self.config.commission_per_lot * position.size * 2
        pnl -= commission
        
        self.capital += pnl
        
        self.log_telemetry(market_state, 'TRADE_CLOSE', reason, pnl=pnl)
        
        self.trades.append({
            'timestamp': market_state.timestamp,
            'action': 'CLOSE',
            'reason': reason,
            'exit_price': market_state.price,
            'pnl': pnl,
            'holding_time': (market_state.timestamp - position.entry_time).total_seconds(),
            'regime': market_state.regime.name
        })
        
        self.current_positions.remove(position)
    
    def get_performance_metrics(self) -> dict:
        """Calculate comprehensive performance metrics"""
        if len(self.equity_curve) == 0:
            return {}
        
        equity = np.array(self.equity_curve)
        returns = np.diff(equity) / (equity[:-1] + 1e-9)
        
        total_return = (self.capital - self.initial_capital) / self.initial_capital
        
        if len(returns) > 0 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252 * 1440)
            negative_returns = returns[returns < 0]
            if len(negative_returns) > 0 and np.std(negative_returns) > 0:
                sortino = np.mean(returns) / np.std(negative_returns) * np.sqrt(252 * 1440)
            else:
                sortino = sharpe
            max_dd = self._calculate_max_drawdown(equity)
        else:
            sharpe = 0
            sortino = 0
            max_dd = 0
        
        closed_trades = [t for t in self.trades if t.get('action') == 'CLOSE']
        if len(closed_trades) > 0:
            winning_trades = [t for t in closed_trades if t.get('pnl', 0) > 0]
            losing_trades = [t for t in closed_trades if t.get('pnl', 0) < 0]
            
            win_rate = len(winning_trades) / len(closed_trades)
            avg_win = np.mean([t['pnl'] for t in winning_trades]) if len(winning_trades) > 0 else 0
            avg_loss = np.mean([abs(t['pnl']) for t in losing_trades]) if len(losing_trades) > 0 else 0
            profit_factor = (avg_win * len(winning_trades)) / (avg_loss * len(losing_trades)) if avg_loss > 0 and len(losing_trades) > 0 else 0
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_dd,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_trades': len(closed_trades),
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'final_capital': self.capital
        }
    
    def _calculate_max_drawdown(self, equity: np.ndarray) -> float:
        """Calculate maximum drawdown"""
        if len(equity) == 0:
            return 0
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / (peak + 1e-9)
        return np.min(drawdown)
    
    def print_trading_summary(self):
        """Print comprehensive trading summary"""
        summary_lines = []
        summary_lines.append("=" * 70)
        summary_lines.append("TRADING SESSION SUMMARY")
        summary_lines.append("=" * 70)
        
        pnl = self.capital - self.initial_capital
        pnl_pct = (pnl / self.initial_capital) * 100
        summary_lines.append(f"\n[CAPITAL]")
        summary_lines.append(f"  Initial Capital:  ${self.initial_capital:,.2f}")
        summary_lines.append(f"  Final Capital:    ${self.capital:,.2f}")
        summary_lines.append(f"  Total P&L:        ${pnl:+,.2f} ({pnl_pct:+.2f}%)")
        
        opened_orders = [t for t in self.trades if t.get('action') == 'OPEN']
        rejected_orders = [t for t in self.trades if t.get('action') == 'REJECTED']
        
        summary_lines.append(f"\n[SIGNAL GENERATION]")
        summary_lines.append(f"  Total Signals Generated: {len(opened_orders) + len(rejected_orders)}")
        summary_lines.append(f"  Orders Opened:          {len(opened_orders)}")
        summary_lines.append(f"  Orders Rejected:        {len(rejected_orders)}")
        
        closed_trades = [t for t in self.trades if t.get('action') == 'CLOSE']
        summary_lines.append(f"\n[POSITION MANAGEMENT]")
        summary_lines.append(f"  Total Closed Positions: {len(closed_trades)}")
        summary_lines.append(f"  Open Positions Now:     {len(self.current_positions)}")
        
        metrics = self.get_performance_metrics()
        summary_lines.append(f"\n[PERFORMANCE METRICS]")
        summary_lines.append(f"  Total Return:     {metrics.get('total_return', 0)*100:+.2f}%")
        summary_lines.append(f"  Sharpe Ratio:     {metrics.get('sharpe_ratio', 0):.2f}")
        summary_lines.append(f"  Sortino Ratio:    {metrics.get('sortino_ratio', 0):.2f}")
        summary_lines.append(f"  Max Drawdown:     {metrics.get('max_drawdown', 0)*100:.2f}%")
        summary_lines.append(f"  Win Rate:         {metrics.get('win_rate', 0)*100:.1f}%")
        summary_lines.append(f"  Profit Factor:    {metrics.get('profit_factor', 0):.2f}")
        
        if self.simulation_id:
            summary_lines.append(f"\n  PostgreSQL Simulation ID: {self.simulation_id}")
        
        summary_lines.append("\n" + "=" * 70)
        
        for line in summary_lines:
            print(line)
            logger.info(line)
    
    def finalize(self):
        """Finalize simulation"""
        metrics = self.get_performance_metrics()
        if self.simulation_id and self.db and self.db.connected:
            self.db.update_simulation_metrics(self.simulation_id, metrics)
            self.db.bulk_insert_telemetry(self.simulation_id, self.telemetry)
            self.db.bulk_insert_trades(self.simulation_id, self.trades)


class MacroBrain:
    """V5.0: Two-Speed Hysteresis & Temperature Scaled Probability"""
    
    def __init__(self):
        self.L_t = 0.0
        self.P_bull = 0.5
        self.regime = RegimeState.NEUTRAL
        
        # Hyperparameters
        self.temperature = 2.0  # Institutional Fix #2: Softens the sigmoid
        self.decay_lambda = 0.95
        
        # Component Weights
        self.W_VR = 0.6
        self.W_discovery = 0.8
        self.W_volatility = 0.8  # Increased per V4.9 findings
        
    def calculate_VR(self, returns_5d: np.ndarray, returns_1d: np.ndarray) -> float:
        """Calculate variance ratio"""
        var_5d = np.var(returns_5d) if len(returns_5d) > 0 else 0
        var_1d = np.var(returns_1d) if len(returns_1d) > 0 else 0
        
        if var_1d == 0:
            return 1.0
        
        return var_5d / (5 * var_1d)
    
    def calculate_discovery_phase(
        self,
        price: float,
        vwap_50d: float,
        sigma_50d: float
    ) -> float:
        """Calculate standardized distance from 50-day anchor"""
        if sigma_50d == 0:
            return 0.0
        return (price - vwap_50d) / sigma_50d
    
    def calculate_volatility_ratio(self, sigma_10d: float, sigma_60d: float) -> float:
        """Calculate volatility ratio"""
        if sigma_60d == 0:
            return 1.0
        return sigma_10d / sigma_60d
    
    def update_daily(
        self,
        VR: float,
        D_t: float,
        vol_ratio: float,
        price_trend: float
    ) -> Tuple[float, RegimeState]:
        """Daily Bayesian update with Temperature Scaling"""
        self.L_t *= self.decay_lambda
        
        # Calculate Signals
        VR_signal = self.W_VR * np.sign(price_trend) if VR > 1.05 else 0.0
        discovery_signal = self.W_discovery * np.sign(D_t) if abs(D_t) > 1.5 else 0.0
        vol_signal = self.W_volatility * np.sign(price_trend) if vol_ratio > 1.2 else 0.0
        
        # Log-Odds Update
        self.L_t += VR_signal + discovery_signal + vol_signal
        
        # INSTITUTIONAL FIX #2: Temperature Scaling
        # Flattens the curve, expanding the tails
        self.P_bull = 1.0 / (1.0 + np.exp(-self.L_t / self.temperature))
        
        # INSTITUTIONAL FIX #1: Two-Speed Hysteresis
        # Inertia-based transitions with asymmetric thresholds
        if self.regime == RegimeState.NEUTRAL:
            if self.P_bull > 0.65:
                self.regime = RegimeState.BULL
            elif self.P_bull < 0.35:
                self.regime = RegimeState.BEAR
            
        elif self.regime == RegimeState.BULL:
            if self.P_bull < 0.45:  # Requires significant degradation to exit BULL
                self.regime = RegimeState.NEUTRAL
            
        elif self.regime == RegimeState.BEAR:
            if self.P_bull > 0.55:  # Requires significant improvement to exit BEAR
                self.regime = RegimeState.NEUTRAL
        
        return self.P_bull, self.regime


class MicroEngine:
    """Tick-by-tick signal generation"""
    
    def __init__(self, config: SimulatorConfig):
        self.config = config
        self.base_Z_threshold = 2.9
        self.RV_circuit_breaker = 2.5
        
    def calculate_Z_score(self, price: float, vwap: float, sigma: float) -> float:
        """Calculate Z-score from VWAP"""
        if sigma == 0:
            return 0.0
        return (price - vwap) / sigma
    
    def calculate_adaptive_threshold(self, RV: float) -> float:
        """Adaptive Z-threshold based on volatility"""
        alpha = 0.3
        return self.base_Z_threshold * (1 + alpha * RV)
    
    def generate_signal(
        self,
        market_state: MarketState,
        regime: RegimeState,
        P_bull: float
    ) -> Optional[Tuple[OrderSide, float]]:
        """Generate trading signal if conditions met"""
        if market_state.RV_1min > self.RV_circuit_breaker:
            logger.debug(f"Circuit breaker: RV={market_state.RV_1min:.2f} > {self.RV_circuit_breaker}")
            return None
        
        Z = self.calculate_Z_score(
            market_state.price,
            market_state.vwap_60,
            market_state.sigma_60
        )
        
        Z_threshold = self.calculate_adaptive_threshold(market_state.RV_1min)
        
        logger.debug(f"Z={Z:.4f}, Threshold={Z_threshold:.4f}, Price={market_state.price:.4f}, VWAP={market_state.vwap_60:.4f}")
        
        if abs(Z) < Z_threshold:
            logger.debug(f"|Z|={abs(Z):.4f} < {Z_threshold:.4f} - no signal")
            return None
        
        logger.info(f"*** SIGNAL: {('SHORT' if Z > Z_threshold else 'LONG')}, Z={Z:.4f} ***")
        side = OrderSide.SHORT if Z > Z_threshold else OrderSide.LONG
        
        if regime == RegimeState.BULL and side == OrderSide.SHORT:
            logger.info(f"Signal blocked: BULL regime rejects SHORT")
            return None
        elif regime == RegimeState.BEAR and side == OrderSide.LONG:
            logger.info(f"Signal blocked: BEAR regime rejects LONG")
            return None
        
        z_confidence = min(abs(Z) / Z_threshold, 1.0)
        regime_confidence = abs(P_bull - 0.5) * 2.0
        confidence = z_confidence * 0.8 + regime_confidence * 0.2
        confidence = max(confidence, 0.30)
        
        logger.info(f"FINAL SIGNAL: {side.name}, Confidence={confidence:.4f}, Regime={regime.name}")
        return (side, confidence)

# ============================================================================
# MONTE CARLO BATCH RUNNER - WITH POSTGRESQL & REAL-TIME VISUALIZATIONS
# ============================================================================

class SimulationVisualizer:
    """Real-time visualization of simulation results"""
    
    def __init__(self, sim_idx: int, results_df: pd.DataFrame = None):
        self.sim_idx = sim_idx
        self.results_df = results_df if results_df is not None else pd.DataFrame()
    
    def plot_simulation_results(self, sim: ExecutionSimulator, save_prefix: str = ""):
        """Plot single simulation results in real-time"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'Simulation #{self.sim_idx + 1} - Results', fontsize=16, fontweight='bold')
        
        # 1. Equity Curve
        ax = axes[0, 0]
        ax.plot(sim.equity_curve, color='steelblue', linewidth=2)
        ax.fill_between(range(len(sim.equity_curve)), sim.equity_curve, alpha=0.3)
        ax.set_title('Equity Curve', fontweight='bold')
        ax.set_ylabel('Capital ($)')
        ax.set_xlabel('Tick')
        ax.grid(True, alpha=0.3)
        
        # 2. Drawdown
        ax = axes[0, 1]
        equity_array = np.array(sim.equity_curve)
        running_max = np.maximum.accumulate(equity_array)
        drawdown = (equity_array - running_max) / running_max * 100
        ax.fill_between(range(len(drawdown)), drawdown, color='red', alpha=0.5)
        ax.set_title('Drawdown Over Time', fontweight='bold')
        ax.set_ylabel('Drawdown (%)')
        ax.set_xlabel('Tick')
        ax.grid(True, alpha=0.3)
        
        # 3. Price vs Trades
        ax = axes[0, 2]
        ax.plot(sim.price_history, color='gray', linewidth=1, label='Price')
        
        #  Mark trade entries and exits
        opens = [t for t in sim.trades if t.get('action') == 'OPEN']
        closes = [t for t in sim.trades if t.get('action') == 'CLOSE']
        
        # (simplified - real implementation would match index to timestamp)
        if opens and closes:
            ax.set_title('Price & Trade Activity', fontweight='bold')
            ax.set_ylabel('Price ($)')
            ax.set_xlabel('Tick')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 4. Win/Loss Distribution
        ax = axes[1, 0]
        closed = [t for t in sim.trades if t.get('action') == 'CLOSE']
        pnls = [t.get('pnl', 0) for t in closed]
        
        if pnls:
            wins = [p for p in pnls if p > 0]
            losses = [p for p in pnls if p < 0]
            
            ax.bar(['Wins', 'Losses'], [len(wins), len(losses)], color=['green', 'red'], alpha=0.7, edgecolor='black')
            ax.set_title('Trade Outcome Distribution', fontweight='bold')
            ax.set_ylabel('Count')
            ax.grid(True, alpha=0.3, axis='y')
        
        # 5. Telemetry Events
        ax = axes[1, 1]
        df_telem = sim.export_telemetry_df()
        if len(df_telem) > 0:
            event_counts = df_telem['event'].value_counts()
            colors = {'SIGNAL_GENERATED': 'blue', 'SIGNAL_REJECTED': 'orange', 'TRADE_OPEN': 'green', 'TRADE_CLOSE': 'red'}
            event_colors = [colors.get(e, 'gray') for e in event_counts.index]
            ax.bar(event_counts.index, event_counts.values, color=event_colors, alpha=0.7, edgecolor='black')
            ax.set_title('Telemetry Event Breakdown', fontweight='bold')
            ax.set_ylabel('Count')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3, axis='y')
        
        # 6. Performance Metrics
        ax = axes[1, 2]
        metrics = sim.get_performance_metrics()
        
        ax.axis('off')
        metrics_text = f"""
PERFORMANCE SUMMARY

Return:        {metrics.get('total_return', 0)*100:+.2f}%
Sharpe:        {metrics.get('sharpe_ratio', 0):.2f}
Sortino:       {metrics.get('sortino_ratio', 0):.2f}
Max DD:        {metrics.get('max_drawdown', 0)*100:.2f}%
Win Rate:      {metrics.get('win_rate', 0)*100:.1f}%
Profit Factor: {metrics.get('profit_factor', 0):.2f}

Trades:        {metrics.get('total_trades', 0)}
Avg Win:       ${metrics.get('avg_win', 0):.2f}
Avg Loss:      ${metrics.get('avg_loss', 0):.2f}
        """
        ax.text(0.1, 0.5, metrics_text, fontsize=11, family='monospace',
                verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        # Save and show
        if save_prefix:
            save_path = f"{save_prefix}_sim_{self.sim_idx:03d}.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"  📊 Saved: {save_path}")
        
        plt.show(block=False)
        plt.pause(0.5)
        plt.close()

def run_monte_carlo_batch(
    n_simulations: int = 100,
    n_ticks: int = 5000,
    vary_parameters: bool = True,
    base_config: SimulatorConfig = SimulatorConfig(),
    verbose_simulations: int = 5,
    return_simulators: bool = False,
    db: Optional[TelemetryDB] = None,
    show_plots: bool = True,
    plot_every: int = 5,
) -> tuple:
    """
    Run Monte Carlo simulations with PostgreSQL logging and real-time visualization.
    
    Args:
        show_plots: If True, display matplotlib visualization after each sim
        plot_every: Plot every N simulations (default 5)
        db: Optional TelemetryDB connection
    """
    results = []
    simulators = []
    
    print(f"Running {n_simulations} Monte Carlo simulations...")
    print(f"Ticks per simulation: {n_ticks}")
    print(f"PostgreSQL telemetry: {'ENABLED' if (db and db.connected) else 'DISABLED (in-memory fallback)'}")
    print(f"Real-time visualizations: {'ENABLED' if show_plots else 'DISABLED'}")
    print("-" * 60)
    
    log_handlers = logging.root.handlers if logging.root.handlers else [logging.StreamHandler()]
    
    for sim_idx in range(n_simulations):
        if sim_idx == verbose_simulations:
            for handler in log_handlers:
                handler.setLevel(logging.WARNING)
            print(f"[Sim {sim_idx}] Switching to silent mode...")
            print("-" * 60)
        
        if (sim_idx + 1) % 10 == 0:
            print(f"Progress: {sim_idx + 1}/{n_simulations}")
        
        # Parameter variation
        if vary_parameters:
            config = SimulatorConfig(
                latency_mean=np.random.uniform(100, 200),
                latency_std=np.random.uniform(20, 40),
                fill_prob_base=0.85,
                fill_prob_decay=0.15,
                slippage_base=np.random.uniform(0.3, 0.7),
                inventory_decay_kappa=np.random.uniform(0.5, 0.8)
            )
        else:
            config = base_config
        
        sim = ExecutionSimulator(config=config, db=db)
        sim.register_simulation(n_ticks)
        
        # Generate market data
        timestamps = pd.date_range("2024-01-01", periods=n_ticks, freq="1min")
        base_price = 30.0
        price_series = [base_price]
        current_regime = RegimeState.NEUTRAL
        regime_duration = 0
        
        for i, ts in enumerate(timestamps):
            # Regime switching
            if regime_duration <= 0:
                regime_probs = [0.3, 0.4, 0.3]
                current_regime = np.random.choice(
                    [RegimeState.BULL, RegimeState.NEUTRAL, RegimeState.BEAR],
                    p=regime_probs
                )
                regime_duration = np.random.randint(50, 200)
            regime_duration -= 1
            
            # Price dynamics
            if current_regime == RegimeState.BULL:
                drift = 0.0003
            elif current_regime == RegimeState.BEAR:
                drift = -0.0003
            else:
                drift = 0
            
            if i > 0:
                noise = np.random.normal(0, 0.01)
                new_price = price_series[-1] + noise + drift
                price_series.append(new_price)
            
            price = price_series[-1]
            vwap = np.mean(price_series[max(0, i-60):i+1])
            sigma = max(np.std(price_series[max(0, i-60):i+1]), 0.01)
            
            market_state = MarketState(
                timestamp=ts,
                price=price,
                bid=price - 0.002,
                ask=price + 0.002,
                volume=np.random.uniform(100, 1000),
                vwap_60=vwap,
                sigma_60=sigma,
                RV_1min=np.random.uniform(0.5, 2.0),
                regime=current_regime,
                P_bull=0.9 if current_regime == RegimeState.BULL else 
                       0.1 if current_regime == RegimeState.BEAR else 0.5,
                hurst=np.random.uniform(0.40, 0.60),
                P_mig=np.random.uniform(0.30, 0.50),
            )
            
            sim.process_tick(market_state)
        
        sim.finalize()
        
        metrics = sim.get_performance_metrics()
        metrics['simulation_id'] = sim_idx
        metrics['latency_mean'] = config.latency_mean
        metrics['fill_prob_base'] = config.fill_prob_base
        metrics['slippage_base'] = config.slippage_base
        metrics['inventory_kappa'] = config.inventory_decay_kappa
        
        results.append(metrics)
        simulators.append(sim)
        
        # Real-time visualization
        if show_plots and ((sim_idx + 1) % plot_every == 0 or sim_idx == 0):
            print(f"\n[Visualization] Plotting simulation #{sim_idx + 1}...")
            viz = SimulationVisualizer(sim_idx, pd.DataFrame(results))
            viz.plot_simulation_results(sim, save_prefix="sim_results")
    
    print("-" * 60)
    print("✓ Simulations complete!")
    print("\nGenerating analysis report...\n")
    
    for handler in log_handlers:
        handler.setLevel(logging.WARNING)
    
    df_results = pd.DataFrame(results)
    
    if return_simulators:
        return (df_results, simulators)
    else:
        return df_results

# ...remaining helper functions (analyze_monte_carlo_results, aggregate_telemetry_from_sims, etc)...

# ============================================================================
# FIX 3: Add missing analyze_monte_carlo_results() function
# ============================================================================

def analyze_monte_carlo_results(df: pd.DataFrame) -> Dict:
    """Analyze Monte Carlo simulation results"""
    print("\n" + "=" * 60)
    print("MONTE CARLO ANALYSIS REPORT")
    print("=" * 60)
    
    metrics_of_interest = ["total_return", "sharpe_ratio", "sortino_ratio", "max_drawdown", "win_rate", "profit_factor"]
    stats = {}
    
    for metric in metrics_of_interest:
        if metric in df.columns:
            values = df[metric].dropna()
            if len(values) > 0:
                stats[metric] = {
                    'mean': values.mean(),
                    'median': values.median(),
                    'std': values.std(),
                    'min': values.min(),
                    'max': values.max(),
                    'p5': values.quantile(0.05),
                    'p95': values.quantile(0.95),
                }
                print(f"\n{metric.upper().replace('_', ' ')}:")
                for k, v in stats[metric].items():
                    print(f"  {k:8s}: {v:+.4f}")
    
    ruin = (df["total_return"] < -0.50).sum()
    ruin_pct = (ruin / len(df) * 100) if len(df) > 0 else 0
    print(f"\n⚠  Risk of Ruin (>50% loss): {ruin_pct:.2f}%  ({ruin}/{len(df)})")
    
    median_sharpe = stats.get("sharpe_ratio", {}).get("median", 0)
    if median_sharpe > 1.5 and ruin_pct == 0:
        print("\n✓✓✓ STRATEGY PASSED – Ready for paper trading")
    elif median_sharpe > 1.0:
        print("\n⚠  STRATEGY MARGINAL – Needs optimisation")
    else:
        print("\n❌ STRATEGY FAILED – Back to drawing board")
    
    print("=" * 60 + "\n")
    return stats

def run_monte_carlo_batch(
    n_simulations: int = 100,
    n_ticks: int = 5000,
    vary_parameters: bool = True,
    base_config: SimulatorConfig = SimulatorConfig(),
    verbose_simulations: int = 5,
    return_simulators: bool = False,
    db: Optional[TelemetryDB] = None,
    show_plots: bool = True,
    plot_every: int = 5,
) -> tuple:
    """Run Monte Carlo simulations with PostgreSQL logging and real-time visualization."""
    results = []
    simulators = []
    
    print(f"Running {n_simulations} Monte Carlo simulations...")
    print(f"Ticks per simulation: {n_ticks}")
    print(f"PostgreSQL telemetry: {'ENABLED' if (db and db.connected) else 'DISABLED (in-memory fallback)'}")
    print(f"Real-time visualizations: {'ENABLED' if show_plots else 'DISABLED'}")
    print("-" * 60)
    
    log_handlers = logging.root.handlers if logging.root.handlers else [logging.StreamHandler()]
    
    for sim_idx in range(n_simulations):
        if sim_idx == verbose_simulations:
            for handler in log_handlers:
                handler.setLevel(logging.WARNING)
            print(f"[Sim {sim_idx}] Switching to silent mode...")
            print("-" * 60)
        
        if (sim_idx + 1) % 10 == 0:
            print(f"Progress: {sim_idx + 1}/{n_simulations}")
        
        # Parameter variation
        if vary_parameters:
            config = SimulatorConfig(
                latency_mean=np.random.uniform(100, 200),
                latency_std=np.random.uniform(20, 40),
                fill_prob_base=0.85,
                fill_prob_decay=0.15,
                slippage_base=np.random.uniform(0.3, 0.7),
                inventory_decay_kappa=np.random.uniform(0.5, 0.8)
            )
        else:
            config = base_config
        
        sim = ExecutionSimulator(config=config, db=db)
        sim.register_simulation(n_ticks)
        
        # Generate market data
        timestamps = pd.date_range("2024-01-01", periods=n_ticks, freq="1min")
        base_price = 30.0
        price_series = [base_price]
        current_regime = RegimeState.NEUTRAL
        regime_duration = 0
        
        for i, ts in enumerate(timestamps):
            # Regime switching
            if regime_duration <= 0:
                regime_probs = [0.3, 0.4, 0.3]
                current_regime = np.random.choice(
                    [RegimeState.BULL, RegimeState.NEUTRAL, RegimeState.BEAR],
                    p=regime_probs
                )
                regime_duration = np.random.randint(50, 200)
            regime_duration -= 1
            
            # Price dynamics
            if current_regime == RegimeState.BULL:
                drift = 0.0003
            elif current_regime == RegimeState.BEAR:
                drift = -0.0003
            else:
                drift = 0
            
            if i > 0:
                noise = np.random.normal(0, 0.01)
                new_price = price_series[-1] + noise + drift
                price_series.append(new_price)
            
            price = price_series[-1]
            vwap = np.mean(price_series[max(0, i-60):i+1])
            sigma = max(np.std(price_series[max(0, i-60):i+1]), 0.01)
            
            market_state = MarketState(
                timestamp=ts,
                price=price,
                bid=price - 0.002,
                ask=price + 0.002,
                volume=np.random.uniform(100, 1000),
                vwap_60=vwap,
                sigma_60=sigma,
                RV_1min=np.random.uniform(0.5, 2.0),
                regime=current_regime,
                P_bull=0.9 if current_regime == RegimeState.BULL else 
                       0.1 if current_regime == RegimeState.BEAR else 0.5,
                hurst=np.random.uniform(0.40, 0.60),
                P_mig=np.random.uniform(0.30, 0.50),
            )
            
            sim.process_tick(market_state)
        
        sim.finalize()
        
        metrics = sim.get_performance_metrics()
        metrics['simulation_id'] = sim_idx
        metrics['latency_mean'] = config.latency_mean
        metrics['fill_prob_base'] = config.fill_prob_base
        metrics['slippage_base'] = config.slippage_base
        metrics['inventory_kappa'] = config.inventory_decay_kappa
        
        results.append(metrics)
        simulators.append(sim)
        
        # Real-time visualization
        if show_plots and ((sim_idx + 1) % plot_every == 0 or sim_idx == 0):
            print(f"\n[Visualization] Plotting simulation #{sim_idx + 1}...")
            viz = SimulationVisualizer(sim_idx, pd.DataFrame(results))
            viz.plot_simulation_results(sim, save_prefix="sim_results")
    
    print("-" * 60)
    print("✓ Simulations complete!")
    print("\nGenerating analysis report...\n")
    
    for handler in log_handlers:
        handler.setLevel(logging.WARNING)
    
    df_results = pd.DataFrame(results)
    
    if return_simulators:
        return (df_results, simulators)
    else:
        return df_results