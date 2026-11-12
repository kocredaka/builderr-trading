"""Momentum sprint bot for builderr Trading Round 2 (forward-return ranking).

Tujuan: memaksimalkan return forward di sisa window pendek (~9 sesi pasar),
dengan rem crash yang keras. BUKAN optimasi Calmar jangka panjang.

Ide inti:
  * Risk-on: ranking momentum cross-sectional di SELURUH ticker yang tersedia
    (live: ~1000 nama; preview: 21 ETF/mega-cap), hanya nama yang uptrend
    (close > SMA50, momentum 20d & 60d positif). Pegang TOP_N teratas.
  * Risk-off: ke CASH (bukan defensif) saat SPY & QQQ kehilangan tren SMA50,
    atau volatilitas QQQ tinggi. Masuk-kembali cepat begitu tren pulih.
  * Rem crash: SPY -4% dalam 5 hari, atau vol QQQ > 55% -> 100% cash.
  * Overlay leverage 2x (QLD/SSO) HANYA saat uptrend tenang, tidak pernah 3x.
  * Rebalance rotasi tiap 3 hari (keluar/de-risking tetap harian, langsung),
    dengan ambang trade minimum untuk menahan churn.

Kepatuhan:
  * Hanya memakai bar harian dari market_state (tidak ada network, LLM,
    API key, file lokal, atau data masa depan). Tidak ada lookahead.
  * Long-only, tiap posisi < 24% target, beta-gross <= 1.40x, <= 45 order.
  * Deterministik (tie-break alfabetis), stdlib saja, decide() < 5 detik
    untuk universe penuh (hanya 131 bar terakhir per ticker yang diproses).
"""
from __future__ import annotations

from math import sqrt
from statistics import mean, pstdev
from typing import Any

# ---------------------------------------------------------------- parameters
# Semua parameter di satu tempat; ubah di sini saja untuk tuning.

TOP_N = 6                 # jumlah posisi momentum teratas saat risk-on
MAX_WEIGHT = 0.24         # cap per ticker (batas kontes: < 30%)
DRIFT_LIMIT = 0.27        # rebalance paksa bila satu posisi melar melewati ini
MAX_BETA_GROSS = 1.40     # cap beta-gross (batas kontes: <= 1.50x)
MIN_TRADE_PCT = 0.015     # abaikan penyesuaian < 1,5% ekuitas (menahan churn)
MAX_ORDERS = 45           # batas kontes: <= 50 trade/hari
ROTATION_EVERY_DAYS = 3   # rotasi risk-on tiap 3 hari (keluar/de-risking: harian)
HOLD_BONUS = 0.0          # bonus skor posisi lama; 0.0 = MATI (teruji memperburuk preview)

# Bobot skor momentum: 0.25*mom20 + 0.55*mom60 + 0.20*trend_gap - 0.15*vol20
# (Bobot condong momentum 60d: teruji lebih baik di 3 window preview dibanding
# bobot condong 20d, dan momentum jangka menengah adalah efek yang paling
# terdokumentasi. Jangan otak-atik tanpa uji ulang di preview.py.)
W_MOM20 = 0.25
W_MOM60 = 0.55
W_TREND = 0.20
W_VOL = 0.15
MAX_NAME_VOL = 0.90       # tolak nama dengan vol 20d tahunan di atas ini

# Regime (indeks acuan: SPY & QQQ, SMA50)
RISK_ON_MAX_VOL = 0.40    # vol QQQ 20d di atas ini -> risk-off
CRASH_VOL = 0.55          # vol QQQ 20d di atas ini -> 100% cash (rem crash)
CRASH_DROP_5D = -0.04     # SPY -4% dalam 5 hari -> 100% cash (rem crash)

# Anggaran notional (maks ~0,98 supaya order selalu terisi penuh dari cash;
# engine tidak memberi margin debt, order di atas cash dipotong).
NOTIONAL_RISK_ON = 0.98
NOTIONAL_HIGH_VOL = 0.80  # risk-on tapi vol 0,28-0,40: tanpa overlay, lebih kecil
NOTIONAL_MIXED = 0.50     # hanya satu indeks di atas SMA50: setengah buku

# Overlay 2x, hanya saat uptrend tenang (QQQ SMA20 > SMA50, mom20 QQQ > 0).
OVERLAY_CALM_VOL = 0.22   # vol < ini -> overlay penuh
OVERLAY_MAX_VOL = 0.28    # vol di atas ini -> overlay mati
OVERLAY_CALM = {"QLD": 0.18, "SSO": 0.10}
OVERLAY_MILD = {"QLD": 0.12, "SSO": 0.06}

LOOKBACK_BARS = 131       # bar terakhir per ticker yang diproses (hemat waktu)
MIN_BARS = 61             # butuh 61 close untuk momentum 60d

# Beta multiples (tabel kontes) — nama leveraged TIDAK masuk ranking momentum.
BETA_MULTIPLE = {
    "TQQQ": 3.0, "SOXL": 3.0, "UPRO": 3.0, "SPXL": 3.0, "TNA": 3.0,
    "FAS": 3.0, "TECL": 3.0, "LABU": 3.0, "CURE": 3.0, "DRN": 3.0,
    "UDOW": 3.0, "NAIL": 3.0,
    "QLD": 2.0, "SSO": 2.0, "DDM": 2.0, "ROM": 2.0, "UWM": 2.0, "AGQ": 2.0,
}

_last_trade_bar_date: str | None = None
_last_rotation_bar_date: str | None = None


# ------------------------------------------------------------------- helpers
def closes(bars: list[dict[str, Any]] | None) -> list[float]:
    """Close terakhir (maks LOOKBACK_BARS), [] bila ada bar rusak."""
    if not bars:
        return []
    out: list[float] = []
    for bar in bars[-LOOKBACK_BARS:]:
        try:
            close = float(bar["close"])
        except (KeyError, TypeError, ValueError):
            return []
        if close <= 0:
            return []
        out.append(close)
    return out


def sma(values: list[float], n: int) -> float | None:
    if len(values) < n:
        return None
    return mean(values[-n:])


def momentum(values: list[float], n: int) -> float | None:
    if len(values) <= n:
        return None
    start = values[-(n + 1)]
    if start <= 0:
        return None
    return values[-1] / start - 1.0


def realized_vol(values: list[float], n: int) -> float | None:
    if len(values) <= n:
        return None
    window = values[-(n + 1):]
    rets = []
    for i in range(1, len(window)):
        prev = window[i - 1]
        if prev <= 0:
            return None
        rets.append(window[i] / prev - 1.0)
    if len(rets) < 5:
        return None
    return pstdev(rets) * sqrt(252.0)


def avg_volume(bars: list[dict[str, Any]] | None, n: int = 20) -> float:
    """Rata-rata volume n bar terakhir; 0 bila data volume tidak tersedia."""
    if not bars:
        return 0.0
    total = 0.0
    count = 0
    for bar in bars[-n:]:
        try:
            total += float(bar.get("volume", 0) or 0)
        except (TypeError, ValueError):
            return 0.0
        count += 1
    return total / count if count else 0.0


def current_positions(portfolio_state: dict[str, Any]) -> dict[str, dict[str, float]]:
    positions: dict[str, dict[str, float]] = {}
    for raw in portfolio_state.get("positions", []) or []:
        ticker = str(raw.get("ticker", "")).upper()
        if not ticker:
            continue
        try:
            qty = float(raw.get("quantity", 0.0))
            avg_cost = float(raw.get("avg_cost", 0.0))
        except (TypeError, ValueError):
            continue
        if qty <= 0:
            continue
        existing = positions.setdefault(ticker, {"quantity": 0.0, "avg_cost": avg_cost})
        existing["quantity"] += qty
        existing["avg_cost"] = avg_cost or existing["avg_cost"]
    return positions


def equity(portfolio_state: dict[str, Any], cash: float) -> float:
    try:
        total = float(portfolio_state.get("cash", cash))
    except (TypeError, ValueError):
        total = float(cash or 0.0)
    last_prices = portfolio_state.get("last_prices", {}) or {}
    for ticker, pos in current_positions(portfolio_state).items():
        try:
            price = float(last_prices.get(ticker, pos["avg_cost"]))
        except (TypeError, ValueError):
            price = pos["avg_cost"]
        total += pos["quantity"] * max(price, 0.0)
    return max(total, 0.0)


def _latest_bar_date(market_state: dict[str, list[dict[str, Any]]]) -> str | None:
    """Tanggal bar terakhir SPY (fallback: QQQ, lalu max semua ticker)."""
    for key in ("SPY", "QQQ"):
        bars = market_state.get(key)
        if bars:
            return str(bars[-1].get("ts", len(bars)))[:10]
    latest: str | None = None
    for bars in market_state.values():
        if bars:
            ts = str(bars[-1].get("ts", ""))[:10]
            if ts and (latest is None or ts > latest):
                latest = ts
    return latest


def _days_since(market_state: dict[str, list[dict[str, Any]]], ref_date: str | None) -> int | None:
    """Jumlah bar sejak ref_date (None bila belum pernah / tanggal tak ditemukan)."""
    if ref_date is None:
        return None
    bars = market_state.get("SPY") or market_state.get("QQQ") or []
    dates = [str(b.get("ts", i))[:10] for i, b in enumerate(bars)]
    if not dates or ref_date not in dates:
        return None
    return len(dates) - dates.index(ref_date) - 1


def _market_prices(market_state: dict[str, list[dict[str, Any]]]) -> dict[str, float]:
    prices: dict[str, float] = {}
    for ticker, bars in market_state.items():
        cs = closes(bars)
        if cs:
            prices[ticker.upper()] = cs[-1]
    return prices


def _scale_caps(weights: dict[str, float]) -> dict[str, float]:
    capped = {t: min(max(w, 0.0), MAX_WEIGHT) for t, w in weights.items() if w > 0.0}
    beta_gross = sum(w * BETA_MULTIPLE.get(t, 1.0) for t, w in capped.items())
    if beta_gross > MAX_BETA_GROSS:
        scale = MAX_BETA_GROSS / beta_gross
        capped = {t: w * scale for t, w in capped.items()}
    return {t: round(w, 6) for t, w in capped.items() if w > 0.001}


# ------------------------------------------------------------- target weights
def target_weights(
    market_state: dict[str, list[dict[str, Any]]],
    held: set[str] | None = None,
) -> dict[str, float]:
    """Target bobot portfolio. {} = 100% cash (risk-off / rem crash).

    `held` = ticker yang sedang dipegang; mereka mendapat HOLD_BONUS supaya
    rotasi tidak churn oleh perubahan ranking kecil dari hari ke hari.
    """
    spy = closes(market_state.get("SPY"))
    qqq = closes(market_state.get("QQQ"))
    if len(spy) < MIN_BARS or len(qqq) < MIN_BARS:
        return {}

    spy_sma50 = sma(spy, 50)
    qqq_sma50 = sma(qqq, 50)
    qqq_vol20 = realized_vol(qqq, 20)
    spy_ret5 = momentum(spy, 5)
    if spy_sma50 is None or qqq_sma50 is None or qqq_vol20 is None or spy_ret5 is None:
        return {}

    # Rem crash: penurunan tajam atau vol ekstrem -> cash penuh, tanpa kecuali.
    if spy_ret5 <= CRASH_DROP_5D or qqq_vol20 >= CRASH_VOL:
        return {}

    spy_up = spy[-1] > spy_sma50
    qqq_up = qqq[-1] > qqq_sma50

    # Kedua indeks di bawah tren, atau vol terlalu tinggi -> cash penuh.
    if (not spy_up and not qqq_up) or qqq_vol20 >= RISK_ON_MAX_VOL:
        return {}

    mixed = spy_up != qqq_up  # hanya satu indeks uptrend: setengah buku

    # --- ranking momentum di semua ticker 1x yang datanya segar & uptrend ---
    latest = _latest_bar_date(market_state)
    scored: list[tuple[float, str]] = []
    for raw_ticker, bars in market_state.items():
        ticker = str(raw_ticker).upper()
        if ticker in BETA_MULTIPLE:
            continue  # leveraged ETF tidak diranking; hanya overlay bergate
        if not bars or str(bars[-1].get("ts", ""))[:10] != latest:
            continue  # bar terakhir basi -> lewati (hindari sinyal beku)
        values = closes(bars)
        if len(values) < MIN_BARS:
            continue
        mom20 = momentum(values, 20)
        mom60 = momentum(values, 60)
        trend50 = sma(values, 50)
        vol20 = realized_vol(values, 20)
        if mom20 is None or mom60 is None or trend50 is None or vol20 is None:
            continue
        if mom20 <= 0 or mom60 <= 0 or values[-1] <= trend50:
            continue  # hanya nama uptrend bermomentum positif
        if vol20 >= MAX_NAME_VOL or values[-1] < 1.0:
            continue  # tolak nama eksplosif / penny
        if avg_volume(bars) <= 0:
            continue  # tidak ada volume -> data tidak bisa dipercaya
        trend_gap = values[-1] / trend50 - 1.0
        score = (W_MOM20 * mom20) + (W_MOM60 * mom60) + (W_TREND * trend_gap) - (W_VOL * vol20)
        if held and ticker in held:
            score += HOLD_BONUS
        if score > 0.0:
            scored.append((score, ticker))

    if not scored:
        return {}
    # Deterministik: skor desc, lalu ticker asc untuk tie-break.
    scored.sort(key=lambda item: (-item[0], item[1]))

    if mixed:
        winners = [ticker for _, ticker in scored[:3]]
        budget = NOTIONAL_MIXED
        overlay: dict[str, float] = {}
    else:
        winners = [ticker for _, ticker in scored[:TOP_N]]
        # Overlay 2x hanya saat uptrend QQQ tenang.
        qqq_sma20 = sma(qqq, 20)
        qqq_mom20 = momentum(qqq, 20)
        overlay_on = bool(
            qqq_sma20 is not None
            and qqq_mom20 is not None
            and qqq_sma20 > qqq_sma50
            and qqq_mom20 > 0.0
            and qqq_vol20 < OVERLAY_MAX_VOL
            and closes(market_state.get("QLD"))
            and closes(market_state.get("SSO"))
        )
        if overlay_on:
            overlay = dict(OVERLAY_CALM if qqq_vol20 < OVERLAY_CALM_VOL else OVERLAY_MILD)
            budget = NOTIONAL_RISK_ON - sum(overlay.values())
        elif qqq_vol20 >= OVERLAY_MAX_VOL:
            overlay = {}
            budget = NOTIONAL_HIGH_VOL
        else:
            overlay = {}
            budget = NOTIONAL_RISK_ON

    weights: dict[str, float] = {}
    per_winner = min(MAX_WEIGHT - 0.02, budget / len(winners))
    for ticker in winners:
        weights[ticker] = per_winner
    weights.update(overlay)
    return _scale_caps(weights)


# -------------------------------------------------------------------- orders
def orders_to_rebalance(
    targets: dict[str, float],
    positions: dict[str, dict[str, float]],
    total_equity: float,
    prices: dict[str, float],
    cash_available: float,
) -> list[dict[str, object]]:
    if total_equity <= 0:
        return []

    min_trade = total_equity * MIN_TRADE_PCT
    orders: list[dict[str, object]] = []
    sell_proceeds = 0.0

    # Jual dulu: buang posisi di luar target & pangkas yang kelebihan bobot.
    for ticker, pos in positions.items():
        price = prices.get(ticker)
        if price is None or price <= 0:
            continue
        qty = pos["quantity"]
        current_value = qty * price
        target_value = total_equity * targets.get(ticker, 0.0)
        delta = target_value - current_value
        if ticker not in targets:
            sell_qty = int(qty)
            if sell_qty > 0 and current_value >= min_trade:
                orders.append({"ticker": ticker, "side": "sell", "quantity": sell_qty})
                sell_proceeds += sell_qty * price
        elif delta < -min_trade:
            sell_qty = min(int(abs(delta) // price), int(qty))
            if sell_qty > 0:
                orders.append({"ticker": ticker, "side": "sell", "quantity": sell_qty})
                sell_proceeds += sell_qty * price

    spendable = max(float(cash_available), 0.0) + (sell_proceeds * 0.98)

    # Beli: pakai cash + hasil jual (diskon 2% untuk slippage), lewati receh.
    for ticker, weight in sorted(targets.items()):
        price = prices.get(ticker)
        if price is None or price <= 0:
            continue
        current_qty = positions.get(ticker, {}).get("quantity", 0.0)
        current_value = current_qty * price
        target_value = total_equity * weight
        delta = target_value - current_value
        if delta < min_trade:
            continue
        buy_value = min(delta, spendable)
        buy_qty = int(buy_value // price)
        if buy_qty > 0:
            orders.append({"ticker": ticker, "side": "buy", "quantity": buy_qty})
            spendable -= buy_qty * price

    return orders[:MAX_ORDERS]


def _has_position_drifted(portfolio_state: dict[str, Any], total_equity: float) -> bool:
    if total_equity <= 0:
        return False
    last_prices = portfolio_state.get("last_prices", {}) or {}
    for ticker, pos in current_positions(portfolio_state).items():
        try:
            price = float(last_prices.get(ticker, pos["avg_cost"]))
        except (TypeError, ValueError):
            price = pos["avg_cost"]
        if price > 0 and (pos["quantity"] * price / total_equity) > DRIFT_LIMIT:
            return True
    return False


# -------------------------------------------------------------------- decide
def decide(
    market_state: dict,
    portfolio_state: dict,
    cash: float,
) -> list[dict]:
    """Return a list of long-only buy/sell orders.

    Keluar/de-risking (target cash atau buku setengah) dieksekusi HARI ITU
    JUGA; rotasi risk-on penuh dibatasi tiap ROTATION_EVERY_DAYS hari untuk
    menahan churn dan lag fill satu hari.
    """
    global _last_trade_bar_date, _last_rotation_bar_date

    if not market_state:
        return []
    # Tanpa bar SPY/QQQ tidak bisa menilai regime -> jangan trading buta.
    if not market_state.get("SPY") or not market_state.get("QQQ"):
        return []

    latest_date = _latest_bar_date(market_state)
    if latest_date is None:
        return []
    # Cegah order ganda bila engine memanggil decide() 2x di tanggal bar sama.
    if latest_date == _last_trade_bar_date:
        return []

    total_equity = equity(portfolio_state, cash)
    positions = current_positions(portfolio_state)
    targets = target_weights(market_state, held=set(positions))
    prices = _market_prices(market_state)

    de_risking = (not targets) or sum(targets.values()) < 0.60
    days_since = _days_since(market_state, _last_rotation_bar_date)
    rotation_due = (
        _last_rotation_bar_date is None
        or days_since is None
        or days_since >= ROTATION_EVERY_DAYS
    )
    drifted = _has_position_drifted(portfolio_state, total_equity)
    if not (de_risking or rotation_due or drifted):
        return []

    orders = orders_to_rebalance(targets, positions, total_equity, prices, cash)
    if orders:
        _last_trade_bar_date = latest_date
        if not de_risking:
            _last_rotation_bar_date = latest_date
    return orders
