from __future__ import annotations

import argparse
import os
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex-matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

from PySide6.QtCore import QThread, Qt, QTimer, Signal, QSize
from PySide6.QtGui import QColor, QFont, QBrush, QIcon, QLinearGradient, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QGraphicsDropShadowEffect,
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
    QFileDialog,
    QDialog,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.dates as mdates
import matplotlib as mpl
from matplotlib.figure import Figure

mpl.rcParams["font.family"] = ["AppleGothic", "DejaVu Sans"]
mpl.rcParams["axes.unicode_minus"] = False

from futures_dashboard.data_service import (
    build_stress_curve,
    cancel_open_orders_for_symbols,
    get_client,
    load_full_snapshot,
    order_distance_stats,
    place_market_order,
    stress_scenario,
)
from risk_report import DEFAULT_BINANCE_CREDENTIALS_PATH, _build_shocks, _resolve_api_credentials

try:
    from binance.exceptions import BinanceAPIException
except Exception:  # pragma: no cover - optional dependency path varies by environment
    BinanceAPIException = None


HISTORY_CSV_PATH = Path(__file__).resolve().parent / "futures_dashboard" / "monitor_history.csv"


def _is_number(value: Any) -> bool:
    """숫자 여부 확인. pd.isna 대신 float() 변환 + NaN!=NaN 으로 재귀 방지."""
    try:
        f = float(value)
        return f == f  # NaN 이면 False (NaN != NaN)
    except (TypeError, ValueError, OverflowError):
        return False


def _fmt_num(value: Any, digits: int = 2, signed: bool = False) -> str:
    if value is None:
        return "-"
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return str(value)
    if number != number:  # NaN check
        return "-"
    if signed:
        return f"{number:+,.{digits}f}"
    return f"{number:,.{digits}f}"


def _fmt_qty(value: Any) -> str:
    return _fmt_num(value, digits=6)


def _fmt_price(value: Any) -> str:
    return _fmt_num(value, digits=4)


def _fmt_pct(value: Any) -> str:
    return _fmt_num(value, digits=2, signed=True) + "%"


def _fmt_balance(value: Any) -> str:
    return _fmt_num(value, digits=0)


def _safe_text(value: Any) -> str:
    if value is None:
        return "-"
    try:
        f = float(value)
        if f != f:  # NaN
            return "-"
    except (TypeError, ValueError, OverflowError):
        pass
    return str(value)


def _make_glyph_icon(glyph: str, start_color: str, end_color: str) -> QIcon:
    SIZE = 48
    pixmap = QPixmap(SIZE, SIZE)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    # 배경 원 (꽉 채움)
    gradient = QLinearGradient(0, 0, SIZE, SIZE)
    gradient.setColorAt(0.0, QColor(start_color))
    gradient.setColorAt(1.0, QColor(end_color))
    painter.setBrush(gradient)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.drawEllipse(0, 0, SIZE, SIZE)
    # 글리프: 크고 굵게
    painter.setPen(QPen(QColor("#ffffff")))
    font = QFont("Apple SD Gothic Neo", 24, QFont.Weight.Black)
    font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 0)
    painter.setFont(font)
    painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, glyph)
    painter.end()
    return QIcon(pixmap)


class NumericItem(QTableWidgetItem):
    def __lt__(self, other: QTableWidgetItem) -> bool:  # type: ignore[override]
        left = self.data(Qt.ItemDataRole.UserRole)
        right = other.data(Qt.ItemDataRole.UserRole)
        if _is_number(left) and _is_number(right):
            return float(left) < float(right)
        return super().__lt__(other)


class ClickableChip(QToolButton):
    def __init__(self, text: str = "") -> None:
        super().__init__()
        self.setText(text)
        self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.setCursor(Qt.CursorShape.PointingHandCursor)


def _make_item(text: str, sort_value: Any = None, *, align_right: bool = True) -> QTableWidgetItem:
    item = NumericItem(text)
    if sort_value is not None:
        item.setData(Qt.ItemDataRole.UserRole, sort_value)
    if align_right:
        item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    else:
        item.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
    if _is_number(sort_value):
        number = float(sort_value)
        if number > 0:
            item.setForeground(QBrush(QColor("#86efac")))
        elif number < 0:
            item.setForeground(QBrush(QColor("#fda4af")))
        else:
            item.setForeground(QBrush(QColor("#e2e8f0")))
    return item


def _df_from_mapping(rows: Iterable[Dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(list(rows or []))


def _collateral_dataframe(collateral: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    rows = []
    for asset, info in collateral.items():
        rows.append(
            {
                "asset": asset,
                "wallet_balance": info.get("wallet_balance", 0.0),
                "current_price": info.get("current_price", 0.0),
                "total_value": info.get("total_value", 0.0),
                "unrealized_pnl": info.get("unrealized_pnl", 0.0),
            }
        )
    return pd.DataFrame(rows)


def _summary_row_for_positions(df: pd.DataFrame) -> Dict[str, Any]:
    if df is None or df.empty:
        return {}
    return {
        "No": "합계",
        "symbol": "합계",
        "position": pd.to_numeric(df.get("position", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum(),
        "entry": pd.NA,
        "mark": pd.NA,
        "notional": pd.to_numeric(df.get("notional", pd.Series(dtype=float)), errors="coerce").fillna(0.0).abs().sum(),
        "unrealized_pnl": pd.to_numeric(df.get("unrealized_pnl", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum(),
        "realized_7d": pd.to_numeric(df.get("realized_7d", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum(),
        "liq_price": pd.NA,
        "leverage": pd.NA,
        "margin_type": "",
    }


def _summary_row_for_open_orders(df: pd.DataFrame) -> Dict[str, Any]:
    if df is None or df.empty:
        return {}
    return {
        "symbol": "합계",
        "side": "",
        "type": "",
        "qty": pd.to_numeric(df.get("qty", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum(),
        "price": pd.NA,
        "stop": pd.NA,
        "mark": pd.NA,
        "est_value_usdt": pd.to_numeric(df.get("est_value_usdt", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum(),
        "time_local": "",
    }


def _summary_row_for_collateral(df: pd.DataFrame) -> Dict[str, Any]:
    if df is None or df.empty:
        return {}
    return {
        "asset": "합계",
        "wallet_balance": pd.to_numeric(df.get("wallet_balance", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum(),
        "current_price": pd.NA,
        "total_value": pd.to_numeric(df.get("total_value", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum(),
        "unrealized_pnl": pd.to_numeric(df.get("unrealized_pnl", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum(),
    }


def _summary_row_for_trades(df: pd.DataFrame) -> Dict[str, Any]:
    if df is None or df.empty:
        return {}
    return {
        "symbol": "합계",
        "side": "",
        "qty": pd.to_numeric(df.get("qty", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum(),
        "price": pd.NA,
        "mark": pd.NA,
        "value_usdt": pd.to_numeric(df.get("value_usdt", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum(),
        "realized_pnl": pd.to_numeric(df.get("realized_pnl", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum(),
        "time_local": "",
    }


def _history_columns() -> List[str]:
    return [
        "timestamp",
        "total_wallet",
        "total_margin_balance",
        "available_balance",
        "total_unrealized",
        "maint_buffer",
        "margin_ratio_pct",
        "positions_count",
        "open_orders_count",
        "realized_7d",
    ]


def _load_history_df() -> pd.DataFrame:
    path = HISTORY_CSV_PATH
    if not path.is_file():
        return pd.DataFrame(columns=_history_columns())
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame(columns=_history_columns())
    for col in _history_columns():
        if col not in df.columns:
            df[col] = pd.NA
    df = df[_history_columns()]
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    for c in _history_columns()[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    return df


def _append_history_row(row: Dict[str, Any]) -> None:
    path = HISTORY_CSV_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    base = _load_history_df()
    new_row = {col: row.get(col, pd.NA) for col in _history_columns()}
    if pd.notna(new_row.get("timestamp")):
        new_row["timestamp"] = pd.to_datetime(new_row["timestamp"], errors="coerce")
    new_df = pd.DataFrame([new_row])
    out = pd.concat([base, new_df], ignore_index=True)
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    for c in _history_columns()[1:]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp")
    if len(out) > 2000:
        out = out.iloc[-2000:].copy()
    out.to_csv(path, index=False)


def _rollup_small_slices(labels: Sequence[str], values: Sequence[float], max_small_pct: float = 3.0) -> Tuple[List[str], List[float]]:
    vals = [max(0.0, float(v)) for v in values]
    total = sum(vals)
    if total <= 1e-12:
        return ["없음"], [1.0]
    kept_labels: List[str] = []
    kept_values: List[float] = []
    other = 0.0
    for lab, val in zip(labels, vals):
        pct = (val / total) * 100.0
        if val > 0 and pct <= max_small_pct:
            other += val
        else:
            kept_labels.append(str(lab))
            kept_values.append(val)
    if other > 1e-12:
        kept_labels.append("기타")
        kept_values.append(other)
    if not kept_labels:
        return ["기타"], [total]
    return kept_labels, kept_values


def _pie_colors(n: int) -> List[str]:
    palette = [
        "#38bdf8",
        "#818cf8",
        "#22c55e",
        "#f59e0b",
        "#f472b6",
        "#a78bfa",
        "#14b8a6",
        "#fb7185",
        "#eab308",
        "#60a5fa",
    ]
    if n <= len(palette):
        return palette[:n]
    out: List[str] = []
    for i in range(n):
        out.append(palette[i % len(palette)])
    return out


def _draw_pie_chart(ax, title: str, labels: Sequence[str], values: Sequence[float], unit: str = "USDT") -> None:
    ax.clear()
    ax.set_facecolor("#0f172a")
    ax.figure.patch.set_facecolor("#0f172a")

    clean_labels, clean_values = _rollup_small_slices(labels, values)
    total = sum(max(0.0, float(v)) for v in clean_values)
    if total <= 1e-12 or clean_labels == ["없음"]:
        ax.text(0.5, 0.5, "데이터 없음", ha="center", va="center", color="#cbd5e1", fontsize=13, fontweight="bold")
        ax.set_axis_off()
        ax.set_title(title, color="#e2e8f0", fontsize=13, pad=12, fontweight="bold")
        return

    colors = _pie_colors(len(clean_labels))
    wedges, _texts, _autotexts = ax.pie(
        clean_values,
        labels=None,
        colors=colors,
        startangle=90,
        counterclock=False,
        autopct=lambda pct: f"{pct:.1f}%" if pct >= 4 else "",
        pctdistance=0.78,
        wedgeprops=dict(width=0.44, edgecolor="#0f172a", linewidth=1.2),
        textprops=dict(color="#e2e8f0", fontsize=10),
    )
    ax.legend(
        wedges,
        [f"{lab} ({val:,.2f} {unit})" for lab, val in zip(clean_labels, clean_values)],
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=9,
        labelcolor="#cbd5e1",
    )
    ax.set_title(title, color="#e2e8f0", fontsize=13, pad=12, fontweight="bold")
    ax.set_aspect("equal")


class PieChartPanel(QWidget):
    def __init__(self, title: str, subtitle: str = "") -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self.title_label = QLabel(title)
        self.title_label.setObjectName("SectionTitle")
        self.title_label.setWordWrap(True)
        self.subtitle_label = QLabel(subtitle)
        self.subtitle_label.setObjectName("SectionHint")
        self.subtitle_label.setWordWrap(True)
        self.figure = Figure(figsize=(4.2, 3.6), dpi=100)
        self.figure.patch.set_facecolor("#0f172a")
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumHeight(320)

        layout.addWidget(self.title_label)
        layout.addWidget(self.subtitle_label)
        layout.addWidget(self.canvas, 1)
        self.top5_container = QWidget()
        self.top5_layout = QGridLayout(self.top5_container)
        self.top5_layout.setContentsMargins(0, 0, 0, 0)
        self.top5_layout.setHorizontalSpacing(8)
        self.top5_layout.setVerticalSpacing(8)
        layout.addWidget(self.top5_container)

    def update_chart(self, title: str, labels: Sequence[str], values: Sequence[float], unit: str = "USDT") -> None:
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        _draw_pie_chart(ax, title, labels, values, unit=unit)
        self.canvas.draw_idle()

    def update_top5_cards(self, rows: Sequence[Tuple[str, float]], *, title: str = "Top 5", unit: str = "USDT", total: Optional[float] = None) -> None:
        while self.top5_layout.count():
            item = self.top5_layout.takeAt(0)
            if item and item.widget():
                item.widget().deleteLater()

        if not rows:
            label = QLabel("No data")
            label.setObjectName("TopEmpty")
            self.top5_layout.addWidget(label, 0, 0)
            return

        header = QLabel(title)
        header.setObjectName("TopHeader")
        header_text = title if total is None else f"{title} · Total {total:,.2f} {unit}"
        header.setText(header_text)
        self.top5_layout.addWidget(header, 0, 0, 1, 2)

        for idx, (label, value) in enumerate(rows[:4], 1):
            card = QFrame()
            card.setObjectName("TopCard")
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(10, 8, 10, 8)
            card_layout.setSpacing(4)
            rank = QLabel(f"#{idx}")
            rank.setObjectName("TopCardRank")
            name = QLabel(str(label))
            name.setObjectName("TopCardName")
            name.setWordWrap(True)
            amount = QLabel(f"{value:,.2f} {unit}")
            amount.setObjectName("TopCardValue")
            card_layout.addWidget(rank)
            card_layout.addWidget(name)
            card_layout.addWidget(amount)
            row = 1 + (idx - 1) // 2
            col = (idx - 1) % 2
            self.top5_layout.addWidget(card, row, col)


class HistoryPanel(QWidget):
    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self.title_label = QLabel("History")
        self.title_label.setObjectName("SectionTitle")
        self.subtitle_label = QLabel("잔고 및 미실현수익 추이를 로컬에 저장하여 표시합니다.")
        self.subtitle_label.setObjectName("SectionHint")
        self.subtitle_label.setWordWrap(True)
        self.figure = Figure(figsize=(10, 5.8), dpi=100)
        self.figure.patch.set_facecolor("#0f172a")
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumHeight(430)

        layout.addWidget(self.title_label)
        layout.addWidget(self.subtitle_label)
        layout.addWidget(self.canvas, 1)

    def update_history(self, history_df: pd.DataFrame) -> None:
        self.figure.clear()
        if history_df is None or history_df.empty:
            ax = self.figure.add_subplot(111)
            ax.set_facecolor("#0f172a")
            ax.text(0.5, 0.5, "No saved history yet", ha="center", va="center", color="#cbd5e1", fontsize=13, fontweight="bold")
            ax.set_axis_off()
            self.canvas.draw_idle()
            return

        import matplotlib.patheffects as _pe
        import numpy as _np
        from matplotlib.colors import LinearSegmentedColormap as _LSC

        df = history_df.copy().sort_values("timestamp")
        ts = pd.to_datetime(df["timestamp"], errors="coerce")
        wallet = pd.to_numeric(df["total_wallet"], errors="coerce")
        upl    = pd.to_numeric(df["total_unrealized"], errors="coerce")

        # ── Crypto8 APY 차트 디자인 토큰 ───────────────────────
        BG_FIG  = "#0b101d"   # figure 배경 (rgba(7,16,29))
        BG_AX   = "#101b2e"   # axes 배경  (rgba(16,27,46))
        COL_WALLET = "#6b8cff"  # SERIES_COLORS[0]
        COL_UPL    = "#ff5c5c"  # COL_BLEND
        COL_POS    = "#47d9a8"  # SERIES_COLORS[2]
        COL_GRID   = "#7c96b3"  # rgba(124,150,179)
        COL_TICK   = "#8aaac4"  # rgba(212,225,240,0.64) 근사
        COL_SPINE  = "#7c96b3"

        self.figure.patch.set_facecolor(BG_FIG)

        # ── 이중 Y축 ────────────────────────────────────────────
        ax1 = self.figure.add_subplot(111)
        ax2 = ax1.twinx()

        # 배경
        ax1.set_facecolor(BG_AX)

        # 스파인: 아래/왼쪽만 연하게, 나머지 숨김
        for side, ax in [("left", ax1), ("right", ax2)]:
            for sp_name, sp in ax.spines.items():
                if sp_name == "bottom":
                    sp.set_color(COL_SPINE); sp.set_alpha(0.45); sp.set_linewidth(1.15)
                elif sp_name == side:
                    sp.set_color(COL_SPINE); sp.set_alpha(0.45); sp.set_linewidth(1.15)
                else:
                    sp.set_visible(False)

        # 그리드: 매우 연한 수평선만
        ax1.yaxis.grid(True, color=COL_GRID, alpha=0.16, linewidth=1.0, linestyle="-")
        ax1.xaxis.grid(True, color=COL_GRID, alpha=0.08, linewidth=0.8, linestyle="-")
        ax1.set_axisbelow(True)

        # 틱 색상
        ax1.tick_params(axis="x", colors=COL_TICK, labelsize=8.5, length=3, width=0.8)
        ax1.tick_params(axis="y", labelcolor=COL_WALLET, labelsize=8.5, length=3, width=0.8)
        ax2.tick_params(axis="y", labelcolor=COL_UPL,    labelsize=8.5, length=3, width=0.8)

        # ── 왼쪽 Y축: 잔고 ─────────────────────────────────────
        # 그라디언트 fill (0%→0.18 alpha, 78%→0.02 alpha)
        if wallet.notna().any():
            w_vals = wallet.ffill().fillna(0)
            w_min  = float(w_vals.min())
            ax1.fill_between(
                ts, w_vals, w_min,
                color=COL_WALLET, alpha=0.0, zorder=2,
            )
            # 폴리곤으로 수직 그라디언트 구현
            poly = ax1.fill_between(ts, w_vals, w_min, color=COL_WALLET, alpha=0.18, zorder=2)
            # 글로우 + 메인 라인
            ln1, = ax1.plot(ts, wallet, color=COL_WALLET, linewidth=2.2, zorder=4,
                            solid_capstyle="round", solid_joinstyle="round", label="잔고")
            ln1.set_path_effects([
                _pe.SimpleLineShadow(shadow_color=COL_WALLET, alpha=0.35, rho=0.6, linewidth=5),
                _pe.Normal(),
            ])
            # 최신값 태그
            last_w  = wallet.dropna().iloc[-1]
            last_ts = ts[wallet.notna()].iloc[-1]
            ax1.annotate(
                f" {last_w:,.0f} USDT",
                xy=(last_ts, last_w), fontsize=8.5,
                color=COL_WALLET, fontweight="bold",
                xytext=(5, 2), textcoords="offset points",
            )
        else:
            ln1, = ax1.plot([], [], color=COL_WALLET, linewidth=2.2, label="잔고")

        ax1.set_ylabel("잔고 (USDT)", color=COL_WALLET, fontsize=9.5, labelpad=6)

        # ── 오른쪽 Y축: 미실현수익 ─────────────────────────────
        if upl.notna().any():
            u_vals  = upl.ffill().fillna(0)
            pos_upl = u_vals.clip(lower=0)
            neg_upl = u_vals.clip(upper=0)
            ax2.fill_between(ts, pos_upl, 0, color=COL_POS,    alpha=0.22, zorder=2)
            ax2.fill_between(ts, neg_upl, 0, color=COL_UPL,    alpha=0.18, zorder=2)
            ax2.axhline(0, color=COL_GRID, linewidth=0.9, linestyle="--",
                        alpha=0.55, zorder=3)
            ln2, = ax2.plot(ts, upl, color=COL_UPL, linewidth=2.0, zorder=4,
                            solid_capstyle="round", solid_joinstyle="round", label="미실현수익")
            ln2.set_path_effects([
                _pe.SimpleLineShadow(shadow_color=COL_UPL, alpha=0.30, rho=0.6, linewidth=5),
                _pe.Normal(),
            ])
            last_upl = upl.dropna().iloc[-1]
            last_ts2 = ts[upl.notna()].iloc[-1]
            tag_color = COL_POS if last_upl >= 0 else COL_UPL
            sign = "+" if last_upl >= 0 else ""
            ax2.annotate(
                f" {sign}{last_upl:,.0f} USDT",
                xy=(last_ts2, last_upl), fontsize=8.5,
                color=tag_color, fontweight="bold",
                xytext=(5, -12), textcoords="offset points",
            )
        else:
            ln2, = ax2.plot([], [], color=COL_UPL, linewidth=2.0, label="미실현수익")

        ax2.set_ylabel("미실현수익 (USDT)", color=COL_UPL, fontsize=9.5, labelpad=6)

        # ── 범례 (pill 스타일) ──────────────────────────────────
        leg = ax1.legend(
            [ln1, ln2], ["잔고", "미실현수익"],
            loc="upper left", frameon=True, fontsize=8.5,
            labelcolor=[COL_WALLET, COL_UPL],
            facecolor="none",
            edgecolor="none",
        )
        leg.get_frame().set_alpha(0)

        # ── 제목 ────────────────────────────────────────────────
        ax1.set_title("잔고 / 미실현수익", color="#f5f9ff", fontsize=11,
                      pad=10, fontweight="heavy", loc="left")

        # ── X축 포맷 ────────────────────────────────────────────
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
        ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
        self.figure.autofmt_xdate(rotation=30, ha="right")
        self.figure.tight_layout(pad=1.4)
        self.canvas.draw_idle()




def _format_table_value(column: str, value: Any) -> Tuple[str, Any]:
    if column in {"position", "qty"}:
        return _fmt_qty(value), value
    if column in {"entry", "mark", "liq_price", "price", "stop", "limit"}:
        return _fmt_price(value), value
    if column in {"notional", "unrealized_pnl", "realized_7d", "est_value_usdt", "value_usdt", "wallet_balance", "current_price", "total_value", "upl_asset", "equity_proxy", "total_unrealized_pnl", "collateral_usdt_equiv", "total_position_value"}:
        return _fmt_num(value, 2), value
    if column in {"dist_pct", "margin_ratio_pct", "loss_to_collateral_pct"}:
        return _fmt_pct(value), value
    if column in {"leverage"}:
        return _fmt_num(value, 1), value
    return _safe_text(value), None


def _populate_table(
    table: QTableWidget,
    df: pd.DataFrame,
    *,
    empty_text: str = "데이터 없음",
    summary_row: Optional[Dict[str, Any]] = None,
) -> None:
    sorting_enabled = table.isSortingEnabled()
    table.setSortingEnabled(False)
    table.clear()
    if df is None or df.empty:
        table.setRowCount(1)
        table.setColumnCount(1)
        table.setHorizontalHeaderLabels(["status"])
        table.setItem(0, 0, _make_item(empty_text, align_right=False))
        table.horizontalHeader().setStretchLastSection(True)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        table.setSortingEnabled(sorting_enabled)
        return

    columns = list(df.columns)
    rows_to_render: List[pd.Series] = []
    if summary_row is not None:
        summary = {col: summary_row.get(col, pd.NA) for col in columns}
        rows_to_render.append(pd.Series(summary))
    rows_to_render.extend([row for _, row in df.iterrows()])
    table.setColumnCount(len(columns))
    table.setRowCount(len(rows_to_render))
    table.setHorizontalHeaderLabels(columns)
    for r, row in enumerate(rows_to_render):
        # 체결 내역에서 SELL 행 감지
        row_side = str(row.get("side", "")).strip().upper() if "side" in row.index else ""
        is_sell_row = (row_side == "SELL") and (summary_row is None or r > 0)

        for c, col in enumerate(columns):
            text, sort_value = _format_table_value(col, row[col])
            item = _make_item(text, sort_value, align_right=False if col in {"symbol", "side", "type", "asset", "time_local", "margin_type", "risk_label"} else True)
            if col in {"side", "type", "risk_label"}:
                item.setFont(QFont("Apple SD Gothic Neo", 10, QFont.Weight.DemiBold))
            if col in {"risk_label"}:
                item.setForeground(QBrush(QColor("#fbbf24")))
            if r == 0 and summary_row is not None:
                font = item.font()
                font.setBold(True)
                item.setFont(font)
                item.setBackground(QBrush(QColor("#111827")))
            elif is_sell_row:
                # SELL 행: 포지션 마이너스와 동일한 붉은 글자색 (#fda4af), 배경은 그대로
                item.setForeground(QBrush(QColor("#fda4af")))
            table.setItem(r, c, item)
    table.horizontalHeader().setStretchLastSection(True)
    table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
    table.verticalHeader().setVisible(False)
    table.setSortingEnabled(sorting_enabled)


class SnapshotWorker(QThread):
    loaded = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        key_file: str,
        shock_min: float,
        shock_max: float,
        shock_step: float,
        preset_shock: float,
    ) -> None:
        super().__init__()
        self._api_key = api_key
        self._api_secret = api_secret
        self._key_file = key_file
        self._shock_min = shock_min
        self._shock_max = shock_max
        self._shock_step = shock_step
        self._preset_shock = preset_shock

    def run(self) -> None:
        try:
            args = argparse.Namespace(
                api_key=self._api_key,
                api_secret=self._api_secret,
                binance_key_file=self._key_file,
            )
            api_key, api_secret = _resolve_api_credentials(args)
            client = get_client(api_key, api_secret)
            snapshot = load_full_snapshot(client)

            positions = snapshot["positions"]
            open_orders = snapshot["open_orders"]
            collateral = snapshot["collateral"]
            marks = snapshot["mark_prices"]
            shocks = _build_shocks(self._shock_min, self._shock_max, self._shock_step)
            stress_df = build_stress_curve(shocks, positions, open_orders, marks, collateral, simulate_fills=False)
            dist_df = order_distance_stats(open_orders, marks)
            preset_sc = stress_scenario(self._preset_shock, positions, open_orders, marks, collateral, simulate_fills=False)

            payload = {
                "snapshot": snapshot,
                "stress_df": stress_df,
                "dist_df": dist_df,
                "preset_sc": preset_sc,
                "shocks": shocks,
            }
            self.loaded.emit(payload)
        except Exception as exc:
            self.failed.emit(self._format_error(exc))

    def _format_error(self, exc: Exception) -> str:
        text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        if BinanceAPIException is not None and isinstance(exc, BinanceAPIException):
            code = getattr(exc, "code", None)
            if code == -1022 or "Signature for this request is not valid" in str(exc):
                return (
                    "바이낸스 인증 서명이 거절되었습니다.\n\n"
                    "점검할 것:\n"
                    "1. API Key와 Secret이 한 쌍인지 확인\n"
                    "2. Futures 권한이 켜져 있는지 확인\n"
                    "3. IP 화이트리스트를 쓰는 경우 현재 PC IP가 허용되는지 확인\n"
                    "4. 키 파일에 다른 줄/공백/옛 값이 섞여 있지 않은지 확인\n"
                    "5. 환경변수 BINANCE_API_KEY / BINANCE_API_SECRET 이 파일 값보다 우선 적용되지 않는지 확인\n\n"
                    f"원본 오류: {exc}"
                )
        if "Signature for this request is not valid" in text:
            return (
                "바이낸스 인증 서명이 거절되었습니다.\n\n"
                "이 오류는 보통 API Key/Secret 불일치, Futures 권한 미설정, IP 제한, "
                "또는 잘못된 키 파일 경로에서 발생합니다.\n\n"
                f"원본 오류: {exc}"
            )
        return text


@dataclass
class PlannedOrder:
    symbol: str
    side: str
    quantity: float
    reduce_only: bool = False


class OrderWorker(QThread):
    loaded = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        key_file: str,
        cancel_symbols: List[str],
        orders: List[PlannedOrder],
        label: str,
    ) -> None:
        super().__init__()
        self._api_key = api_key
        self._api_secret = api_secret
        self._key_file = key_file
        self._cancel_symbols = cancel_symbols
        self._orders = orders
        self._label = label

    def run(self) -> None:
        try:
            args = argparse.Namespace(
                api_key=self._api_key,
                api_secret=self._api_secret,
                binance_key_file=self._key_file,
            )
            api_key, api_secret = _resolve_api_credentials(args)
            client = get_client(api_key, api_secret)
            cancel_results = cancel_open_orders_for_symbols(client, self._cancel_symbols)
            order_results: List[Dict[str, Any]] = []
            for order in self._orders:
                response = place_market_order(
                    client,
                    order.symbol,
                    order.side,
                    order.quantity,
                    reduce_only=order.reduce_only,
                )
                order_results.append(
                    {
                        "symbol": order.symbol,
                        "side": order.side,
                        "quantity": order.quantity,
                        "reduce_only": order.reduce_only,
                        "response": response,
                    }
                )
            self.loaded.emit(
                {
                    "label": self._label,
                    "cancel_results": cancel_results,
                    "order_results": order_results,
                }
            )
        except Exception as exc:
            self.failed.emit("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))


class MetricCard(QFrame):
    def __init__(self, title: str, value: str, subtitle: str = "", *, accent: str = "#818cf8") -> None:
        super().__init__()
        self.setObjectName("MetricCard")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(6)

        title_lbl = QLabel(title)
        title_lbl.setObjectName("MetricTitle")
        value_lbl = QLabel(value)
        value_lbl.setObjectName("MetricValue")
        subtitle_lbl = QLabel(subtitle or "")
        subtitle_lbl.setObjectName("MetricSubtitle")
        subtitle_lbl.setWordWrap(True)

        accent_bar = QFrame()
        accent_bar.setFixedHeight(3)
        accent_bar.setStyleSheet(f"background: {accent}; border-radius: 1px;")

        layout.addWidget(title_lbl)
        layout.addWidget(value_lbl)
        layout.addWidget(subtitle_lbl)
        layout.addStretch(1)
        layout.addWidget(accent_bar)

        self._value_label = value_lbl
        self._subtitle_label = subtitle_lbl

    def set_values(self, value: str, subtitle: str = "") -> None:
        self._value_label.setText(value)
        self._subtitle_label.setText(subtitle)




class BtcPriceWorker(QThread):
    """BTC 마크 가격을 3초마다 실시간으로 가져오는 전용 워커."""
    price_updated = Signal(float)

    def __init__(self, api_key: str, api_secret: str, key_file: str) -> None:
        super().__init__()
        self._api_key = api_key
        self._api_secret = api_secret
        self._key_file = key_file
        self._running = True

    def stop(self) -> None:
        self._running = False

    def run(self) -> None:
        import time
        client = None
        while self._running:
            try:
                if client is None:
                    args = argparse.Namespace(
                        api_key=self._api_key,
                        api_secret=self._api_secret,
                        binance_key_file=self._key_file,
                    )
                    api_key, api_secret = _resolve_api_credentials(args)
                    client = get_client(api_key, api_secret)
                result = client.futures_mark_price(symbol="BTCUSDT")
                price = float(result.get("markPrice", 0))
                if price > 0:
                    self.price_updated.emit(price)
            except Exception:
                client = None  # 오류 시 클라이언트 재생성
            time.sleep(3)

class PositionsOverviewBar(QFrame):
    """Crypto8-스타일 Positions Overview 히어로 바 — 화면 최상단에 표시."""

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("PositionsOverviewBar")

        outer = QHBoxLayout(self)
        outer.setContentsMargins(24, 18, 24, 18)
        outer.setSpacing(24)

        # ── 왼쪽: eyebrow + 총 노셔널 ──────────────────────────
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(4)

        eyebrow = QLabel("POSITIONS OVERVIEW")
        eyebrow.setObjectName("OverviewEyebrow")

        self._total_label = QLabel("$0")
        self._total_label.setObjectName("OverviewTotal")

        subtitle = QLabel("포지션 노출 총액 · 손익 현황")
        subtitle.setObjectName("OverviewSubtitle")

        left_layout.addWidget(eyebrow)
        left_layout.addWidget(self._total_label)
        left_layout.addWidget(subtitle)
        outer.addWidget(left)
        outer.addStretch(1)

        # ── 오른쪽: 지표 칩들 ──────────────────────────────────
        chips_widget = QWidget()
        chips_layout = QHBoxLayout(chips_widget)
        chips_layout.setContentsMargins(0, 0, 0, 0)
        chips_layout.setSpacing(10)

        self._chips: Dict[str, QLabel] = {}
        self._orders_amt_label: Optional[QLabel] = None
        chip_defs = [
            ("wallet",    "총 잔고",     "—"),
            ("positions", "포지션 수",   "—"),
            ("orders",    "미체결",      "—"),
            ("upl",       "미실현 PnL",  "—"),
            ("margin",    "마진 비율",   "—"),
        ]
        for key, label_text, default in chip_defs:
            chip = QFrame()
            chip.setObjectName("OverviewChip")
            chip_inner = QVBoxLayout(chip)
            chip_inner.setContentsMargins(16, 10, 16, 10)
            chip_inner.setSpacing(3)
            lbl_title = QLabel(label_text)
            lbl_title.setObjectName("OverviewChipTitle")
            lbl_value = QLabel(default)
            lbl_value.setObjectName("OverviewChipValue")
            chip_inner.addWidget(lbl_title)
            chip_inner.addWidget(lbl_value)
            # 미체결 칩에는 잔액 줄 추가
            if key == "orders":
                lbl_amt = QLabel("—")
                lbl_amt.setObjectName("OverviewChipSub")
                chip_inner.addWidget(lbl_amt)
                self._orders_amt_label = lbl_amt
            chips_layout.addWidget(chip)
            self._chips[key] = lbl_value

        outer.addWidget(chips_widget)

    def update_data(
        self,
        total_wallet: float,
        total_notional: float,
        positions_count: int,
        orders_count: int,
        orders_amount: float,
        unrealized_pnl: float,
        margin_ratio_pct: float,
    ) -> None:
        self._total_label.setText(f"${total_notional:,.0f}")
        self._chips["wallet"].setText(f"{total_wallet:,.0f} USDT")
        self._chips["positions"].setText(f"{positions_count}건")
        self._chips["orders"].setText(f"{orders_count}건")
        if self._orders_amt_label is not None:
            self._orders_amt_label.setText(f"${orders_amount:,.0f}")
        sign = "+" if unrealized_pnl >= 0 else ""
        self._chips["upl"].setText(f"{sign}{unrealized_pnl:,.0f} USDT")
        if unrealized_pnl > 0:
            self._chips["upl"].setStyleSheet(
                "color: #86efac; font-size: 15px; font-weight: 700;"
            )
        elif unrealized_pnl < 0:
            self._chips["upl"].setStyleSheet(
                "color: #fda4af; font-size: 15px; font-weight: 700;"
            )
        else:
            self._chips["upl"].setStyleSheet(
                "color: #e2e8f0; font-size: 15px; font-weight: 700;"
            )
        self._chips["margin"].setText(f"{margin_ratio_pct:.2f}%")

class FuturesMonitor(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Binance USD-M Futures Monitor")
        # 창 크기: main()에서 showMaximized() 호출 (show() 보다 나중에 적용)
        screen = QApplication.primaryScreen()
        if screen is not None:
            ag = screen.availableGeometry()
            self.setGeometry(ag)

        self._worker: Optional[SnapshotWorker] = None
        self._order_worker: Optional[OrderWorker] = None
        self._btc_worker: Optional[BtcPriceWorker] = None
        self._loading = False
        self._latest_payload: Optional[Dict[str, Any]] = None
        self._status_message = ""
        self._history_df = _load_history_df()
        self._order_auto_baselines: Dict[str, Dict[str, float]] = {"bulk": {}, "single": {}}

        self._build_ui()
        self._apply_theme()
        self._load_defaults()
        self._setup_timer()
        self.refresh()

    def _build_ui(self) -> None:
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(18, 18, 18, 18)
        root.setSpacing(14)

        header_row = QHBoxLayout()
        header_row.setSpacing(10)
        title = QLabel("바이낸스 USD-M 선물 모니터")
        title.setObjectName("AppTitle")
        header_row.addWidget(title)
        header_row.addStretch(1)

        self.settings_toggle_btn = QToolButton()
        self.settings_toggle_btn.setObjectName("HeaderIconSettings")
        self.settings_toggle_btn.setToolTip("조회 설정")
        self.settings_toggle_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        self.settings_toggle_btn.clicked.connect(self._toggle_settings_panel)

        self.order_toggle_btn = QToolButton()
        self.order_toggle_btn.setObjectName("HeaderIconOrder")
        self.order_toggle_btn.setToolTip("주문 모드")
        self.order_toggle_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        self.order_toggle_btn.clicked.connect(self._show_order_dialog)

        self.metrics_toggle_btn_icon = QToolButton()
        self.metrics_toggle_btn_icon.setObjectName("HeaderIconMetrics")
        self.metrics_toggle_btn_icon.setToolTip("핵심 지표 보기/숨기기")
        self.metrics_toggle_btn_icon.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        self.metrics_toggle_btn_icon.clicked.connect(self._toggle_metrics_panel)

        self.refresh_icon_btn = QToolButton()
        self.refresh_icon_btn.setObjectName("HeaderIconRefresh")
        self.refresh_icon_btn.setToolTip("새로고침")
        self.refresh_icon_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        self.refresh_icon_btn.clicked.connect(self.refresh)

        self._decorate_header_button(self.settings_toggle_btn,    _make_glyph_icon("⚙", "#5b8cff", "#7be8c3"))
        self._decorate_header_button(self.order_toggle_btn,       _make_glyph_icon("⊕", "#f59e0b", "#fb923c"))
        self._decorate_header_button(self.metrics_toggle_btn_icon, _make_glyph_icon("⊞", "#818cf8", "#5b8cff"))
        self._decorate_header_button(self.refresh_icon_btn,       _make_glyph_icon("↻", "#7be8c3", "#22c55e"))

        # BTC 가격 표시 (아이콘 앞)
        self.btc_price_label = QLabel("BTC  —")
        self.btc_price_label.setObjectName("BtcPriceLabel")
        header_row.addWidget(self.btc_price_label)
        header_row.addSpacing(6)
        header_row.addWidget(self.settings_toggle_btn)
        header_row.addWidget(self.order_toggle_btn)
        header_row.addWidget(self.metrics_toggle_btn_icon)
        header_row.addWidget(self.refresh_icon_btn)
        root.addLayout(header_row)

        # ── Positions Overview 히어로 바 ──────────────────────
        self.overview_bar = PositionsOverviewBar()
        root.addWidget(self.overview_bar)

        self.controls_group = QGroupBox("조회 설정")
        controls_layout = QGridLayout(self.controls_group)
        controls_layout.setHorizontalSpacing(10)
        controls_layout.setVerticalSpacing(8)

        self.api_key_edit = QLineEdit()
        self.api_key_edit.setPlaceholderText("BINANCE_API_KEY 또는 직접 입력")
        self.api_secret_edit = QLineEdit()
        self.api_secret_edit.setPlaceholderText("BINANCE_API_SECRET 또는 직접 입력")
        self.api_secret_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.key_file_path = str(DEFAULT_BINANCE_CREDENTIALS_PATH)

        self.auto_refresh_check = QCheckBox("자동 새로고침")
        self.auto_refresh_check.setChecked(True)
        self.interval_spin = QSpinBox()
        self.interval_spin.setRange(5, 600)
        self.interval_spin.setValue(30)
        self.interval_spin.setSuffix(" 초")

        self.shock_min_spin = QDoubleSpinBox()
        self.shock_min_spin.setRange(-99.0, 99.0)
        self.shock_min_spin.setDecimals(1)
        self.shock_min_spin.setValue(-40.0)
        self.shock_min_spin.setSuffix(" %")

        self.shock_max_spin = QDoubleSpinBox()
        self.shock_max_spin.setRange(-99.0, 99.0)
        self.shock_max_spin.setDecimals(1)
        self.shock_max_spin.setValue(15.0)
        self.shock_max_spin.setSuffix(" %")

        self.shock_step_spin = QDoubleSpinBox()
        self.shock_step_spin.setRange(1.0, 50.0)
        self.shock_step_spin.setDecimals(1)
        self.shock_step_spin.setValue(10.0)
        self.shock_step_spin.setSuffix(" %")

        self.preset_shock_spin = QDoubleSpinBox()
        self.preset_shock_spin.setRange(-99.0, 99.0)
        self.preset_shock_spin.setDecimals(1)
        self.preset_shock_spin.setValue(-10.0)
        self.preset_shock_spin.setSuffix(" %")

        self.refresh_btn = self.refresh_icon_btn

        self.last_refresh_label = QLabel("마지막 갱신: -")
        self.last_refresh_label.setObjectName("StatusLabel")
        self.state_label = QLabel("대기 중")
        self.state_label.setObjectName("StatusLabel")
        self.error_label = QLabel("")
        self.error_label.setObjectName("ErrorLabel")
        self.error_label.setWordWrap(True)

        controls_layout.addWidget(QLabel("API Key"), 0, 0)
        controls_layout.addWidget(self.api_key_edit, 0, 1)
        controls_layout.addWidget(QLabel("API Secret"), 0, 2)
        controls_layout.addWidget(self.api_secret_edit, 0, 3)
        controls_layout.addWidget(QLabel("자동 새로고침"), 1, 0)
        controls_layout.addWidget(self.auto_refresh_check, 1, 1)
        controls_layout.addWidget(QLabel("간격"), 1, 2)
        controls_layout.addWidget(self.interval_spin, 1, 3)
        controls_layout.addWidget(QLabel("충격 최소"), 2, 0)
        controls_layout.addWidget(self.shock_min_spin, 2, 1)
        controls_layout.addWidget(QLabel("충격 최대"), 2, 2)
        controls_layout.addWidget(self.shock_max_spin, 2, 3)
        controls_layout.addWidget(QLabel("스텝"), 3, 0)
        controls_layout.addWidget(self.shock_step_spin, 3, 1)
        controls_layout.addWidget(QLabel("시나리오"), 3, 2)
        controls_layout.addWidget(self.preset_shock_spin, 3, 3)
        controls_layout.addWidget(self.last_refresh_label, 4, 0, 1, 4)
        controls_layout.addWidget(self.error_label, 5, 0, 1, 4)

        root.addWidget(self.controls_group)
        self.order_group = self._build_order_panel()
        self.order_dialog = QDialog(self)
        self.order_dialog.setWindowTitle("주문 모드")
        self.order_dialog.setModal(False)
        # 화면 크기에 맞게 다이얼로그 크기 설정
        _screen = QApplication.primaryScreen()
        if _screen is not None:
            _ag = _screen.availableGeometry()
            self.order_dialog.resize(min(1300, int(_ag.width() * 0.82)),
                                     min(860, int(_ag.height() * 0.82)))
        else:
            self.order_dialog.resize(1200, 820)
        order_dialog_layout = QVBoxLayout(self.order_dialog)
        order_dialog_layout.setContentsMargins(20, 20, 20, 20)
        order_dialog_layout.setSpacing(16)
        # 다이얼로그 헤더
        dlg_hdr = QHBoxLayout()
        order_dialog_title = QLabel("⊕  주문 모드")
        order_dialog_title.setObjectName("AppTitle")
        dlg_hdr.addWidget(order_dialog_title)
        dlg_hdr.addStretch(1)
        self.order_state_label_hdr = QLabel("대기 중")
        self.order_state_label_hdr.setObjectName("StatusLabel")
        dlg_hdr.addWidget(self.order_state_label_hdr)
        order_dialog_layout.addLayout(dlg_hdr)
        order_dialog_layout.addWidget(self.order_group, 1)
        # 메인화면과 동일한 테마 적용
        self.order_dialog.setStyleSheet(self.styleSheet())

        self.metrics_box = QGroupBox("핵심 지표")
        metrics_layout = QGridLayout(self.metrics_box)
        metrics_layout.setHorizontalSpacing(12)
        metrics_layout.setVerticalSpacing(12)
        self.metric_cards: Dict[str, MetricCard] = {}
        metric_defs = [
            ("총 지갑 잔고", "0.00 USDT", "계정 전체 wallet balance", "#38bdf8"),
            ("마진 잔고", "0.00 USDT", "total margin balance", "#818cf8"),
            ("가용 잔고", "0.00 USDT", "available balance", "#22c55e"),
            ("미실현손익", "0.00 USDT", "open position 기준", "#fb7185"),
            ("포지션 수", "0", "열린 USD-M 포지션", "#fbbf24"),
            ("미체결 수", "0", "open orders count", "#c084fc"),
            ("유지증거금 여유", "0.00 USDT", "margin - maint margin", "#34d399"),
            ("시나리오 위험", "-", "preset shock 결과", "#f97316"),
        ]
        for idx, (title_txt, value_txt, subtitle_txt, accent) in enumerate(metric_defs):
            card = MetricCard(title_txt, value_txt, subtitle_txt, accent=accent)
            self.metric_cards[title_txt] = card
            metrics_layout.addWidget(card, idx // 4, idx % 4)
        root.addWidget(self.metrics_box)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.setMovable(False)
        self.tabs.setTabPosition(QTabWidget.TabPosition.North)

        self.positions_table = self._make_table()
        self.orders_table = self._make_table()
        self.collateral_table = self._make_table()
        self.trades_table = self._make_table()
        self.distance_table = self._make_table()
        self.stress_table = self._make_table()
        self.position_pie_panel = PieChartPanel("Position Exposure", "Position exposure by symbol.")
        self.order_pie_panel = PieChartPanel("Open Order Exposure", "Open order exposure by symbol.")
        self.collateral_pie_panel = PieChartPanel("Collateral Allocation", "Collateral allocation by asset.")

        overview_page = QWidget()
        overview_layout = QGridLayout(overview_page)
        overview_layout.setContentsMargins(8, 8, 8, 8)
        overview_layout.setHorizontalSpacing(12)
        overview_layout.setVerticalSpacing(12)
        overview_layout.addWidget(self.position_pie_panel, 0, 0)
        overview_layout.addWidget(self.order_pie_panel, 0, 1)
        overview_layout.setColumnStretch(0, 1)
        overview_layout.setColumnStretch(1, 1)
        overview_layout.setRowStretch(0, 1)
        overview_layout.setRowStretch(1, 1)

        charts_page = QWidget()
        charts_page_layout = QVBoxLayout(charts_page)
        charts_page_layout.setContentsMargins(8, 8, 8, 8)
        charts_page_layout.addWidget(overview_page)

        self.tabs.addTab(charts_page, "Charts")
        self.history_panel = HistoryPanel()
        self.history_panel.update_history(self._history_df)
        self.tabs.addTab(self.history_panel, "History")
        self.tabs.addTab(self._table_page(self.positions_table, "포지션", "futures_account_position_information 결과를 요약합니다."), "포지션")
        self.tabs.addTab(self._table_page(self.orders_table, "미체결", "현재 열려 있는 지정가/스톱 주문입니다."), "미체결")
        self.tabs.addTab(self._table_page(self.trades_table, "최근 체결", "최근 3일 체결 내역입니다."), "체결")
        self.tabs.addTab(self._table_page(self.collateral_table, "담보 자산", "계정 자산을 USDT 환산으로 봅니다."), "담보")
        self.tabs.addTab(self._table_page(self.distance_table, "주문-마크 거리", "지정가가 현재 마크 가격과 얼마나 떨어져 있는지 봅니다."), "거리")
        self.tabs.addTab(self._table_page(self.stress_table, "스트레스 시나리오", "가격 충격 스윕 요약입니다."), "시나리오")
        root.addWidget(self.tabs, 1)

        # 로그 박스: UI에 표시하지 않되 내부 참조 유지
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)

        self.setCentralWidget(central)
        self.statusBar().showMessage("준비됨")

    def _show_collateral_tab(self) -> None:
        self.tabs.setCurrentIndex(5)

    def _show_positions_tab(self) -> None:
        self.tabs.setCurrentIndex(2)

    def _decorate_header_button(self, button: QToolButton, icon: QIcon) -> None:
        button.setIcon(icon)
        button.setFixedSize(42, 42)
        button.setIconSize(QSize(40, 40))  # 버튼에 꽉 차게
        effect = QGraphicsDropShadowEffect(self)
        effect.setBlurRadius(14)
        effect.setOffset(0, 2)
        effect.setColor(QColor(0, 0, 0, 100))
        button.setGraphicsEffect(effect)

    def _symbol_search_box(self, combo: QComboBox, label: str, on_search: Any) -> QWidget:
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(combo, 1)
        search_btn = QToolButton()
        search_btn.setObjectName("SearchIconButton")
        search_btn.setToolTip("코인 검색")
        search_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        search_btn.setIcon(_make_glyph_icon("⌕", "#f59e0b", "#f97316"))
        search_btn.setIconSize(QSize(32, 32))
        search_btn.setFixedSize(46, 46)
        effect = QGraphicsDropShadowEffect(self)
        effect.setBlurRadius(18)
        effect.setOffset(0, 4)
        effect.setColor(QColor(15, 23, 42, 160))
        search_btn.setGraphicsEffect(effect)
        search_btn.clicked.connect(on_search)
        layout.addWidget(search_btn)
        return container

    def _search_symbol_combo(self, combo: QComboBox) -> None:
        items = [combo.itemText(i) for i in range(combo.count()) if combo.itemText(i)]
        if not items:
            self.statusBar().showMessage("검색할 코인 목록이 아직 없습니다.")
            return
        query, ok = QInputDialog.getText(self, "코인 검색", "검색할 코인 심볼을 입력하세요", text=combo.currentText().strip())
        if not ok:
            return
        query = query.strip().upper()
        if not query:
            return
        matches = [item for item in items if query in item.upper()]
        if not matches:
            self.statusBar().showMessage(f"'{query}' 검색 결과가 없습니다.")
            return
        if len(matches) == 1:
            combo.setCurrentText(matches[0])
            return
        selected, ok2 = QInputDialog.getItem(self, "검색 결과", "코인을 선택하세요", matches, 0, False)
        if ok2 and selected:
            combo.setCurrentText(selected)

    def _build_order_panel(self) -> QWidget:
        """4-카드 플랫 그리드 주문 패널."""
        container = QWidget()
        root_layout = QVBoxLayout(container)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(14)

        # ── 상단 2개 카드 (일괄) ──────────────────────────────
        top_row = QHBoxLayout()
        top_row.setSpacing(14)
        top_row.addWidget(self._build_bulk_manual_tab(), 1)
        top_row.addWidget(self._build_bulk_auto_tab(), 1)
        root_layout.addLayout(top_row, 1)

        # ── 하단 2개 카드 (개별) ──────────────────────────────
        bot_row = QHBoxLayout()
        bot_row.setSpacing(14)
        bot_row.addWidget(self._build_single_manual_tab(), 1)
        bot_row.addWidget(self._build_single_auto_tab(), 1)
        root_layout.addLayout(bot_row, 1)

        # ── 상태 바 ───────────────────────────────────────────
        status_row = QHBoxLayout()
        self.order_state_label = QLabel("주문 모드 대기 중")
        self.order_state_label.setObjectName("StatusLabel")
        status_row.addWidget(self.order_state_label)
        status_row.addStretch(1)
        root_layout.addLayout(status_row)
        return container

    def _build_bulk_order_page(self) -> QWidget:
        return QWidget()  # 호환성 유지 (미사용)

    def _build_single_order_page(self) -> QWidget:
        return QWidget()  # 호환성 유지 (미사용)

    def _build_bulk_manual_tab(self) -> QFrame:
        card = QFrame()
        card.setObjectName("OrderCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(14)

        # 카드 헤더
        hdr = QHBoxLayout()
        badge = QLabel("일괄")
        badge.setObjectName("OrderBadgeBulk")
        title = QLabel("전체 포지션 수동 청산")
        title.setObjectName("OrderCardTitle")
        hdr.addWidget(badge)
        hdr.addWidget(title)
        hdr.addStretch(1)
        layout.addLayout(hdr)

        desc = QLabel("열린 모든 포지션을 지정 비율만큼 즉시 시장가로 실행합니다.")
        desc.setObjectName("SectionHint")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # 컨트롤
        ctrl = QGridLayout()
        ctrl.setHorizontalSpacing(12)
        ctrl.setVerticalSpacing(10)

        lbl_side = QLabel("방향")
        lbl_side.setObjectName("OrderFieldLabel")
        self.bulk_manual_side_combo = QComboBox()
        self.bulk_manual_side_combo.addItems(["BUY", "SELL"])

        lbl_pct = QLabel("비율")
        lbl_pct.setObjectName("OrderFieldLabel")
        self.bulk_manual_pct_spin = QDoubleSpinBox()
        self.bulk_manual_pct_spin.setRange(0.1, 100.0)
        self.bulk_manual_pct_spin.setDecimals(1)
        self.bulk_manual_pct_spin.setSingleStep(0.3)
        self.bulk_manual_pct_spin.setValue(0.3)
        self.bulk_manual_pct_spin.setSuffix(" %")

        self.bulk_manual_reduce_only = QCheckBox("Reduce only")
        self.bulk_manual_reduce_only.setChecked(True)

        self.bulk_manual_exec_btn = QPushButton("⚡  전체 실행")
        self.bulk_manual_exec_btn.setObjectName("OrderExecButton")
        self.bulk_manual_exec_btn.clicked.connect(self._run_bulk_manual_order)

        ctrl.addWidget(lbl_side, 0, 0)
        ctrl.addWidget(self.bulk_manual_side_combo, 0, 1)
        ctrl.addWidget(lbl_pct, 0, 2)
        ctrl.addWidget(self.bulk_manual_pct_spin, 0, 3)
        ctrl.addWidget(self.bulk_manual_reduce_only, 1, 0, 1, 2)
        ctrl.addWidget(self.bulk_manual_exec_btn, 1, 2, 1, 2)
        layout.addLayout(ctrl)
        layout.addStretch(1)

        self.bulk_manual_hint = QLabel("")
        self.bulk_manual_hint.setObjectName("SectionHint")
        self.bulk_manual_hint.setWordWrap(True)
        layout.addWidget(self.bulk_manual_hint)
        return card

    def _build_bulk_auto_tab(self) -> QFrame:
        card = QFrame()
        card.setObjectName("OrderCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(14)

        hdr = QHBoxLayout()
        badge = QLabel("일괄 자동")
        badge.setObjectName("OrderBadgeAuto")
        title = QLabel("가격 하락 시 자동 청산")
        title.setObjectName("OrderCardTitle")
        hdr.addWidget(badge)
        hdr.addWidget(title)
        hdr.addStretch(1)
        layout.addLayout(hdr)

        desc = QLabel("감시 심볼이 설정 시점 대비 기준치 이상 하락하면 모든 미체결을 취소하고 롱 포지션 일부를 시장가 매도합니다.")
        desc.setObjectName("SectionHint")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        ctrl = QGridLayout()
        ctrl.setHorizontalSpacing(12)
        ctrl.setVerticalSpacing(10)

        lbl_sym = QLabel("감시 심볼")
        lbl_sym.setObjectName("OrderFieldLabel")
        self.bulk_auto_symbol_combo = QComboBox()
        self.bulk_auto_symbol_combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)

        lbl_drop = QLabel("하락 기준")
        lbl_drop.setObjectName("OrderFieldLabel")
        self.bulk_auto_drop_spin = QDoubleSpinBox()
        self.bulk_auto_drop_spin.setRange(0.1, 99.0)
        self.bulk_auto_drop_spin.setDecimals(1)
        self.bulk_auto_drop_spin.setValue(5.0)
        self.bulk_auto_drop_spin.setSuffix(" %")

        lbl_pct = QLabel("매도 비율")
        lbl_pct.setObjectName("OrderFieldLabel")
        self.bulk_auto_pct_spin = QDoubleSpinBox()
        self.bulk_auto_pct_spin.setRange(0.1, 100.0)
        self.bulk_auto_pct_spin.setDecimals(1)
        self.bulk_auto_pct_spin.setValue(20.0)
        self.bulk_auto_pct_spin.setSuffix(" %")

        self.bulk_auto_arm_btn = QPushButton("🔍  감시 시작")
        self.bulk_auto_arm_btn.setObjectName("OrderWatchButton")
        self.bulk_auto_arm_btn.setCheckable(True)
        self.bulk_auto_arm_btn.toggled.connect(self._toggle_bulk_auto_arm)

        ctrl.addWidget(lbl_sym, 0, 0)
        ctrl.addWidget(self._symbol_search_box(self.bulk_auto_symbol_combo, "검색", lambda: self._search_symbol_combo(self.bulk_auto_symbol_combo)), 0, 1)
        ctrl.addWidget(lbl_drop, 0, 2)
        ctrl.addWidget(self.bulk_auto_drop_spin, 0, 3)
        ctrl.addWidget(lbl_pct, 1, 0)
        ctrl.addWidget(self.bulk_auto_pct_spin, 1, 1)
        ctrl.addWidget(self.bulk_auto_arm_btn, 1, 2, 1, 2)
        layout.addLayout(ctrl)
        layout.addStretch(1)

        self.bulk_auto_hint = QLabel("")
        self.bulk_auto_hint.setObjectName("OrderAutoStatus")
        self.bulk_auto_hint.setWordWrap(True)
        layout.addWidget(self.bulk_auto_hint)
        return card

    def _build_single_manual_tab(self) -> QFrame:
        card = QFrame()
        card.setObjectName("OrderCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(14)

        hdr = QHBoxLayout()
        badge = QLabel("개별")
        badge.setObjectName("OrderBadgeSingle")
        title = QLabel("심볼 선택 수동 실행")
        title.setObjectName("OrderCardTitle")
        hdr.addWidget(badge)
        hdr.addWidget(title)
        hdr.addStretch(1)
        layout.addLayout(hdr)

        desc = QLabel("특정 심볼의 포지션을 지정 비율만큼 즉시 시장가로 실행합니다.")
        desc.setObjectName("SectionHint")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        ctrl = QGridLayout()
        ctrl.setHorizontalSpacing(12)
        ctrl.setVerticalSpacing(10)

        lbl_sym = QLabel("심볼")
        lbl_sym.setObjectName("OrderFieldLabel")
        self.single_manual_symbol_combo = QComboBox()
        self.single_manual_symbol_combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.single_manual_symbol_combo.currentTextChanged.connect(self._update_single_manual_preview)

        lbl_side = QLabel("방향")
        lbl_side.setObjectName("OrderFieldLabel")
        self.single_manual_side_combo = QComboBox()
        self.single_manual_side_combo.addItems(["BUY", "SELL"])
        self.single_manual_side_combo.currentTextChanged.connect(self._update_single_manual_preview)

        lbl_pct = QLabel("비율")
        lbl_pct.setObjectName("OrderFieldLabel")
        self.single_manual_pct_spin = QDoubleSpinBox()
        self.single_manual_pct_spin.setRange(0.1, 100.0)
        self.single_manual_pct_spin.setDecimals(1)
        self.single_manual_pct_spin.setSingleStep(0.3)
        self.single_manual_pct_spin.setValue(0.3)
        self.single_manual_pct_spin.setSuffix(" %")
        self.single_manual_pct_spin.valueChanged.connect(
            lambda *_: self._update_single_manual_preview(self.single_manual_symbol_combo.currentText()))

        lbl_qty = QLabel("직접 수량")
        lbl_qty.setObjectName("OrderFieldLabel")
        self.single_manual_qty_spin = QDoubleSpinBox()
        self.single_manual_qty_spin.setRange(0.0, 999999.0)
        self.single_manual_qty_spin.setDecimals(4)
        self.single_manual_qty_spin.setValue(0.0)
        self.single_manual_qty_spin.setSpecialValueText("비율 사용")
        self.single_manual_qty_spin.setToolTip("0이면 위 비율로 계산. 0보다 크면 직접 입력한 수량으로 실행.")
        self.single_manual_qty_spin.valueChanged.connect(
            lambda *_: self._update_single_manual_preview(self.single_manual_symbol_combo.currentText()))

        self.single_manual_reduce_only = QCheckBox("Reduce only")
        self.single_manual_reduce_only.setChecked(True)
        self.single_manual_reduce_only.toggled.connect(
            lambda *_: self._update_single_manual_preview(self.single_manual_symbol_combo.currentText()))

        self.single_manual_exec_btn = QPushButton("⚡  개별 실행")
        self.single_manual_exec_btn.setObjectName("OrderExecButton")
        self.single_manual_exec_btn.clicked.connect(self._run_single_manual_order)

        ctrl.addWidget(lbl_sym, 0, 0)
        ctrl.addWidget(self._symbol_search_box(self.single_manual_symbol_combo, "검색", lambda: self._search_symbol_combo(self.single_manual_symbol_combo)), 0, 1)
        ctrl.addWidget(lbl_side, 0, 2)
        ctrl.addWidget(self.single_manual_side_combo, 0, 3)
        ctrl.addWidget(lbl_pct, 1, 0)
        ctrl.addWidget(self.single_manual_pct_spin, 1, 1)
        ctrl.addWidget(lbl_qty, 1, 2)
        ctrl.addWidget(self.single_manual_qty_spin, 1, 3)
        ctrl.addWidget(self.single_manual_reduce_only, 2, 0)
        ctrl.addWidget(self.single_manual_exec_btn, 2, 1, 1, 3)
        layout.addLayout(ctrl)

        self.single_manual_preview = QLabel("심볼을 선택하면 실행 예정량이 표시됩니다.")
        self.single_manual_preview.setObjectName("OrderPreviewLabel")
        self.single_manual_preview.setWordWrap(True)
        layout.addWidget(self.single_manual_preview)
        layout.addStretch(1)
        return card

    def _build_single_auto_tab(self) -> QFrame:
        card = QFrame()
        card.setObjectName("OrderCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(14)

        hdr = QHBoxLayout()
        badge = QLabel("개별 자동")
        badge.setObjectName("OrderBadgeAutoSingle")
        title = QLabel("심볼 가격 변동 자동 감시")
        title.setObjectName("OrderCardTitle")
        hdr.addWidget(badge)
        hdr.addWidget(title)
        hdr.addStretch(1)
        layout.addLayout(hdr)

        desc = QLabel("선택한 심볼이 설정 시점 대비 기준 이상 움직이면 해당 심볼의 미체결을 취소하고 포지션 일부를 실행합니다.")
        desc.setObjectName("SectionHint")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        ctrl = QGridLayout()
        ctrl.setHorizontalSpacing(12)
        ctrl.setVerticalSpacing(10)

        lbl_sym = QLabel("심볼")
        lbl_sym.setObjectName("OrderFieldLabel")
        self.single_auto_symbol_combo = QComboBox()
        self.single_auto_symbol_combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.single_auto_symbol_combo.currentTextChanged.connect(self._update_single_auto_preview)

        lbl_side = QLabel("방향")
        lbl_side.setObjectName("OrderFieldLabel")
        self.single_auto_side_combo = QComboBox()
        self.single_auto_side_combo.addItems(["SELL", "BUY"])
        self.single_auto_side_combo.currentTextChanged.connect(self._update_single_auto_preview)

        lbl_drop = QLabel("변동 기준")
        lbl_drop.setObjectName("OrderFieldLabel")
        self.single_auto_drop_spin = QDoubleSpinBox()
        self.single_auto_drop_spin.setRange(0.1, 99.0)
        self.single_auto_drop_spin.setDecimals(1)
        self.single_auto_drop_spin.setValue(5.0)
        self.single_auto_drop_spin.setSuffix(" %")
        self.single_auto_drop_spin.valueChanged.connect(
            lambda *_: self._update_single_auto_preview(self.single_auto_symbol_combo.currentText()))

        lbl_pct = QLabel("실행 비율")
        lbl_pct.setObjectName("OrderFieldLabel")
        self.single_auto_pct_spin = QDoubleSpinBox()
        self.single_auto_pct_spin.setRange(0.1, 100.0)
        self.single_auto_pct_spin.setDecimals(1)
        self.single_auto_pct_spin.setValue(20.0)
        self.single_auto_pct_spin.setSuffix(" %")
        self.single_auto_pct_spin.valueChanged.connect(
            lambda *_: self._update_single_auto_preview(self.single_auto_symbol_combo.currentText()))

        self.single_auto_arm_btn = QPushButton("🔍  개별 감시 시작")
        self.single_auto_arm_btn.setObjectName("OrderWatchButton")
        self.single_auto_arm_btn.setCheckable(True)
        self.single_auto_arm_btn.toggled.connect(self._toggle_single_auto_arm)

        ctrl.addWidget(lbl_sym, 0, 0)
        ctrl.addWidget(self._symbol_search_box(self.single_auto_symbol_combo, "검색", lambda: self._search_symbol_combo(self.single_auto_symbol_combo)), 0, 1)
        ctrl.addWidget(lbl_side, 0, 2)
        ctrl.addWidget(self.single_auto_side_combo, 0, 3)
        ctrl.addWidget(lbl_drop, 1, 0)
        ctrl.addWidget(self.single_auto_drop_spin, 1, 1)
        ctrl.addWidget(lbl_pct, 1, 2)
        ctrl.addWidget(self.single_auto_pct_spin, 1, 3)
        ctrl.addWidget(self.single_auto_arm_btn, 2, 0, 1, 4)
        layout.addLayout(ctrl)

        self.single_auto_hint = QLabel("")
        self.single_auto_hint.setObjectName("OrderAutoStatus")
        self.single_auto_hint.setWordWrap(True)
        layout.addWidget(self.single_auto_hint)
        layout.addStretch(1)
        return card

    def _show_order_dialog(self) -> None:
        if self._latest_payload and isinstance(self._latest_payload, dict):
            snapshot = self._latest_payload.get("snapshot", {})
            if isinstance(snapshot, dict):
                self._update_order_symbol_options(snapshot)
        self.order_dialog.show()
        self.order_dialog.raise_()
        self.order_dialog.activateWindow()

    def _current_snapshot(self) -> Dict[str, Any]:
        payload = self._latest_payload or {}
        snapshot = payload.get("snapshot")
        return snapshot if isinstance(snapshot, dict) else {}

    def _snapshot_positions(self) -> List[Dict[str, Any]]:
        snapshot = self._current_snapshot()
        positions = snapshot.get("positions", [])
        return positions if isinstance(positions, list) else []

    def _snapshot_open_orders(self) -> List[Dict[str, Any]]:
        snapshot = self._current_snapshot()
        orders = snapshot.get("open_orders", [])
        return orders if isinstance(orders, list) else []

    def _snapshot_marks(self) -> Dict[str, float]:
        snapshot = self._current_snapshot()
        marks = snapshot.get("mark_prices", {})
        return marks if isinstance(marks, dict) else {}

    def _update_order_symbol_options(self, snapshot: Dict[str, Any]) -> None:
        symbols = sorted(
            {p.get("symbol") for p in snapshot.get("positions", []) or [] if p.get("symbol")}
            | {o.get("symbol") for o in snapshot.get("open_orders", []) or [] if o.get("symbol")}
        )
        if not symbols:
            symbols = ["BTCUSDT"]
        combos = [
            getattr(self, "bulk_auto_symbol_combo", None),
            getattr(self, "single_manual_symbol_combo", None),
            getattr(self, "single_auto_symbol_combo", None),
        ]
        for combo in combos:
            if combo is None:
                continue
            current = combo.currentText().strip()
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(symbols)
            if current:
                combo.setEditText(current)
            else:
                combo.setCurrentIndex(0)
            combo.blockSignals(False)
        if not self.bulk_auto_arm_btn.isChecked():
            self.bulk_auto_symbol_combo.setCurrentText(symbols[0])
        if not self.single_auto_arm_btn.isChecked():
            self.single_auto_symbol_combo.setCurrentText(symbols[0])
        self._update_single_manual_preview(self.single_manual_symbol_combo.currentText())
        self._update_single_auto_preview(self.single_auto_symbol_combo.currentText())

    def _selected_symbol_mark(self, symbol: str) -> float:
        marks = self._snapshot_marks()
        return float(marks.get(symbol, 0.0) or 0.0)

    def _position_amount_for_symbol(self, symbol: str) -> float:
        for pos in self._snapshot_positions():
            if pos.get("symbol") == symbol:
                try:
                    return float(pos.get("positionAmt", 0.0) or 0.0)
                except (TypeError, ValueError):
                    return 0.0
        return 0.0

    def _manual_order_quantity(self, symbol: str, pct: float) -> float:
        pos_amt = abs(self._position_amount_for_symbol(symbol))
        return pos_amt * max(0.0, pct) / 100.0

    def _update_single_manual_preview(self, symbol: str) -> None:
        symbol = (symbol or "").strip().upper()
        if not symbol:
            self.single_manual_preview.setText("심볼을 선택하면 현재 포지션 기준으로 실행량이 표시됩니다.")
            return
        pct = float(self.single_manual_pct_spin.value())
        direct_qty = float(self.single_manual_qty_spin.value())
        side = self.single_manual_side_combo.currentText().upper()
        reduce_only = self.single_manual_reduce_only.isChecked()
        pos_amt = self._position_amount_for_symbol(symbol)
        mark = self._selected_symbol_mark(symbol)
        if direct_qty > 0:
            qty = direct_qty
            mode_str = f"직접 수량 {qty:.4f}"
        else:
            qty = self._manual_order_quantity(symbol, pct)
            mode_str = f"{pct:.1f}% → {qty:.4f}"
        self.single_manual_preview.setText(
            f"{symbol} | 포지션 {pos_amt:+.4f} | 기준가 {_fmt_price(mark)} | "
            f"{side} {mode_str} | reduceOnly={reduce_only}"
        )

    def _update_single_auto_preview(self, symbol: str) -> None:
        symbol = (symbol or "").strip().upper()
        if not symbol:
            self.single_auto_hint.setText("선택 심볼이 설정 시점 대비 기준 이상 움직이면, 해당 심볼의 미체결을 취소하고 현재 포지션 일부를 실행합니다.")
            return
        baseline = self._order_auto_baselines.get("single", {}).get(symbol)
        mark = self._selected_symbol_mark(symbol)
        if baseline:
            move = ((mark - baseline) / baseline * 100.0) if baseline else 0.0
            self.single_auto_hint.setText(
                f"기준가 {_fmt_price(baseline)} / 현재가 {_fmt_price(mark)} / 변동 {move:+.2f}%"
            )
        else:
            self.single_auto_hint.setText(
                "자동 감시를 시작하면 현재가가 기준점이 됩니다. 기준 하락 시 해당 심볼의 미체결을 취소하고 실행합니다."
            )

    def _toggle_bulk_auto_arm(self, checked: bool) -> None:
        symbol = self.bulk_auto_symbol_combo.currentText().strip().upper()
        if checked:
            mark = self._selected_symbol_mark(symbol)
            if mark <= 0:
                self.bulk_auto_arm_btn.blockSignals(True)
                self.bulk_auto_arm_btn.setChecked(False)
                self.bulk_auto_arm_btn.blockSignals(False)
                self.order_state_label.setText("자동 감시를 시작할 기준가를 찾지 못했습니다.")
                return
            self._order_auto_baselines["bulk"] = {symbol: mark}
            self.order_state_label.setText(f"일괄 자동 감시 시작: {symbol} 기준가 {_fmt_price(mark)}")
            self.bulk_auto_arm_btn.setText("자동 감시 중지")
            self.bulk_auto_symbol_combo.setEnabled(False)
        else:
            self._order_auto_baselines["bulk"] = {}
            self.order_state_label.setText("일괄 자동 감시 해제")
            self.bulk_auto_arm_btn.setText("자동 감시 시작")
            self.bulk_auto_symbol_combo.setEnabled(True)

    def _toggle_single_auto_arm(self, checked: bool) -> None:
        symbol = self.single_auto_symbol_combo.currentText().strip().upper()
        if checked:
            mark = self._selected_symbol_mark(symbol)
            if mark <= 0:
                self.single_auto_arm_btn.blockSignals(True)
                self.single_auto_arm_btn.setChecked(False)
                self.single_auto_arm_btn.blockSignals(False)
                self.order_state_label.setText("자동 감시를 시작할 기준가를 찾지 못했습니다.")
                return
            self._order_auto_baselines["single"] = {symbol: mark}
            self.order_state_label.setText(f"개별 자동 감시 시작: {symbol} 기준가 {_fmt_price(mark)}")
            self.single_auto_arm_btn.setText("자동 감시 중지")
            self.single_auto_symbol_combo.setEnabled(False)
        else:
            self._order_auto_baselines["single"] = {}
            self.order_state_label.setText("개별 자동 감시 해제")
            self.single_auto_arm_btn.setText("자동 감시 시작")
            self.single_auto_symbol_combo.setEnabled(True)

    def _build_manual_orders(self, symbol: str, side: str, pct: float, reduce_only: bool) -> List[PlannedOrder]:
        pos_amt = self._position_amount_for_symbol(symbol)
        if abs(pos_amt) <= 0:
            return []
        if reduce_only:
            if side == "SELL" and pos_amt <= 0:
                return []
            if side == "BUY" and pos_amt >= 0:
                return []
        qty = abs(pos_amt) * max(0.0, pct) / 100.0
        if qty <= 0:
            return []
        return [PlannedOrder(symbol=symbol, side=side, quantity=qty, reduce_only=reduce_only)]

    def _build_bulk_manual_plan(self) -> Tuple[List[str], List[PlannedOrder], str]:
        snapshot = self._current_snapshot()
        side = self.bulk_manual_side_combo.currentText().upper()
        pct = float(self.bulk_manual_pct_spin.value())
        reduce_only = self.bulk_manual_reduce_only.isChecked()
        cancel_symbols: List[str] = []
        orders: List[PlannedOrder] = []
        for pos in snapshot.get("positions", []) or []:
            symbol = pos.get("symbol")
            if not symbol:
                continue
            try:
                pos_amt = float(pos.get("positionAmt", 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
            if abs(pos_amt) <= 0:
                continue
            if reduce_only:
                if side == "SELL" and pos_amt <= 0:
                    continue
                if side == "BUY" and pos_amt >= 0:
                    continue
            qty = abs(pos_amt) * max(0.0, pct) / 100.0
            if qty > 0:
                orders.append(PlannedOrder(symbol=symbol, side=side, quantity=qty, reduce_only=reduce_only))
        label = f"일괄 수동 {side} {pct:.1f}%"
        return cancel_symbols, orders, label

    def _build_single_manual_plan(self) -> Tuple[List[str], List[PlannedOrder], str]:
        symbol = self.single_manual_symbol_combo.currentText().strip().upper()
        side = self.single_manual_side_combo.currentText().upper()
        pct = float(self.single_manual_pct_spin.value())
        direct_qty = float(self.single_manual_qty_spin.value())
        reduce_only = self.single_manual_reduce_only.isChecked()
        if direct_qty > 0:
            orders = [PlannedOrder(symbol=symbol, side=side, quantity=direct_qty, reduce_only=reduce_only)] if symbol else []
            label = f"{symbol} 개별 수동 {side} 직접수량 {direct_qty:.4f}"
        else:
            orders = self._build_manual_orders(symbol, side, pct, reduce_only) if symbol else []
            label = f"{symbol} 개별 수동 {side} {pct:.1f}%"
        return [], orders, label

    def _build_bulk_auto_plan(self) -> Tuple[List[str], List[PlannedOrder], str]:
        symbol = self.bulk_auto_symbol_combo.currentText().strip().upper()
        baseline = self._order_auto_baselines.get("bulk", {}).get(symbol)
        current = self._selected_symbol_mark(symbol)
        trigger = float(self.bulk_auto_drop_spin.value())
        pct = float(self.bulk_auto_pct_spin.value())
        if not baseline or current <= 0:
            return [], [], f"일괄 자동 대기: {symbol}"
        drop = (baseline - current) / baseline * 100.0
        if drop < trigger:
            return [], [], f"일괄 자동 대기: {symbol} {drop:+.2f}% / 기준 {trigger:.1f}%"

        cancel_symbols = [o.get("symbol") for o in self._snapshot_open_orders() if o.get("symbol")]
        orders: List[PlannedOrder] = []
        for pos in self._snapshot_positions():
            try:
                pos_amt = float(pos.get("positionAmt", 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
            if pos_amt <= 0:
                continue
            qty = abs(pos_amt) * max(0.0, pct) / 100.0
            if qty > 0:
                orders.append(PlannedOrder(symbol=pos.get("symbol"), side="SELL", quantity=qty, reduce_only=True))
        label = f"일괄 자동 트리거: {symbol} drop {drop:+.2f}%"
        return cancel_symbols, orders, label

    def _build_single_auto_plan(self) -> Tuple[List[str], List[PlannedOrder], str]:
        symbol = self.single_auto_symbol_combo.currentText().strip().upper()
        baseline = self._order_auto_baselines.get("single", {}).get(symbol)
        current = self._selected_symbol_mark(symbol)
        trigger = float(self.single_auto_drop_spin.value())
        pct = float(self.single_auto_pct_spin.value())
        side = self.single_auto_side_combo.currentText().upper()
        if not baseline or current <= 0:
            return [], [], f"개별 자동 대기: {symbol}"
        drop = (baseline - current) / baseline * 100.0
        if drop < trigger:
            return [], [], f"개별 자동 대기: {symbol} {drop:+.2f}% / 기준 {trigger:.1f}%"
        pos_amt = self._position_amount_for_symbol(symbol)
        if pos_amt == 0:
            return [symbol], [], f"개별 자동 대기: {symbol} 포지션 없음"
        if side == "SELL" and pos_amt <= 0:
            return [symbol], [], f"개별 자동 대기: {symbol} SELL 대상 아님"
        if side == "BUY" and pos_amt >= 0:
            return [symbol], [], f"개별 자동 대기: {symbol} BUY 대상 아님"
        qty = abs(pos_amt) * max(0.0, pct) / 100.0
        orders = [PlannedOrder(symbol=symbol, side=side, quantity=qty, reduce_only=True)] if qty > 0 else []
        label = f"{symbol} 개별 자동 트리거 drop {drop:+.2f}%"
        return [symbol], orders, label

    def _execute_order_plan(self, cancel_symbols: List[str], orders: List[PlannedOrder], label: str) -> None:
        if self._order_worker is not None and self._order_worker.isRunning():
            return
        if not cancel_symbols and not orders:
            self.order_state_label.setText(label)
            return
        self.order_state_label.setText(f"{label} 실행 중...")
        self.statusBar().showMessage(f"{label} 주문 실행 중...")
        self._order_worker = OrderWorker(
            self.api_key_edit.text().strip(),
            self.api_secret_edit.text().strip(),
            self.key_file_path,
            cancel_symbols,
            orders,
            label,
        )
        self._order_worker.loaded.connect(self._on_order_worker_finished)
        self._order_worker.failed.connect(self._on_order_worker_failed)
        self._order_worker.start()

    def _format_order_worker_log(self, payload: Dict[str, Any]) -> str:
        lines = [f"[주문] {payload.get('label', '-')}"]
        for cancel in payload.get("cancel_results", []) or []:
            sym = cancel.get("symbol", "-")
            if "error" in cancel:
                lines.append(f"- {sym} cancel failed: {cancel['error']}")
            else:
                lines.append(f"- {sym} open orders cancelled")
        for order in payload.get("order_results", []) or []:
            lines.append(
                f"- {order.get('symbol')} {order.get('side')} {order.get('quantity', 0):.6f} "
                f"reduceOnly={order.get('reduce_only')}"
            )
        return "\n".join(lines)

    def _on_order_worker_finished(self, payload: object) -> None:
        data = payload if isinstance(payload, dict) else {}
        self.order_state_label.setText(data.get("label", "주문 완료"))
        self.statusBar().showMessage("주문 실행 완료")
        self.log_box.setPlainText(self._format_order_worker_log(data))
        self._order_worker = None
        if self.bulk_auto_arm_btn.isChecked() and data.get("label", "").startswith("일괄 자동 트리거"):
            self.bulk_auto_arm_btn.setChecked(False)
        if self.single_auto_arm_btn.isChecked() and data.get("label", "").startswith("개별 자동 트리거"):
            self.single_auto_arm_btn.setChecked(False)

    def _on_order_worker_failed(self, error: str) -> None:
        self.order_state_label.setText("주문 실행 실패")
        self.statusBar().showMessage("주문 실행 실패")
        self.error_label.setText(error.splitlines()[0] if error else "주문 실행 실패")
        self.log_box.setPlainText(error)
        self._order_worker = None

    def _run_bulk_manual_order(self) -> None:
        snapshot = self._current_snapshot()
        if not snapshot:
            self.order_state_label.setText("먼저 조회를 실행해 주세요.")
            return
        cancel_symbols, orders, label = self._build_bulk_manual_plan()
        if not orders:
            self.order_state_label.setText("실행할 주문이 없습니다.")
            return
        if QMessageBox.question(self, "주문 확인", f"{label}\n\n{len(orders)}개 주문을 실행할까요?") != QMessageBox.StandardButton.Yes:
            return
        self._execute_order_plan(cancel_symbols, orders, label)

    def _run_single_manual_order(self) -> None:
        snapshot = self._current_snapshot()
        if not snapshot:
            self.order_state_label.setText("먼저 조회를 실행해 주세요.")
            return
        cancel_symbols, orders, label = self._build_single_manual_plan()
        if not orders:
            self.order_state_label.setText("실행할 주문이 없습니다.")
            return
        if QMessageBox.question(self, "주문 확인", f"{label}\n\n{len(orders)}개 주문을 실행할까요?") != QMessageBox.StandardButton.Yes:
            return
        self._execute_order_plan(cancel_symbols, orders, label)

    def _process_auto_orders(self, snapshot: Dict[str, Any]) -> None:
        self._update_order_symbol_options(snapshot)
        if self.bulk_auto_arm_btn.isChecked():
            cancel_symbols, orders, label = self._build_bulk_auto_plan()
            if orders or cancel_symbols:
                self._execute_order_plan(cancel_symbols, orders, label)
        if self.single_auto_arm_btn.isChecked():
            cancel_symbols, orders, label = self._build_single_auto_plan()
            if orders or cancel_symbols:
                self._execute_order_plan(cancel_symbols, orders, label)

    def _toggle_settings_panel(self) -> None:
        visible = self.controls_group.isVisible()
        self.controls_group.setVisible(not visible)
        self.settings_toggle_btn.setToolTip("조회 설정 보기" if visible else "조회 설정 숨기기")

    def _toggle_metrics_panel(self) -> None:
        visible = self.metrics_box.isVisible()
        self.metrics_box.setVisible(not visible)
        tip = "핵심 지표 보기" if visible else "핵심 지표 숨기기"
        self.metrics_toggle_btn_icon.setToolTip(tip)

    def _toggle_log_panel(self) -> None:
        pass  # 로그 패널 제거됨

    def _save_latest_log(self) -> None:
        text = self.log_box.toPlainText().strip()
        if not text:
            return
        default_name = f"latest_log_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt"
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "로그 저장",
            str(Path.home() / default_name),
            "Text Files (*.txt);;All Files (*)",
        )
        if not file_path:
            return
        Path(file_path).write_text(text + "\n", encoding="utf-8")

    def _table_page(self, table: QTableWidget, title: str, description: str) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 6, 0, 0)
        layout.setSpacing(8)
        header = QLabel(description)
        header.setWordWrap(True)
        header.setObjectName("SectionHint")
        layout.addWidget(header)
        layout.addWidget(table, 1)
        return page

    def _make_table(self) -> QTableWidget:
        table = QTableWidget()
        table.setSortingEnabled(True)
        table.setAlternatingRowColors(True)
        table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        table.horizontalHeader().setStretchLastSection(True)
        return table

    def _apply_theme(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QDialog {
                background: #0f172a;
            }
            QDialog QLabel {
                color: #e2e8f0;
            }
            QLabel {
                color: #e2e8f0;
                font-size: 13px;
            }
            QLabel#AppTitle {
                font-size: 28px;
                font-weight: 700;
                color: #f8fafc;
                letter-spacing: -0.02em;
            }
            QLabel#AppSubtitle {
                color: #cbd5e1;
                font-size: 14px;
                margin-bottom: 4px;
            }
            QLabel#SectionTitle {
                color: #e2e8f0;
                font-size: 14px;
                font-weight: 700;
            }
            QLabel#SectionHint, QLabel#StatusLabel {
                color: #94a3b8;
                font-size: 12px;
            }
            QLabel#TopHeader {
                color: #cbd5e1;
                font-size: 11px;
                font-weight: 700;
                padding: 0 2px 2px 2px;
            }
            QLabel#TopEmpty {
                color: #94a3b8;
                font-size: 11px;
                padding: 4px 2px;
            }
            QLabel#QuickStat {
                color: #dbeafe;
                background: rgba(15,23,42,0.7);
                border: 1px solid rgba(148,163,184,0.2);
                border-radius: 10px;
                padding: 8px 12px;
                font-size: 12px;
                font-weight: 700;
            }

            QFrame#TopCard {
                background: rgba(15,23,42,0.72);
                border: 1px solid rgba(148,163,184,0.14);
                border-radius: 10px;
            }
            QLabel#TopCardRank {
                color: #93c5fd;
                font-size: 10px;
                font-weight: 700;
            }
            QLabel#TopCardName {
                color: #e2e8f0;
                font-size: 11px;
                font-weight: 600;
            }
            QLabel#TopCardValue {
                color: #f8fafc;
                font-size: 11px;
                font-weight: 700;
            }
            QLabel#ErrorLabel {
                color: #fca5a5;
                font-size: 12px;
                padding-top: 4px;
            }
            QGroupBox {
                color: #e2e8f0;
                border: 1px solid rgba(148,163,184,0.18);
                border-radius: 14px;
                margin-top: 14px;
                padding: 14px;
                background: rgba(15,23,42,0.55);
                font-weight: 600;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
            }
            QFrame#MetricCard {
                background: linear-gradient(145deg, rgba(30,41,59,0.78), rgba(15,23,42,0.92));
                border: 1px solid rgba(148,163,184,0.15);
                border-radius: 14px;
                min-height: 112px;
            }
            QLabel#MetricTitle {
                color: #94a3b8;
                font-size: 12px;
            }
            QLabel#MetricValue {
                color: #f8fafc;
                font-size: 22px;
                font-weight: 700;
            }
            QLabel#MetricSubtitle {
                color: #cbd5e1;
                font-size: 11px;
            }
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
                background: rgba(15,23,42,0.95);
                color: #e2e8f0;
                border: 1px solid rgba(148,163,184,0.28);
                border-radius: 10px;
                padding: 7px 10px;
                selection-background-color: #334155;
            }
            QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
                border: 1px solid #818cf8;
            }
            QComboBox:hover {
                border: 1px solid rgba(148,163,184,0.5);
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 26px;
                border-left: 1px solid rgba(148,163,184,0.18);
                border-top-right-radius: 10px;
                border-bottom-right-radius: 10px;
                background: rgba(30,41,59,0.9);
            }
            QComboBox::down-arrow {
                image: none;
                width: 10px;
                height: 10px;
                border-left: 2px solid #94a3b8;
                border-bottom: 2px solid #94a3b8;
                margin-top: -4px;
            }
            QComboBox QAbstractItemView {
                background: rgba(17,24,39,0.98);
                color: #e2e8f0;
                border: 1px solid rgba(148,163,184,0.25);
                border-radius: 8px;
                selection-background-color: rgba(91,140,255,0.35);
                selection-color: #f8fafc;
                outline: none;
                padding: 4px;
            }
            QComboBox QAbstractItemView::item {
                padding: 6px 10px;
                min-height: 26px;
                border-radius: 4px;
            }
            QComboBox QAbstractItemView::item:hover {
                background: rgba(91,140,255,0.18);
                color: #f8fafc;
            }
            QSpinBox::up-button, QSpinBox::down-button,
            QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
                background: rgba(30,41,59,0.9);
                border: none;
                border-left: 1px solid rgba(148,163,184,0.18);
                width: 20px;
            }
            QSpinBox::up-button, QDoubleSpinBox::up-button {
                border-top-right-radius: 10px;
            }
            QSpinBox::down-button, QDoubleSpinBox::down-button {
                border-bottom-right-radius: 10px;
            }
            QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {
                image: none;
                width: 6px;
                height: 6px;
                border-left: 2px solid #94a3b8;
                border-top: 2px solid #94a3b8;
                margin-bottom: -2px;
            }
            QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
                image: none;
                width: 6px;
                height: 6px;
                border-left: 2px solid #94a3b8;
                border-bottom: 2px solid #94a3b8;
                margin-top: -2px;
            }
            QPushButton {
                background: linear-gradient(180deg, rgba(79,70,229,0.55), rgba(30,41,59,0.92));
                color: #eef2ff;
                border: 1px solid rgba(129,140,248,0.42);
                border-radius: 10px;
                padding: 8px 12px;
                font-weight: 600;
            }
            QPushButton:hover {
                border-color: #a5b4fc;
                color: #ffffff;
            }
            QPushButton:disabled {
                color: #94a3b8;
                background: rgba(30,41,59,0.8);
            }
            QPushButton#PrimaryButton {
                background: linear-gradient(180deg, rgba(56,189,248,0.45), rgba(15,23,42,0.95));
                border-color: rgba(56,189,248,0.4);
            }
            /* ── Crypto8 스타일 원형 아이콘 버튼 (테두리 없음) ── */
            QToolButton#HeaderIconSettings,
            QToolButton#HeaderIconOrder,
            QToolButton#HeaderIconMetrics,
            QToolButton#HeaderIconRefresh,
            QToolButton#IconToggleButton,
            QToolButton#SearchIconButton {
                border-radius: 999px;
                padding: 4px;
                min-width: 38px;
                min-height: 38px;
                border: none;
                background: transparent;
            }
            QToolButton#HeaderIconSettings:hover,
            QToolButton#HeaderIconOrder:hover,
            QToolButton#HeaderIconMetrics:hover,
            QToolButton#HeaderIconRefresh:hover,
            QToolButton#IconToggleButton:hover,
            QToolButton#SearchIconButton:hover {
                background: rgba(123, 232, 195, 0.10);
            }
            QToolButton#HeaderIconSettings:pressed,
            QToolButton#HeaderIconOrder:pressed,
            QToolButton#HeaderIconMetrics:pressed,
            QToolButton#HeaderIconRefresh:pressed {
                background: rgba(123, 232, 195, 0.18);
            }
            QCheckBox {
                color: #e2e8f0;
                spacing: 8px;
            }
            QTabWidget::pane {
                border: 1px solid rgba(148,163,184,0.18);
                border-bottom-left-radius: 14px;
                border-bottom-right-radius: 14px;
                border-top-left-radius: 0px;
                border-top-right-radius: 14px;
                top: 0px;
                background: rgba(15,23,42,0.45);
            }
            QTabBar::tab {
                background: rgba(22,32,50,0.80);
                color: #94a3b8;
                padding: 9px 18px;
                border: 1px solid rgba(148,163,184,0.12);
                border-bottom: none;
                border-top-left-radius: 10px;
                border-top-right-radius: 10px;
                margin-right: 3px;
            }
            QTabBar::tab:selected {
                background: rgba(15,23,42,0.92);
                color: #f8fafc;
                border-color: rgba(148,163,184,0.22);
                border-bottom: 2px solid #5b8cff;
            }
            QTabBar::tab:hover:!selected {
                background: rgba(30,41,59,0.85);
                color: #e2e8f0;
            }
            QTableWidget {
                background: rgba(15,23,42,0.82);
                alternate-background-color: rgba(30,41,59,0.8);
                color: #e2e8f0;
                gridline-color: rgba(148,163,184,0.16);
                selection-background-color: rgba(56,189,248,0.22);
                selection-color: #ffffff;
                border: 0px;
            }
            QHeaderView::section {
                background: rgba(17,24,39,0.95);
                color: #cbd5e1;
                padding: 8px 10px;
                border: none;
                border-right: 1px solid rgba(148,163,184,0.12);
                font-weight: 600;
            }
            QTextEdit {
                background: rgba(15,23,42,0.9);
                color: #cbd5e1;
                border: 1px solid rgba(148,163,184,0.18);
                border-radius: 12px;
                padding: 8px;
                font-family: Menlo, Monaco, Consolas, monospace;
            }

            /* ── BTC 가격 라벨 ── */
            QLabel#BtcPriceLabel {
                color: #fbbf24;
                font-size: 13px;
                font-weight: 900;
                letter-spacing: 0.02em;
                padding: 5px 10px;
                background: rgba(251, 191, 36, 0.12);
                border: 1px solid rgba(251, 191, 36, 0.30);
                border-radius: 10px;
            }

            /* ── 아이콘 버튼 크기 조정 ── */
            QToolButton#HeaderIconSettings,
            QToolButton#HeaderIconOrder,
            QToolButton#HeaderIconMetrics,
            QToolButton#HeaderIconRefresh {
                min-width: 42px;
                min-height: 42px;
            }

            /* ── 주문 카드 ── */
            QFrame#OrderCard {
                background: rgba(15, 23, 42, 0.72);
                border: 1px solid rgba(148, 163, 184, 0.15);
                border-radius: 16px;
            }
            QLabel#OrderCardTitle {
                color: #e2e8f0;
                font-size: 15px;
                font-weight: 700;
            }
            QLabel#OrderFieldLabel {
                color: #94a3b8;
                font-size: 12px;
                font-weight: 600;
            }
            QLabel#OrderPreviewLabel {
                color: #7be8c3;
                font-size: 12px;
                font-weight: 600;
                padding: 6px 10px;
                background: rgba(123, 232, 195, 0.08);
                border-radius: 8px;
                border: 1px solid rgba(123, 232, 195, 0.18);
            }
            QLabel#OrderAutoStatus {
                color: #fbbf24;
                font-size: 12px;
                font-weight: 600;
                padding: 6px 10px;
                background: rgba(251, 191, 36, 0.08);
                border-radius: 8px;
                border: 1px solid rgba(251, 191, 36, 0.18);
            }
            /* 배지 */
            QLabel#OrderBadgeBulk {
                color: #38bdf8;
                background: rgba(56,189,248,0.14);
                border: 1px solid rgba(56,189,248,0.35);
                border-radius: 6px;
                padding: 2px 8px;
                font-size: 11px;
                font-weight: 800;
            }
            QLabel#OrderBadgeAuto {
                color: #fbbf24;
                background: rgba(251,191,36,0.14);
                border: 1px solid rgba(251,191,36,0.35);
                border-radius: 6px;
                padding: 2px 8px;
                font-size: 11px;
                font-weight: 800;
            }
            QLabel#OrderBadgeSingle {
                color: #a78bfa;
                background: rgba(167,139,250,0.14);
                border: 1px solid rgba(167,139,250,0.35);
                border-radius: 6px;
                padding: 2px 8px;
                font-size: 11px;
                font-weight: 800;
            }
            QLabel#OrderBadgeAutoSingle {
                color: #f472b6;
                background: rgba(244,114,182,0.14);
                border: 1px solid rgba(244,114,182,0.35);
                border-radius: 6px;
                padding: 2px 8px;
                font-size: 11px;
                font-weight: 800;
            }
            /* 실행 버튼 */
            QPushButton#OrderExecButton {
                background: linear-gradient(135deg, rgba(56,189,248,0.55), rgba(91,140,255,0.65));
                color: #ffffff;
                border: 1px solid rgba(56,189,248,0.5);
                border-radius: 10px;
                padding: 9px 16px;
                font-size: 13px;
                font-weight: 700;
            }
            QPushButton#OrderExecButton:hover {
                background: linear-gradient(135deg, rgba(56,189,248,0.75), rgba(91,140,255,0.85));
                border-color: rgba(123,232,195,0.6);
            }
            /* 감시 버튼 */
            QPushButton#OrderWatchButton {
                background: rgba(30, 41, 59, 0.9);
                color: #fbbf24;
                border: 1px solid rgba(251,191,36,0.45);
                border-radius: 10px;
                padding: 9px 16px;
                font-size: 13px;
                font-weight: 700;
            }
            QPushButton#OrderWatchButton:hover {
                background: rgba(251,191,36,0.12);
                border-color: rgba(251,191,36,0.7);
            }
            QPushButton#OrderWatchButton:checked {
                background: rgba(251,191,36,0.20);
                border-color: rgba(251,191,36,0.85);
                color: #fde68a;
            }

            /* ── Positions Overview Bar (Crypto8 스타일) ── */
            QFrame#PositionsOverviewBar {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 rgba(7, 16, 28, 0.96),
                    stop:1 rgba(12, 30, 43, 0.92)
                );
                border: 1px solid rgba(123, 232, 195, 0.22);
                border-radius: 16px;
            }
            QLabel#OverviewEyebrow {
                color: #7be8c3;
                font-size: 11px;
                font-weight: 800;
                letter-spacing: 0.1em;
            }
            QLabel#OverviewTotal {
                color: #f8fafc;
                font-size: 36px;
                font-weight: 800;
                letter-spacing: -0.04em;
            }
            QLabel#OverviewSubtitle {
                color: #9eb7cd;
                font-size: 12px;
            }
            QFrame#OverviewChip {
                background: rgba(255, 255, 255, 0.04);
                border: 1px solid rgba(255, 255, 255, 0.07);
                border-radius: 10px;
                min-width: 110px;
            }
            QLabel#OverviewChipTitle {
                color: #9eb7cd;
                font-size: 10px;
                font-weight: 800;
                letter-spacing: 0.06em;
                text-transform: uppercase;
            }
            QLabel#OverviewChipValue {
                color: #f8fafc;
                font-size: 15px;
                font-weight: 700;
            }
            QLabel#OverviewChipSub {
                color: #9eb7cd;
                font-size: 11px;
                font-weight: 600;
            }
            """
        )

    def _load_defaults(self) -> None:
        self.api_key_edit.setText(os.environ.get("BINANCE_API_KEY", ""))
        self.api_secret_edit.setText(os.environ.get("BINANCE_API_SECRET", ""))
        # 시작 시 두 패널 모두 감춤
        self.controls_group.setVisible(False)
        self.settings_toggle_btn.setToolTip("조회 설정 보기")
        self.metrics_box.setVisible(False)
        self.metrics_toggle_btn_icon.setToolTip("핵심 지표 보기")

    def _setup_timer(self) -> None:
        self.timer = QTimer(self)
        self.timer.setInterval(self.interval_spin.value() * 1000)
        self.timer.timeout.connect(self._on_timer)
        self.interval_spin.valueChanged.connect(self._update_timer_interval)
        self.auto_refresh_check.toggled.connect(self._toggle_timer)
        self.timer.start()

    def _update_timer_interval(self, value: int) -> None:
        self.timer.setInterval(max(1, value) * 1000)

    def _toggle_timer(self, enabled: bool) -> None:
        if enabled:
            self.timer.start()
        else:
            self.timer.stop()

    def _on_timer(self) -> None:
        if self.auto_refresh_check.isChecked():
            self.refresh()

    def refresh(self) -> None:
        if self._loading:
            return
        self._loading = True
        self.refresh_btn.setEnabled(False)
        self.state_label.setText("조회 중...")
        self.statusBar().showMessage("바이낸스 선물 데이터 조회 중...")

        # API 키가 변경된 경우 BTC 워커 재시작
        if self._btc_worker is not None:
            cur_key = self.api_key_edit.text().strip()
            if cur_key and cur_key != self._btc_worker._api_key:
                self._btc_worker.stop()
                self._btc_worker = None

        self._worker = SnapshotWorker(
            self.api_key_edit.text().strip(),
            self.api_secret_edit.text().strip(),
            self.key_file_path,
            float(self.shock_min_spin.value()),
            float(self.shock_max_spin.value()),
            float(self.shock_step_spin.value()),
            float(self.preset_shock_spin.value()),
        )
        self._worker.loaded.connect(self._on_refresh_finished)
        self._worker.failed.connect(self._on_refresh_failed)
        self._worker.start()

    def _on_refresh_finished(self, payload: object) -> None:
        self._loading = False
        self.refresh_btn.setEnabled(True)
        data = payload if isinstance(payload, dict) else {}
        self._latest_payload = data

        snapshot = data.get("snapshot", {})
        stress_df = data.get("stress_df", pd.DataFrame())
        dist_df = data.get("dist_df", pd.DataFrame())
        preset_sc = data.get("preset_sc", {})

        summary = snapshot.get("summary")
        pos_df = snapshot.get("pos_df", pd.DataFrame())
        oo_df = snapshot.get("oo_df", pd.DataFrame())
        tr_df = snapshot.get("tr_df", pd.DataFrame())
        collateral = snapshot.get("collateral", {})
        cdf = _collateral_dataframe(collateral)
        realized_sum = float(sum(snapshot.get("realized_7d", {}).values()) or 0.0)

        # BTC 실시간 가격 워커 최초 기동 (성공적인 첫 조회 후 시작)
        if self._btc_worker is None:
            self._btc_worker = BtcPriceWorker(
                self.api_key_edit.text().strip(),
                self.api_secret_edit.text().strip(),
                self.key_file_path,
            )
            self._btc_worker.price_updated.connect(self._on_btc_price)
            self._btc_worker.start()

        self._update_metrics(snapshot, oo_df, tr_df, preset_sc)
        self._update_charts(pos_df, oo_df, cdf)
        self._update_order_symbol_options(snapshot)
        self._set_table(self.positions_table, pos_df, _summary_row_for_positions(pos_df))
        self._set_table(self.orders_table, oo_df, _summary_row_for_open_orders(oo_df))
        self._set_table(self.trades_table, tr_df, _summary_row_for_trades(tr_df))
        self._set_table(self.collateral_table, cdf, _summary_row_for_collateral(cdf))
        self._set_table(self.distance_table, dist_df)
        self._set_table(self.stress_table, stress_df)
        self._process_auto_orders(snapshot)

        last_refresh = pd.Timestamp.now(tz="Asia/Seoul").strftime("%Y-%m-%d %H:%M:%S")
        self.last_refresh_label.setText(f"마지막 갱신: {last_refresh}")
        self.state_label.setText("정상")
        self.error_label.setText("")
        self.statusBar().showMessage("조회 완료")

        lines = [
            f"[갱신 완료] {last_refresh}",
            f"포지션 {len(pos_df)}건 / 미체결 {len(oo_df)}건 / 최근 체결 {len(tr_df)}건",
            f"실현손익 합계(7일): {_fmt_num(realized_sum, 2)} USDT",
        ]
        if summary is not None:
            lines.append(
                f"지갑 {_fmt_num(getattr(summary, 'total_wallet', 0.0), 2)} / "
                f"마진 {_fmt_num(getattr(summary, 'total_margin_balance', 0.0), 2)} / "
                f"가용 {_fmt_num(getattr(summary, 'available_balance', 0.0), 2)}"
            )
        if preset_sc:
            lines.append(
                f"시나리오 {self.preset_shock_spin.value():+.1f}% -> "
                f"위험 {preset_sc.get('risk_label', '-')}, "
                f"순자산 {_fmt_num(preset_sc.get('equity_proxy', 0.0), 2)} USDT"
            )
        self.log_box.setPlainText("\n".join(lines))

        self._append_history(snapshot)
        self.history_panel.update_history(self._history_df)

    def _update_charts(self, pos_df: pd.DataFrame, oo_df: pd.DataFrame, cdf: pd.DataFrame) -> None:
        def _top_pairs(df: pd.DataFrame, name_col: str, value_col: str) -> List[Tuple[str, float]]:
            if df is None or df.empty or name_col not in df.columns or value_col not in df.columns:
                return []
            data = (
                df[[name_col, value_col]]
                .copy()
                .assign(_value=lambda x: pd.to_numeric(x[value_col], errors="coerce").fillna(0.0).abs())
                .sort_values("_value", ascending=False)
                .head(5)
            )
            return [(str(row[name_col]), float(row["_value"])) for _, row in data.iterrows()]

        def _group_by_symbol(df: pd.DataFrame, value_col: str) -> Tuple[List[str], List[float], List[Tuple[str, float]]]:
            if df is None or df.empty or "symbol" not in df.columns or value_col not in df.columns:
                return [], [], []
            grouped = (
                df[["symbol", value_col]]
                .copy()
                .assign(_value=lambda x: pd.to_numeric(x[value_col], errors="coerce").fillna(0.0).abs())
                .groupby("symbol", as_index=False)["_value"]
                .sum()
                .sort_values("_value", ascending=False)
            )
            labels = [str(v) for v in grouped["symbol"].tolist()]
            values = [float(v) for v in grouped["_value"].tolist()]
            top_rows = [(str(row["symbol"]), float(row["_value"])) for _, row in grouped.head(5).iterrows()]
            return labels, values, top_rows

        if pos_df is None or pos_df.empty:
            pos_labels: List[str] = []
            pos_values: List[float] = []
        else:
            pos_labels = [str(v) for v in pos_df.get("symbol", pd.Series(dtype=str)).fillna("-").tolist()]
            pos_values = [abs(float(v)) for v in pos_df.get("notional", pd.Series(dtype=float)).fillna(0.0).tolist()]

        oo_labels, oo_values, oo_top_rows = _group_by_symbol(oo_df, "est_value_usdt")

        if cdf is None or cdf.empty:
            c_labels = []
            c_values = []
        else:
            c_labels = [str(v) for v in cdf.get("asset", pd.Series(dtype=str)).fillna("-").tolist()]
            c_values = [abs(float(v)) for v in cdf.get("total_value", pd.Series(dtype=float)).fillna(0.0).tolist()]

        self.position_pie_panel.update_chart("Position Exposure", pos_labels, pos_values, unit="USDT")
        self.position_pie_panel.update_top5_cards(
            _top_pairs(pos_df, "symbol", "notional"),
            title="Top 4 Positions",
            unit="USDT",
            total=float(pd.to_numeric(pos_df.get("notional", pd.Series(dtype=float)), errors="coerce").fillna(0.0).abs().sum()) if pos_df is not None and not pos_df.empty else 0.0,
        )
        self.order_pie_panel.update_chart("Open Order Exposure", oo_labels, oo_values, unit="USDT")
        self.order_pie_panel.update_top5_cards(
            oo_top_rows,
            title="Top 4 Open Orders",
            unit="USDT",
            total=float(sum(oo_values)) if oo_values else 0.0,
        )
        self.collateral_pie_panel.update_chart("Collateral Allocation", c_labels, c_values, unit="USDT")
        # 담보는 차트만 보여주고 하단 텍스트는 생략

    def _update_metrics(self, snapshot: Dict[str, Any], oo_df: pd.DataFrame, tr_df: pd.DataFrame, preset_sc: Dict[str, Any]) -> None:
        summary = snapshot.get("summary")
        maint_buffer = float(snapshot.get("maint_buffer", 0.0))
        margin_ratio_pct = float(snapshot.get("margin_ratio_pct", 0.0))
        hhi = float(snapshot.get("hhi", 0.0))
        positions = snapshot.get("positions", [])
        if summary is None:
            return

        self.metric_cards["총 지갑 잔고"].set_values(
            f"{_fmt_balance(getattr(summary, 'total_wallet', 0.0))} USDT",
            "계정 wallet balance",
        )
        self.metric_cards["마진 잔고"].set_values(
            f"{_fmt_balance(getattr(summary, 'total_margin_balance', 0.0))} USDT",
            f"유지증거금 비중 {margin_ratio_pct:.2f}%",
        )
        self.metric_cards["가용 잔고"].set_values(
            f"{_fmt_balance(getattr(summary, 'available_balance', 0.0))} USDT",
            "즉시 사용할 수 있는 잔고",
        )
        self.metric_cards["미실현손익"].set_values(
            f"{getattr(summary, 'total_unrealized', 0.0):,.2f} USDT",
            f"7일 실현손익 합 {_fmt_num(sum(snapshot.get('realized_7d', {}).values()), 2)} USDT",
        )
        self.metric_cards["포지션 수"].set_values(
            f"{len(positions)}",
            f"HHI {hhi:.3f}",
        )
        self.metric_cards["미체결 수"].set_values(
            f"{len(oo_df)}",
            f"최근 체결 {len(tr_df)}건",
        )
        self.metric_cards["유지증거금 여유"].set_values(
            f"{_fmt_balance(maint_buffer)} USDT",
            "margin balance - maint margin",
        )
        # Positions Overview 히어로 바 업데이트
        pos_notional_total = float(
            sum(
                abs(float(p.get("notional", 0.0)))
                for p in positions
                if isinstance(p, dict)
            )
        )
        _oo_amount = float(
            pd.to_numeric(oo_df.get("est_value_usdt", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum()
        ) if oo_df is not None and not oo_df.empty else 0.0
        self.overview_bar.update_data(
            total_wallet=float(getattr(summary, "total_wallet", 0.0)),
            total_notional=pos_notional_total,
            positions_count=len(positions),
            orders_count=len(oo_df) if oo_df is not None else 0,
            orders_amount=_oo_amount,
            unrealized_pnl=float(getattr(summary, "total_unrealized", 0.0)),
            margin_ratio_pct=margin_ratio_pct,
        )
        if preset_sc:
            self.metric_cards["시나리오 위험"].set_values(
                str(preset_sc.get("risk_label", "-")),
                f"충격 {_fmt_num(self.preset_shock_spin.value(), 1, signed=True)}% / "
                f"순자산 {_fmt_num(preset_sc.get('equity_proxy', 0.0), 2)} USDT",
            )

    def _append_history(self, snapshot: Dict[str, Any]) -> None:
        summary = snapshot.get("summary")
        if summary is None:
            return
        row = {
            "timestamp": pd.Timestamp.now(tz="Asia/Seoul").isoformat(),
            "total_wallet": float(getattr(summary, "total_wallet", 0.0) or 0.0),
            "total_margin_balance": float(getattr(summary, "total_margin_balance", 0.0) or 0.0),
            "available_balance": float(getattr(summary, "available_balance", 0.0) or 0.0),
            "total_unrealized": float(getattr(summary, "total_unrealized", 0.0) or 0.0),
            "maint_buffer": float(snapshot.get("maint_buffer", 0.0) or 0.0),
            "margin_ratio_pct": float(snapshot.get("margin_ratio_pct", 0.0) or 0.0),
            "positions_count": int(len(snapshot.get("positions", []))),
            "open_orders_count": int(len(snapshot.get("open_orders", []))),
            "realized_7d": float(sum(snapshot.get("realized_7d", {}).values()) or 0.0),
        }
        _append_history_row(row)
        self._history_df = _load_history_df()

    def _set_table(self, table: QTableWidget, df: pd.DataFrame, summary_row: Optional[Dict[str, Any]] = None) -> None:
        _populate_table(table, df, summary_row=summary_row)

    def _on_btc_price(self, price: float) -> None:
        """BtcPriceWorker 에서 3초마다 호출되는 슬롯."""
        self.btc_price_label.setText(f"₿  {price:,.0f}")

    def _on_refresh_failed(self, error: str) -> None:
        self._loading = False
        self.refresh_btn.setEnabled(True)
        self.state_label.setText("오류")
        self.statusBar().showMessage("조회 실패")
        self.log_box.setPlainText(error)
        self.error_label.setText(error.splitlines()[0] if error else "조회 실패")

        lowered = error.lower()
        if "api-key format invalid" in lowered or "-2014" in lowered or "invalid api-key" in lowered:
            self.auto_refresh_check.setChecked(False)
            self.timer.stop()
            self.statusBar().showMessage("API 키 형식 오류로 자동 새로고침을 중지했습니다.")


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("Binance USD-M Futures Monitor")
    app.setStyle("Fusion")
    window = FuturesMonitor()
    window.showMaximized()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
