"""
바이낸스 USDT-M 선물 — 리스크·민감도 리포트 (CLI).

`futures_dashboard/app.py` Streamlit 대시보드와 동일한 data_service 로직을 사용해
터미널에서 요약·테이블·선택적으로 Plotly HTML을 출력합니다.

API 키 우선순위: --api-key / --api-secret → 키 파일(기본 ~/Documents/바이낸스.txt, 1줄 Key·2줄 Secret)
  → HARDCODED_* → 환경변수(BINANCE_API_KEY / BINANCE_API_SECRET). .rtf 경로면 textutil로 변환.

실행 시 기본으로 Plotly 대시보드(원본 Streamlit과 유사한 구성)를 브라우저에서 엽니다.
  --no-browser 로 텍스트만 출력.

그래프: Plotly 차트 3개(심볼 비중 도넛 2 / 누적 체결 / 가격 충격 민감도) 순으로 한 HTML에 표시합니다.
  --no-browser 는 HTML을 열지 않고, --export-html 만 쓸 때 함께 사용하세요.

실행 예 (저장소 루트):
  python -m futures_dashboard.risk_report
  python -m futures_dashboard.risk_report --export-html ./risk_dashboard.html
  python -m futures_dashboard.risk_report --no-browser
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

_HERE = Path(__file__).resolve().parent
_PROJECTS_ROOT = _HERE.parent
_IMPORT_ROOTS = [
    _HERE,
    _PROJECTS_ROOT,
    _PROJECTS_ROOT / "BFut",
]
for _root in _IMPORT_ROOTS:
    if _root.is_dir() and str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from futures_dashboard.data_service import (
    build_stress_curve,
    get_client,
    load_full_snapshot,
    order_distance_stats,
    stress_scenario,
)

# 기본 키 파일: UTF-8 텍스트, 1줄 API Key·2줄 Secret (.rtf면 macOS에서 textutil 변환)
DEFAULT_BINANCE_CREDENTIALS_PATH = Path("/Users/yugjingwan/Documents/바이낸스.txt")

# 로컬 전용 보조: 파일·환경변수가 없을 때만 사용. 저장소에 커밋하지 마세요.
HARDCODED_BINANCE_API_KEY = ""
HARDCODED_BINANCE_API_SECRET = ""


# 세부 내역 HTML 테이블: 헤더 클릭 시 오름/내림차순 정렬
_DETAIL_TABLE_SORT_SCRIPT = r"""
<script>
function bfutShowPanel(id) {
  document.querySelectorAll('.panel-box').forEach(function(el) { el.classList.remove('visible'); });
  document.querySelectorAll('.detail-toolbar button').forEach(function(b) { b.classList.remove('active'); });
  var p = document.getElementById(id);
  if (p) p.classList.add('visible');
  var btn = document.querySelector('.detail-toolbar button[data-panel="' + id + '"]');
  if (btn) btn.classList.add('active');
}

function bfutCellSortVal(td) {
  if (!td) return { t: 's', v: '' };
  var raw = td.textContent.replace(/\u00a0/g, ' ').trim();
  if (raw === '' || raw === '\u2014' || raw === '-') return { t: 's', v: raw.toLowerCase() };
  var s = raw.replace(/,/g, '').trim();
  if (s.charAt(0) === '+') s = s.substring(1);
  if (/^-?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$/.test(s)) {
    var n = parseFloat(s);
    if (!isNaN(n) && isFinite(n)) return { t: 'n', v: n };
  }
  return { t: 's', v: raw.toLowerCase() };
}

function bfutCompare(ax, bx, dir) {
  var a = ax.sortKey, b = bx.sortKey;
  if (a.t === 'n' && b.t === 'n') {
    var d = a.v - b.v;
    if (isNaN(d)) d = 0;
    return dir * d;
  }
  if (a.t === 'n') return -dir;
  if (b.t === 'n') return dir;
  if (a.v < b.v) return -dir;
  if (a.v > b.v) return dir;
  return 0;
}

function bfutInitSortableTables() {
  document.querySelectorAll('table.detail-table').forEach(function(table) {
    var headRow = table.querySelector('thead tr');
    var tbody = table.querySelector('tbody');
    if (!headRow || !tbody) return;
    var ths = headRow.querySelectorAll('th');
    ths.forEach(function(th, colIdx) {
      th.classList.add('sortable-th');
      th.title = '클릭: 오름차순 / 다시 클릭: 내림차순';
      th.addEventListener('click', function(ev) {
        ev.preventDefault();
        var curCol = table.getAttribute('data-sort-col');
        var curDir = table.getAttribute('data-sort-dir');
        var dir = 1;
        if (String(colIdx) === curCol) dir = curDir === 'asc' ? -1 : 1;
        table.setAttribute('data-sort-col', String(colIdx));
        table.setAttribute('data-sort-dir', dir === 1 ? 'asc' : 'desc');
        ths.forEach(function(h) { h.classList.remove('sort-asc', 'sort-desc'); });
        th.classList.add(dir === 1 ? 'sort-asc' : 'sort-desc');

        var rows = Array.prototype.slice.call(tbody.querySelectorAll('tr'));
        var keyed = rows.map(function(tr) {
          var td = tr.children[colIdx];
          return { tr: tr, sortKey: bfutCellSortVal(td) };
        });
        keyed.sort(function(x, y) { return bfutCompare(x, y, dir); });
        keyed.forEach(function(item) { tbody.appendChild(item.tr); });
      });
    });
  });
}

document.addEventListener('DOMContentLoaded', function() {
  bfutInitSortableTables();
});
</script>
"""


def _normalize_labeled_line(line: str) -> str:
    """'API_KEY=...', 'Key: ...', '키: ...' 형태면 값만 남김."""
    s = line.strip()
    for sep in ("=", ":", "："):
        if sep in s:
            left, right = s.split(sep, 1)
            if right.strip():
                return right.strip()
    return s


def _sanitize_binance_credential(raw: Optional[str]) -> Optional[str]:
    """
    RTF/textutil 산출물에 흔한 BOM·제로폭공백·줄바꿈·따옴표 제거.
    바이낸스 키/시크릿은 영숫자만 사용하므로 그 외 문자는 제거.
    """
    if not raw:
        return None
    s = str(raw).strip()
    for ch in ("\ufeff", "\u200b", "\u200c", "\u200d", "\u2060"):
        s = s.replace(ch, "")
    s = "".join(s.split())  # 모든 공백·개행 제거
    s = s.strip("\"'""\u201c\u201d\u2018\u2019")
    s = re.sub(r"[^A-Za-z0-9]", "", s)
    return s or None


def _strip_rtf_to_plain(rtf_text: str) -> str:
    """비 macOS에서 RTF를 대략적인 평문으로만 변환 (textutil 없을 때 폴백)."""
    s = rtf_text

    def _hex_byte(m: re.Match) -> str:
        try:
            return chr(int(m.group(1), 16))
        except ValueError:
            return ""

    s = re.sub(r"\\'([0-9a-f]{2})", _hex_byte, s, flags=re.I)
    s = re.sub(r"\\u(-?\d+)\s*\?", lambda m: chr(int(m.group(1))), s)
    s = re.sub(r"\\[a-z]+\d*\s?", " ", s, flags=re.I)
    s = re.sub(r"[{}]", " ", s)
    s = re.sub(r"\s+", "\n", s)
    return s.strip()


def _load_binance_credentials_from_file(cred_path: Path) -> Tuple[Optional[str], Optional[str]]:
    """텍스트(.txt)는 UTF-8 평문, .rtf는 textutil(macOS) 또는 단순 RTF 제거 후 동일 규칙으로 파싱."""
    if not cred_path.is_file():
        return None, None
    try:
        suffix = cred_path.suffix.lower()
        if suffix == ".txt":
            text = cred_path.read_text(encoding="utf-8-sig", errors="replace").strip()
        elif sys.platform == "darwin":
            r = subprocess.run(
                ["textutil", "-convert", "txt", "-stdout", str(cred_path)],
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if r.returncode != 0:
                return None, None
            text = (r.stdout or "").strip()
        else:
            raw = cred_path.read_bytes()
            try:
                enc = raw.decode("utf-8")
            except UnicodeDecodeError:
                enc = raw.decode("latin-1", errors="replace")
            text = _strip_rtf_to_plain(enc).strip()
        if not text:
            return None, None
        raw_lines = [ln for ln in text.splitlines() if str(ln).strip()]
        tokens: List[str] = []
        for ln in raw_lines:
            norm = _normalize_labeled_line(ln)
            cred = _sanitize_binance_credential(norm)
            if cred and len(cred) >= 16:
                tokens.append(cred)
        if len(tokens) >= 2:
            return tokens[0], tokens[1]
        if len(raw_lines) == 1:
            parts = _normalize_labeled_line(raw_lines[0]).split()
            if len(parts) >= 2:
                k = _sanitize_binance_credential(parts[0])
                sec = _sanitize_binance_credential(parts[1])
                if k and sec:
                    return k, sec
    except Exception:
        return None, None
    return None, None


def _resolve_api_credentials(args: argparse.Namespace) -> Tuple[Optional[str], Optional[str]]:
    key = _sanitize_binance_credential(_normalize_labeled_line(args.api_key or ""))
    secret = _sanitize_binance_credential(_normalize_labeled_line(args.api_secret or ""))
    if not key or not secret:
        path_arg = getattr(args, "binance_key_file", None) or DEFAULT_BINANCE_CREDENTIALS_PATH
        cred_path = Path(path_arg).expanduser().resolve()
        rk, rs = _load_binance_credentials_from_file(cred_path)
        if not key:
            key = rk
        if not secret:
            secret = rs
    if not key:
        key = _sanitize_binance_credential(HARDCODED_BINANCE_API_KEY)
    if not secret:
        secret = _sanitize_binance_credential(HARDCODED_BINANCE_API_SECRET)
    return key, secret


def _chart_recent_trades_cum(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        fig = go.Figure()
        fig.update_layout(title="최근 체결 없음")
        return fig
    key = "time_ms" if "time_ms" in df.columns else "time_local"
    d = df.sort_values(key)
    running: list[float] = []
    total = 0.0
    for _, r in d.iterrows():
        v = float(r["value_usdt"])
        if str(r["side"]).upper() == "BUY":
            total += v
        else:
            total -= v
        running.append(total)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            y=running,
            x=list(range(len(running))),
            mode="lines",
            name="누적 순매수 금액",
            line=dict(color="#38bdf8", width=2),
            fill="tozeroy",
            fillcolor="rgba(56,189,248,0.15)",
        )
    )
    fig.update_layout(
        title="최근 체결 기준 누적 순매수 흐름(시간순 인덱스)",
        paper_bgcolor="rgba(15,23,42,0.95)",
        plot_bgcolor="rgba(15,23,42,0.35)",
        font=dict(color="#cbd5e1"),
        xaxis=dict(gridcolor="rgba(148,163,184,0.12)"),
        yaxis=dict(title="USDT", gridcolor="rgba(148,163,184,0.12)"),
        height=320,
        margin=dict(t=48, b=32),
    )
    return fig


def _chart_stress(df: pd.DataFrame) -> go.Figure:
    """`futures_dashboard/app.py` 와 동일한 가격 충격(이중 Y) 라인 차트."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=df["shock_pct"],
            y=df["equity_proxy"],
            name="추정 순자산(담보+미실현손익)",
            line=dict(color="#818cf8", width=3),
            fill="tozeroy",
            fillcolor="rgba(129,140,248,0.12)",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=df["shock_pct"],
            y=df["total_unrealized_pnl"],
            name="미실현손익",
            line=dict(color="#f472b6", width=2),
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=df["shock_pct"],
            y=df["loss_to_collateral_pct"],
            name="담보 대비 |미실현손익| 비율(%)",
            line=dict(color="#fbbf24", width=2, dash="dot"),
        ),
        secondary_y=True,
    )
    fig.update_xaxes(title_text="가격 충격 (%)", gridcolor="rgba(148,163,184,0.15)", zeroline=True)
    fig.update_yaxes(title_text="USDT", secondary_y=False, gridcolor="rgba(148,163,184,0.12)")
    fig.update_yaxes(title_text="비율 %", secondary_y=True, gridcolor="rgba(148,163,184,0.08)")
    fig.update_layout(
        title=dict(text="가격 충격 민감도 (스윕)", font=dict(size=16, color="#e2e8f0")),
        paper_bgcolor="rgba(15,23,42,0.95)",
        plot_bgcolor="rgba(15,23,42,0.35)",
        font=dict(color="#cbd5e1"),
        legend=dict(orientation="h", yanchor="bottom", y=1.08, x=0, font=dict(size=11)),
        height=460,
        margin=dict(t=56, b=48, l=56, r=56),
    )
    return fig


def _rollup_pie_small_slices(
    labels: List[str],
    values: List[float],
    max_small_pct: float = 2.0,
) -> Tuple[List[str], List[float]]:
    """비중이 max_small_pct 이하인 조각은 합쳐서 '기타' 한 덩어리로 표시 (라벨 밀집 완화)."""
    vals = [max(0.0, float(v)) for v in values]
    total = sum(vals)
    if total <= 1e-12:
        return ["—"], [1.0]
    keep_l: List[str] = []
    keep_v: List[float] = []
    other = 0.0
    for lab, v in zip(labels, vals):
        pct = (v / total) * 100.0
        if v > 0 and pct <= max_small_pct:
            other += v
        else:
            keep_l.append(str(lab))
            keep_v.append(v)
    if other > 1e-12:
        keep_l.append("기타")
        keep_v.append(other)
    if not keep_l:
        return ["기타"], [total]
    return keep_l, keep_v


def build_pies_figure(pos_df: pd.DataFrame) -> go.Figure:
    """원본 대시보드의 도넛 2개(노셔널 / |미실현손익|). 2% 이하는 '기타'로 묶음. 제목에 |노셔널|·|미실현손익| 합계(USDT) 표시."""
    if pos_df.empty:
        tot_n = 0.0
        tot_u = 0.0
        title_l = ("심볼별 |노셔널| 비중 (합계 —)", "심볼별 |미실현손익| 비중 (합계 —)")
    else:
        tot_n = float(pos_df["notional"].abs().sum())
        tot_u = float(pos_df["unrealized_pnl"].abs().sum())
        title_l = (
            f"심볼별 |노셔널| 비중 · 합계 {tot_n:,.2f} USDT",
            f"심볼별 |미실현손익| 비중 · 합계 {tot_u:,.2f} USDT",
        )
    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "domain"}, {"type": "domain"}]],
        subplot_titles=title_l,
        horizontal_spacing=0.04,
    )
    pie_line = dict(color="rgba(15,23,42,0.9)", width=1)
    if pos_df.empty:
        for col in (1, 2):
            fig.add_trace(
                go.Pie(
                    labels=["열린 포지션 없음"],
                    values=[1.0],
                    hole=0.52,
                    textinfo="label",
                    textfont=dict(size=14),
                    marker=dict(line=pie_line),
                ),
                row=1,
                col=col,
            )
    else:
        syms = pos_df["symbol"].tolist()
        abs_n = pos_df["notional"].abs().tolist()
        abs_u = pos_df["unrealized_pnl"].abs().tolist()
        lab_n, val_n = _rollup_pie_small_slices(syms, abs_n, max_small_pct=2.0)
        lab_u, val_u = _rollup_pie_small_slices(syms, abs_u, max_small_pct=2.0)
        fig.add_trace(
            go.Pie(
                labels=lab_n,
                values=val_n,
                hole=0.52,
                textinfo="percent+label",
                textposition="inside",
                insidetextorientation="radial",
                textfont=dict(size=13),
                marker=dict(line=pie_line),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Pie(
                labels=lab_u,
                values=val_u,
                hole=0.52,
                textinfo="percent+label",
                textposition="inside",
                insidetextorientation="radial",
                textfont=dict(size=13),
                marker=dict(line=pie_line),
            ),
            row=1,
            col=2,
        )
    fig.update_layout(
        paper_bgcolor="rgba(15,23,42,0.95)",
        font=dict(color="#cbd5e1", size=14),
        showlegend=False,
        height=680,
        margin=dict(t=88, b=48, l=48, r=48),
    )
    fig.update_annotations(font_size=13)
    return fig


def _collateral_to_dataframe(collateral: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    rows = []
    for name, info in collateral.items():
        rows.append(
            {
                "asset": name,
                "wallet": info["wallet_balance"],
                "price": info["current_price"],
                "value_usdt": info["total_value"],
                "upl_asset": info["unrealized_pnl"],
            }
        )
    return pd.DataFrame(rows)


def _format_position_table_html(pos_df: pd.DataFrame) -> str:
    if pos_df.empty:
        return '<p class="empty-msg">포지션 없음</p>'
    disp = pos_df.copy()
    for c in ["entry", "mark", "liq_price"]:
        if c in disp.columns:
            disp[c] = disp[c].map(lambda x: f"{float(x):,.4f}")
    for c in ["notional", "unrealized_pnl", "realized_7d"]:
        if c in disp.columns:
            disp[c] = disp[c].map(lambda x: f"{float(x):,.2f}")
    return disp.to_html(index=False, classes="detail-table", border=0, escape=True)


def _format_collateral_table_html(cdf: pd.DataFrame) -> str:
    if cdf.empty:
        return '<p class="empty-msg">표시할 자산이 없습니다.</p>'
    d2 = cdf.copy()
    d2["wallet"] = d2["wallet"].map(lambda x: f"{float(x):,.6f}")
    d2["price"] = d2["price"].map(lambda x: f"{float(x):,.4f}")
    d2["value_usdt"] = d2["value_usdt"].map(lambda x: f"{float(x):,.2f}")
    d2["upl_asset"] = d2["upl_asset"].map(lambda x: f"{float(x):,.2f}")
    return d2.to_html(index=False, classes="detail-table", border=0, escape=True)


def _format_dist_table_html(dist_df: pd.DataFrame) -> str:
    if dist_df.empty:
        return '<p class="empty-msg">분석 가능한 리밋 주문이 없습니다.</p>'
    d2 = dist_df.copy()
    if "limit" in d2.columns:
        d2["limit"] = d2["limit"].map(lambda x: f"{float(x):,.4f}")
    if "mark" in d2.columns:
        d2["mark"] = d2["mark"].map(lambda x: f"{float(x):,.4f}")
    if "dist_pct" in d2.columns:
        d2["dist_pct"] = d2["dist_pct"].map(lambda x: f"{float(x):+.2f}")
    return d2.to_html(index=False, classes="detail-table", border=0, escape=True)


def _format_stress_table_html(stress_df: pd.DataFrame) -> str:
    sf = stress_df.copy()
    for c in [
        "equity_proxy",
        "total_unrealized_pnl",
        "collateral_usdt_equiv",
        "loss_to_collateral_pct",
        "total_position_value",
    ]:
        if c in sf.columns:
            sf[c] = sf[c].map(lambda x: f"{float(x):,.2f}")
    return sf.to_html(index=False, classes="detail-table", border=0, escape=True)


def _summary_box_pos(pos_df: pd.DataFrame) -> str:
    if pos_df.empty:
        return '<div class="detail-summary muted"><strong>합계</strong> 포지션 없음</div>'
    n = len(pos_df)
    abs_n = float(pos_df["notional"].abs().sum()) if "notional" in pos_df.columns else 0.0
    su = float(pos_df["unrealized_pnl"].sum()) if "unrealized_pnl" in pos_df.columns else 0.0
    sr = float(pos_df["realized_7d"].sum()) if "realized_7d" in pos_df.columns else 0.0
    parts = (
        f'<span class="sum-item"><span class="k">건수</span> <span class="v">{n}</span></span>'
        f'<span class="sum-sep">·</span>'
        f'<span class="sum-item"><span class="k">|노셔널| 합</span> <span class="v">{abs_n:,.2f}</span> USDT</span>'
        f'<span class="sum-sep">·</span>'
        f'<span class="sum-item"><span class="k">미실현손익 합</span> <span class="v">{su:,.2f}</span> USDT</span>'
        f'<span class="sum-sep">·</span>'
        f'<span class="sum-item"><span class="k">7일 실현 합</span> <span class="v">{sr:,.2f}</span> USDT</span>'
    )
    return f'<div class="detail-summary"><strong>합계</strong> {parts}</div>'


def _summary_box_oo(oo_df: pd.DataFrame, oo_est_sum: float) -> str:
    if oo_df.empty:
        return '<div class="detail-summary muted"><strong>합계</strong> 미체결 없음</div>'
    n = len(oo_df)
    parts = (
        f'<span class="sum-item"><span class="k">건수</span> <span class="v">{n}</span></span>'
        f'<span class="sum-sep">·</span>'
        f'<span class="sum-item"><span class="k">추정 명목 합</span> <span class="v">{oo_est_sum:,.2f}</span> USDT</span>'
    )
    return f'<div class="detail-summary"><strong>합계</strong> {parts}</div>'


def _summary_box_tr(tr_df: pd.DataFrame) -> str:
    if tr_df.empty:
        return '<div class="detail-summary muted"><strong>합계</strong> 체결 없음</div>'
    n = len(tr_df)
    sv = float(tr_df["value_usdt"].sum()) if "value_usdt" in tr_df.columns else 0.0
    sr = float(tr_df["realized_pnl"].sum()) if "realized_pnl" in tr_df.columns else 0.0
    parts = (
        f'<span class="sum-item"><span class="k">건수</span> <span class="v">{n}</span></span>'
        f'<span class="sum-sep">·</span>'
        f'<span class="sum-item"><span class="k">체결 금액 합</span> <span class="v">{sv:,.2f}</span> USDT</span>'
        f'<span class="sum-sep">·</span>'
        f'<span class="sum-item"><span class="k">실현손익 합</span> <span class="v">{sr:,.2f}</span> USDT</span>'
    )
    return f'<div class="detail-summary"><strong>합계</strong> {parts}</div>'


def _summary_box_collateral(cdf: pd.DataFrame) -> str:
    if cdf.empty:
        return '<div class="detail-summary muted"><strong>합계</strong> 자산 없음</div>'
    n = len(cdf)
    sv = float(cdf["value_usdt"].sum()) if "value_usdt" in cdf.columns else 0.0
    su = float(cdf["upl_asset"].sum()) if "upl_asset" in cdf.columns else 0.0
    parts = (
        f'<span class="sum-item"><span class="k">자산 종</span> <span class="v">{n}</span></span>'
        f'<span class="sum-sep">·</span>'
        f'<span class="sum-item"><span class="k">USDT 환산 합</span> <span class="v">{sv:,.2f}</span> USDT</span>'
        f'<span class="sum-sep">·</span>'
        f'<span class="sum-item"><span class="k">자산별 미실현손익 합</span> <span class="v">{su:,.2f}</span> USDT</span>'
    )
    return f'<div class="detail-summary"><strong>합계</strong> {parts}</div>'


def _summary_box_dist(dist_df: pd.DataFrame) -> str:
    if dist_df.empty:
        return '<div class="detail-summary muted"><strong>합계</strong> 해당 리밋 주문 없음</div>'
    n = len(dist_df)
    if "dist_pct" in dist_df.columns:
        avg = float(dist_df["dist_pct"].mean())
        parts = (
            f'<span class="sum-item"><span class="k">건수</span> <span class="v">{n}</span></span>'
            f'<span class="sum-sep">·</span>'
            f'<span class="sum-item"><span class="k">거리(%) 평균</span> <span class="v">{avg:+.2f}</span>%</span>'
        )
    else:
        parts = f'<span class="sum-item"><span class="k">건수</span> <span class="v">{n}</span></span>'
    return f'<div class="detail-summary"><strong>합계</strong> {parts}</div>'


def _summary_box_stress(stress_df: pd.DataFrame) -> str:
    if stress_df.empty:
        return '<div class="detail-summary muted"><strong>합계</strong> 스윕 데이터 없음</div>'
    n = len(stress_df)
    if "shock_pct" in stress_df.columns:
        smin = float(stress_df["shock_pct"].min())
        smax = float(stress_df["shock_pct"].max())
        parts = (
            f'<span class="sum-item"><span class="k">시나리오 행</span> <span class="v">{n}</span></span>'
            f'<span class="sum-sep">·</span>'
            f'<span class="sum-item"><span class="k">충격 구간</span> <span class="v">{smin:+.1f}</span>% ~ <span class="v">{smax:+.1f}</span>%</span>'
        )
    else:
        parts = f'<span class="sum-item"><span class="k">행 수</span> <span class="v">{n}</span></span>'
    return f'<div class="detail-summary"><strong>합계</strong> {parts}</div>'


def _build_detail_section_html(
    stress_df: pd.DataFrame,
    pos_df: pd.DataFrame,
    oo_df: pd.DataFrame,
    tr_df: pd.DataFrame,
    collateral: Dict[str, Dict[str, float]],
    dist_df: pd.DataFrame,
    oo_est_sum: float,
) -> str:
    pos_h = _format_position_table_html(pos_df)
    oo_h = (
        '<p class="empty-msg">미체결 없음</p>'
        if oo_df.empty
        else oo_df.to_html(index=False, classes="detail-table", border=0, escape=True)
    )
    cdf = _collateral_to_dataframe(collateral)
    pos_sum = _summary_box_pos(pos_df)
    oo_sum = _summary_box_oo(oo_df, oo_est_sum)
    tr_sum = _summary_box_tr(tr_df)
    col_sum = _summary_box_collateral(cdf)
    dist_sum = _summary_box_dist(dist_df)
    stress_sum = _summary_box_stress(stress_df)
    tr_h = (
        '<p class="empty-msg">체결 내역 없음</p>'
        if tr_df.empty
        else tr_df.to_html(index=False, classes="detail-table", border=0, escape=True)
    )
    col_h = _format_collateral_table_html(cdf)
    dist_h = _format_dist_table_html(dist_df)
    stress_h = _format_stress_table_html(stress_df)

    return f"""
<section class="detail-section">
  <h2 class="section-title">세부 내역</h2>
  <p class="hint">버튼으로 패널을 고르고, 각 표의 <strong>열 제목(헤더)</strong>을 누르면 그 열 기준 오름차순·내림차순이 번갈아 적용됩니다. (스윕 표 포함)</p>
  <div class="detail-toolbar">
    <button type="button" class="active" data-panel="panel-pos" onclick="bfutShowPanel('panel-pos')">포지션</button>
    <button type="button" data-panel="panel-oo" onclick="bfutShowPanel('panel-oo')">미체결</button>
    <button type="button" data-panel="panel-tr" onclick="bfutShowPanel('panel-tr')">최근 체결</button>
    <button type="button" data-panel="panel-col" onclick="bfutShowPanel('panel-col')">담보 자산</button>
    <button type="button" data-panel="panel-dist" onclick="bfutShowPanel('panel-dist')">주문–마크 거리</button>
  </div>
  <div id="panel-pos" class="panel-box visible"><h3>포지션 상세</h3>{pos_sum}{pos_h}</div>
  <div id="panel-oo" class="panel-box"><h3>미체결 주문</h3>{oo_sum}{oo_h}</div>
  <div id="panel-tr" class="panel-box"><h3>최근 약 3일 체결</h3>{tr_sum}{tr_h}</div>
  <div id="panel-col" class="panel-box"><h3>담보(지갑) 자산</h3>{col_sum}{col_h}</div>
  <div id="panel-dist" class="panel-box"><h3>지정가 주문 vs 마크 (%)</h3>{dist_sum}
    <p class="caption">매수: 마크가 리밋보다 낮을수록 음수(아래에 걸림). 매도: 반대.</p>{dist_h}</div>
  <details class="stress-details">
    <summary>시나리오별 수치 테이블 (전체 스윕)</summary>
    {stress_sum}
    <div class="stress-table-wrap">{stress_h}</div>
  </details>
</section>
""" + _DETAIL_TABLE_SORT_SCRIPT


def build_dashboard_html_document(
    stress_df: pd.DataFrame,
    pos_df: pd.DataFrame,
    tr_df: pd.DataFrame,
    oo_df: pd.DataFrame,
    collateral: Dict[str, Dict[str, float]],
    open_orders: List[Dict[str, Any]],
    marks: Dict[str, float],
) -> str:
    """
    Figure 3개: 비중 도넛 → 누적 체결 → 충격 민감도 순. 단일 HTML + 세부 표.
    """
    dist_df = order_distance_stats(open_orders, marks)
    oo_est = (
        float(oo_df["est_value_usdt"].sum())
        if not oo_df.empty and "est_value_usdt" in oo_df.columns
        else 0.0
    )
    detail_html = _build_detail_section_html(
        stress_df, pos_df, oo_df, tr_df, collateral, dist_df, oo_est
    )

    f_stress = _chart_stress(stress_df)
    f_pies = build_pies_figure(pos_df)
    f_trades = _chart_recent_trades_cum(tr_df)
    plot_cfg = {"displayModeBar": True, "responsive": True, "scrollZoom": True}

    chunks: list[str] = []
    chunks.append("<!DOCTYPE html>\n<html lang=\"ko\">\n<head>\n")
    chunks.append('<meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>\n')
    chunks.append("<title>Binance Futures Risk — BFut</title>\n<style>")
    chunks.append(
        "body{margin:0;background:#0f172a;color:#cbd5e1;"
        "font-family:system-ui,-apple-system,'Segoe UI',sans-serif;}"
        ".wrap{max-width:1280px;margin:0 auto;padding:20px 18px 48px;}"
        ".hero{background:linear-gradient(120deg,#0f172a 0%,#1e293b 42%,#312e81 100%);"
        "border-radius:18px;padding:22px 26px;margin-bottom:22px;border:1px solid rgba(99,102,241,0.28);"
        "box-shadow:0 12px 40px rgba(49,46,129,0.22);}"
        ".hero h1{color:#f8fafc;margin:0 0 8px;font-size:1.5rem;font-weight:700;letter-spacing:-0.02em;}"
        ".hero p{color:#cbd5e1;margin:0;line-height:1.5;font-size:0.95rem;}"
        ".chart{background:rgba(30,41,59,0.4);border-radius:14px;padding:10px 6px 16px;margin-bottom:28px;"
        "border:1px solid rgba(148,163,184,0.2);}"
        ".chart-pies{min-height:700px;padding:16px 12px 24px;}"
        ".chart-pies .js-plotly-plot,.chart-pies .plotly-graph-div{min-height:640px !important;}"
        ".section-title{font-size:1.15rem;color:#e2e8f0;margin:32px 0 8px;font-weight:600;}"
        ".hint{color:#94a3b8;font-size:0.88rem;margin:0 0 14px;}"
        ".detail-toolbar{display:flex;flex-wrap:wrap;gap:10px;margin-bottom:16px;}"
        ".detail-toolbar button{padding:10px 18px;border-radius:10px;cursor:pointer;font-weight:600;"
        "border:1px solid rgba(99,102,241,0.45);"
        "background:linear-gradient(180deg,rgba(79,70,229,0.35),rgba(30,27,75,0.55));color:#e0e7ff;"
        "font-size:0.92rem;}"
        ".detail-toolbar button:hover{border-color:#a5b4fc;color:#fff;}"
        ".detail-toolbar button.active{border-color:#a5b4fc;background:rgba(99,102,241,0.4);color:#fff;}"
        ".panel-box{display:none;background:rgba(15,23,42,0.65);border-radius:12px;padding:16px 14px 20px;"
        "margin-bottom:20px;border:1px solid rgba(148,163,184,0.2);overflow-x:auto;}"
        ".panel-box.visible{display:block;}"
        ".panel-box h3{margin:0 0 10px;font-size:1.05rem;color:#e2e8f0;font-weight:600;}"
        ".detail-summary{background:rgba(51,65,85,0.45);border:1px solid rgba(129,140,248,0.25);"
        "border-radius:10px;padding:10px 14px;margin:0 0 14px;font-size:0.9rem;line-height:1.55;color:#cbd5e1;}"
        ".detail-summary strong{color:#e2e8f0;margin-right:8px;font-weight:600;}"
        ".detail-summary.muted{opacity:0.9;color:#94a3b8;}"
        ".detail-summary .sum-item{white-space:nowrap;}"
        ".detail-summary .k{color:#94a3b8;font-size:0.86em;margin-right:4px;}"
        ".detail-summary .v{color:#f1f5f9;font-weight:600;}"
        ".detail-summary .sum-sep{margin:0 6px;color:#64748b;}"
        ".stress-details .detail-summary{margin-top:4px;margin-bottom:12px;}"
        ".empty-msg{color:#94a3b8;margin:8px 0;}"
        ".caption{color:#94a3b8;font-size:0.88rem;margin:8px 0 12px;line-height:1.45;}"
        "table.detail-table{border-collapse:collapse;width:100%;font-size:0.88rem;}"
        "table.detail-table th,table.detail-table td{padding:8px 10px;text-align:left;"
        "border-bottom:1px solid rgba(148,163,184,0.18);}"
        "table.detail-table th{color:#94a3b8;font-weight:600;white-space:nowrap;}"
        "table.detail-table th.sortable-th{cursor:pointer;user-select:none;position:relative;}"
        "table.detail-table th.sortable-th:hover{background:rgba(99,102,241,0.14);color:#e2e8f0;}"
        "table.detail-table th.sort-asc::after{content:' \\25b2';font-size:0.55em;margin-left:5px;opacity:0.9;vertical-align:middle;}"
        "table.detail-table th.sort-desc::after{content:' \\25bc';font-size:0.55em;margin-left:5px;opacity:0.9;vertical-align:middle;}"
        "table.detail-table tr:hover td{background:rgba(51,65,85,0.35);}"
        ".stress-details{margin-top:20px;background:rgba(30,41,59,0.45);border-radius:12px;"
        "padding:12px 16px;border:1px solid rgba(148,163,184,0.2);}"
        ".stress-details summary{cursor:pointer;color:#e2e8f0;font-weight:600;padding:6px 0;}"
        ".stress-table-wrap{margin-top:12px;overflow-x:auto;}"
        ".footnote{color:#64748b;font-size:0.82rem;margin-top:28px;line-height:1.5;}"
        "</style>\n</head>\n<body>\n<div class=\"wrap\">\n"
    )
    chunks.append(
        '<div class="hero"><h1>선물 계정 리스크 & 민감도</h1>'
        "<p>가격 충격은 <b>모든 심볼 마크 가격</b>과 <b>비-USDT 담보</b>에 동일 비율을 가정한 단순 시나리오입니다.</p>"
        "</div>\n"
    )

    chunks.append('<div class="chart chart-pies">')
    chunks.append(pio.to_html(f_pies, full_html=False, include_plotlyjs="cdn", config=plot_cfg))
    chunks.append('</div>\n<div class="chart">')
    chunks.append(pio.to_html(f_trades, full_html=False, include_plotlyjs=False, config=plot_cfg))
    chunks.append('</div>\n<div class="chart">')
    chunks.append(pio.to_html(f_stress, full_html=False, include_plotlyjs=False, config=plot_cfg))
    chunks.append("</div>\n")
    chunks.append(detail_html)
    chunks.append(
        '<p class="footnote">본 도구는 참고용이며, 실제 강제청산·증거금은 거래소 규칙·교차/격리·MMR 등으로 달라질 수 있습니다.</p>'
        "</div>\n</body>\n</html>"
    )
    return "".join(chunks)


def _write_html_sync(path: Path, html: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(html)
        f.flush()
        os.fsync(f.fileno())


def _open_html_file(path: Path) -> None:
    """OS 기본 브라우저로 HTML을 바로 연다 (macOS `open` 등)."""
    path = path.resolve()
    p = str(path)
    try:
        if sys.platform == "darwin":
            subprocess.run(["open", p], check=False, timeout=60)
        elif os.name == "nt":
            os.startfile(p)  # type: ignore[attr-defined]
        else:
            subprocess.run(["xdg-open", p], check=False, timeout=60)
    except Exception:
        webbrowser.open(path.as_uri())


def open_dashboard_in_browser(html: str) -> str:
    fd, path = tempfile.mkstemp(suffix=".html", prefix="bfut_risk_dashboard_")
    os.close(fd)
    tmp = Path(path)
    _write_html_sync(tmp, html)
    _open_html_file(tmp)
    return str(tmp.resolve())


def _coerce_bool(raw: Any, default: bool = False) -> bool:
    if raw is None:
        return default
    if isinstance(raw, bool):
        return raw
    s = str(raw).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _coerce_float(raw: Any, default: float) -> float:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)


def _build_shocks(shock_min: float, shock_max: float, shock_step: float) -> List[float]:
    lo = float(shock_min)
    hi = float(shock_max)
    step = abs(float(shock_step)) or 2.5
    if hi < lo:
        lo, hi = hi, lo
    shocks: List[float] = []
    x = lo
    while x <= hi + 1e-9:
        shocks.append(round(x, 6))
        x += step
    return shocks or [0.0]


def _make_runtime_options(
    api_key: Optional[str],
    api_secret: Optional[str],
    binance_key_file: Optional[str],
    simulate_fills: bool,
    shock_min: float,
    shock_max: float,
    shock_step: float,
    preset: float,
) -> argparse.Namespace:
    return argparse.Namespace(
        api_key=api_key,
        api_secret=api_secret,
        binance_key_file=binance_key_file or str(DEFAULT_BINANCE_CREDENTIALS_PATH),
        simulate_fills=simulate_fills,
        shock_min=shock_min,
        shock_max=shock_max,
        shock_step=shock_step,
        preset=preset,
    )


def _build_kpi_cards_html(snapshot: Dict[str, Any], sc: Dict[str, Any], params: Dict[str, Any]) -> str:
    summary = snapshot["summary"]
    pos_df = snapshot["pos_df"]
    tr_df = snapshot["tr_df"]
    parts = [
        ("총 마진 잔고", f"{summary.total_margin_balance:,.2f} USDT", "계정 전체 마진 기준"),
        ("가용 잔고", f"{summary.available_balance:,.2f} USDT", "즉시 사용 가능한 증거금"),
        ("유지증거금 여유", f"{snapshot['maint_buffer']:,.2f} USDT", "낮을수록 리스크 확대"),
        ("미실현손익", f"{summary.total_unrealized:,.2f} USDT", "현재 열린 포지션 기준"),
        ("포지션 수", f"{len(snapshot['positions'])}", f"HHI {snapshot['hhi']:.3f}"),
        ("최근 체결", f"{len(tr_df)}건", f"약 3일 누적 · preset {params['preset']:+.1f}%"),
        ("프리셋 위험도", str(sc["risk_label"]), f"담보 대비 손익 {sc['loss_to_collateral_pct']:.2f}%"),
        ("프리셋 순자산", f"{sc['equity_proxy']:,.2f} USDT", "담보 + 미실현손익 단순 추정"),
    ]
    chunks = []
    for title, value, meta in parts:
        chunks.append(
            f'<article class="metric-card"><span class="metric-label">{title}</span>'
            f'<strong class="metric-value">{value}</strong>'
            f'<small class="metric-meta">{meta}</small></article>'
        )
    if pos_df.empty:
        chunks.append(
            '<article class="metric-card metric-card-muted"><span class="metric-label">상태</span>'
            '<strong class="metric-value">포지션 없음</strong>'
            '<small class="metric-meta">열린 선물 포지션이 없습니다.</small></article>'
        )
    return "".join(chunks)


def _build_ops_rail_html(snapshot: Dict[str, Any], sc: Dict[str, Any]) -> str:
    oo_df = snapshot["oo_df"]
    tr_df = snapshot["tr_df"]
    pos_df = snapshot["pos_df"]
    return (
        '<div class="ops-status-rail">'
        f'<div class="ops-pill"><span>Open Positions</span><strong>{len(pos_df)}</strong></div>'
        f'<div class="ops-pill"><span>Open Orders</span><strong>{len(oo_df)}</strong></div>'
        f'<div class="ops-pill"><span>Recent Trades</span><strong>{len(tr_df)}</strong></div>'
        f'<div class="ops-pill"><span>Preset Risk</span><strong>{sc["risk_label"]}</strong></div>'
        "</div>"
    )


def _build_overview_table_html(snapshot: Dict[str, Any], sc: Dict[str, Any], params: Dict[str, Any]) -> str:
    summary = snapshot["summary"]
    rows = [
        ("지갑 잔고", f"{summary.total_wallet:,.2f} USDT"),
        ("마진 잔고", f"{summary.total_margin_balance:,.2f} USDT"),
        ("가용 잔고", f"{summary.available_balance:,.2f} USDT"),
        ("유지증거금 비중", f"{snapshot['margin_ratio_pct']:.2f}%"),
        ("7일 실현손익 합", f"{sum(snapshot['realized_7d'].values()):,.2f} USDT"),
        ("프리셋 충격", f"{params['preset']:+.1f}%"),
        ("프리셋 포지션 가치", f"{sc['total_position_value']:,.2f} USDT"),
        ("프리셋 담보 환산", f"{sc['collateral_usdt_equiv']:,.2f} USDT"),
    ]
    body = "".join(
        f"<tr><th>{k}</th><td>{v}</td></tr>"
        for k, v in rows
    )
    return (
        '<section class="card overview-card"><div class="card-head">'
        '<div><p class="section-eyebrow">Overview</p><h2>계정 요약</h2></div>'
        '<p>CLI 리포트의 핵심 수치를 웹 카드로 재배치했습니다.</p></div>'
        f'<table class="overview-table"><tbody>{body}</tbody></table></section>'
    )


def _build_web_payload(options: argparse.Namespace) -> Dict[str, Any]:
    ak, sk = _resolve_api_credentials(options)
    client = get_client(api_key=ak, api_secret=sk)
    snapshot = load_full_snapshot(client)
    shocks = _build_shocks(options.shock_min, options.shock_max, options.shock_step)
    stress_df = build_stress_curve(
        shocks,
        snapshot["positions"],
        snapshot["open_orders"],
        snapshot["mark_prices"],
        snapshot["collateral"],
        options.simulate_fills,
    )
    sc = stress_scenario(
        float(options.preset),
        snapshot["positions"],
        snapshot["open_orders"],
        snapshot["mark_prices"],
        snapshot["collateral"],
        options.simulate_fills,
    )
    dist_df = order_distance_stats(snapshot["open_orders"], snapshot["mark_prices"])
    oo_est = (
        float(snapshot["oo_df"]["est_value_usdt"].sum())
        if not snapshot["oo_df"].empty and "est_value_usdt" in snapshot["oo_df"].columns
        else 0.0
    )
    plot_cfg = {"displayModeBar": True, "responsive": True, "scrollZoom": True}
    charts = {
        "pies": pio.to_html(build_pies_figure(snapshot["pos_df"]), full_html=False, include_plotlyjs="cdn", config=plot_cfg),
        "trades": pio.to_html(_chart_recent_trades_cum(snapshot["tr_df"]), full_html=False, include_plotlyjs=False, config=plot_cfg),
        "stress": pio.to_html(_chart_stress(stress_df), full_html=False, include_plotlyjs=False, config=plot_cfg),
    }
    params = {
        "simulate_fills": bool(options.simulate_fills),
        "shock_min": float(options.shock_min),
        "shock_max": float(options.shock_max),
        "shock_step": float(options.shock_step),
        "preset": float(options.preset),
    }
    return {
        "generated_at": pd.Timestamp.utcnow().tz_localize("UTC").tz_convert("Asia/Seoul").strftime("%Y-%m-%d %H:%M:%S KST"),
        "params": params,
        "charts": charts,
        "html": {
            "metrics": _build_kpi_cards_html(snapshot, sc, params),
            "ops_rail": _build_ops_rail_html(snapshot, sc),
            "overview": _build_overview_table_html(snapshot, sc, params),
            "positions_table": _format_position_table_html(snapshot["pos_df"]),
            "orders_table": (
                '<p class="empty-msg">미체결 없음</p>'
                if snapshot["oo_df"].empty
                else snapshot["oo_df"].to_html(index=False, classes="detail-table", border=0, escape=True)
            ),
            "trades_table": (
                '<p class="empty-msg">체결 내역 없음</p>'
                if snapshot["tr_df"].empty
                else snapshot["tr_df"].to_html(index=False, classes="detail-table", border=0, escape=True)
            ),
            "collateral_table": _format_collateral_table_html(_collateral_to_dataframe(snapshot["collateral"])),
            "distance_table": _format_dist_table_html(dist_df),
            "stress_table": _format_stress_table_html(stress_df),
            "positions_summary": _summary_box_pos(snapshot["pos_df"]),
            "orders_summary": _summary_box_oo(snapshot["oo_df"], oo_est),
            "trades_summary": _summary_box_tr(snapshot["tr_df"]),
            "collateral_summary": _summary_box_collateral(_collateral_to_dataframe(snapshot["collateral"])),
            "distance_summary": _summary_box_dist(dist_df),
            "stress_summary": _summary_box_stress(stress_df),
        },
        "dashboard_html": build_dashboard_html_document(
            stress_df,
            snapshot["pos_df"],
            snapshot["tr_df"],
            snapshot["oo_df"],
            snapshot["collateral"],
            snapshot["open_orders"],
            snapshot["mark_prices"],
        ),
    }


def _build_web_app_html() -> str:
    return """<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>BFut Risk Monitor</title>
  <style>
    :root{
      color:#ecf2ff;
      background:#060a14;
      --c8-radius-sm:8px;
      --c8-radius-md:12px;
      --c8-radius-lg:16px;
      --c8-space-1:6px;
      --c8-space-2:10px;
      --c8-space-3:16px;
      --c8-space-4:22px;
      --c8-accent:#7be8c3;
      --c8-accent-2:#5b8cff;
      --c8-accent-3:#ff8cc6;
      --c8-text-muted:#9eb7cd;
      --c8-border:rgba(126,146,204,0.26);
      --c8-card:rgba(9,18,31,0.82);
    }
    *{box-sizing:border-box}
    body{
      margin:0;
      min-height:100vh;
      color:#ecf2ff;
      font-family:Inter,system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
      background:
        radial-gradient(circle at 10% 10%, rgba(42,28,77,0.88) 0%, transparent 32%),
        radial-gradient(circle at 85% 18%, rgba(16,82,96,0.55) 0%, transparent 28%),
        linear-gradient(180deg, #060a14 0%, #07111f 100%);
    }
    .container{max-width:1380px;margin:0 auto;padding:24px 18px 42px}
    .card{
      background:var(--c8-card);
      border:1px solid var(--c8-border);
      border-radius:var(--c8-radius-lg);
      box-shadow:0 18px 46px rgba(0,0,0,0.24);
      backdrop-filter:blur(18px);
    }
    .top-header{
      margin-top:0;
      display:grid;
      grid-template-columns:auto 1fr auto;
      gap:18px;
      align-items:center;
      padding:12px 14px;
      position:sticky;
      top:8px;
      z-index:20;
      background:
        linear-gradient(135deg, rgba(7,16,28,0.92), rgba(12,30,43,0.88)),
        radial-gradient(circle at 0% 0%, rgba(123,232,195,0.16), transparent 36%);
    }
    .brand-subtitle{
      color:var(--c8-accent);
      font-size:0.72rem;
      font-weight:900;
      letter-spacing:0.1em;
      text-transform:uppercase;
    }
    .brand-title{font-size:1.18rem;font-weight:950;letter-spacing:-0.04em}
    .header-meta{justify-self:end;color:var(--c8-text-muted);font-size:0.83rem}
    .hero{
      margin-top:18px;
      padding:22px 24px;
      display:grid;
      grid-template-columns:1.6fr 1fr;
      gap:18px;
      align-items:start;
      background:
        linear-gradient(135deg, rgba(6,16,27,0.96), rgba(10,33,45,0.9)),
        radial-gradient(circle at 100% 0%, rgba(123,232,195,0.16), transparent 32%);
    }
    .hero h1{margin:0 0 8px;font-size:2rem;line-height:1.05;letter-spacing:-0.05em}
    .hero p{margin:0;color:#c7d8e7;line-height:1.55}
    .hero-notes{
      display:grid;
      gap:10px;
      padding:14px;
      border-radius:14px;
      border:1px solid rgba(123,232,195,0.18);
      background:rgba(7,20,31,0.7);
    }
    .hero-note{display:grid;gap:2px}
    .hero-note span{font-size:0.72rem;color:var(--c8-accent);text-transform:uppercase;letter-spacing:0.08em;font-weight:800}
    .hero-note strong{font-size:0.95rem}
    .controls-card{margin-top:18px;padding:18px}
    .controls-head{display:flex;justify-content:space-between;gap:16px;align-items:flex-start;margin-bottom:14px}
    .controls-head h2,.section-head h2,.card-head h2,.panel-box h3{margin:0;font-size:1.08rem}
    .section-eyebrow{
      margin:0 0 4px;
      color:var(--c8-accent);
      font-size:0.74rem;
      font-weight:900;
      letter-spacing:0.08em;
      text-transform:uppercase;
    }
    .controls-grid{
      display:grid;
      grid-template-columns:repeat(6, minmax(0, 1fr));
      gap:12px;
      align-items:end;
    }
    .field{display:grid;gap:6px}
    .field label{font-size:0.78rem;color:var(--c8-text-muted);font-weight:700}
    .field input,.field select{
      width:100%;
      border-radius:10px;
      border:1px solid rgba(126,146,204,0.48);
      background:rgba(8,17,28,0.92);
      color:#ecf2ff;
      padding:11px 12px;
      font:inherit;
    }
    .checkline{
      display:flex;
      align-items:center;
      gap:10px;
      min-height:44px;
      padding:0 12px;
      border-radius:10px;
      border:1px solid rgba(126,146,204,0.48);
      background:rgba(8,17,28,0.92);
    }
    .button-row{display:flex;gap:10px;flex-wrap:wrap}
    button{
      border:1px solid rgba(123,232,195,0.24);
      background:linear-gradient(180deg, rgba(23,94,74,0.98), rgba(9,49,46,0.98));
      color:#ecfffb;
      padding:11px 14px;
      border-radius:12px;
      font:inherit;
      font-weight:800;
      cursor:pointer;
    }
    button.secondary{
      border-color:rgba(91,140,255,0.28);
      background:linear-gradient(180deg, rgba(33,52,105,0.92), rgba(18,31,66,0.96));
      color:#dbe8ff;
    }
    button.ghost{
      background:rgba(18,33,55,0.9);
      border-color:rgba(126,146,204,0.48);
      color:#d7e6ff;
    }
    .status-line{margin-top:12px;color:var(--c8-text-muted);font-size:0.84rem}
    .ops-status-rail{
      margin-top:18px;
      display:grid;
      grid-template-columns:repeat(4, minmax(0, 1fr));
      gap:12px;
    }
    .ops-pill{
      padding:14px 16px;
      border-radius:14px;
      border:1px solid rgba(123,232,195,0.18);
      background:rgba(8,24,35,0.72);
      display:flex;
      justify-content:space-between;
      align-items:center;
      gap:12px;
    }
    .ops-pill span{color:var(--c8-text-muted);font-size:0.8rem;font-weight:700}
    .ops-pill strong{font-size:1.02rem}
    .metrics-grid{
      margin-top:18px;
      display:grid;
      grid-template-columns:repeat(4, minmax(0, 1fr));
      gap:12px;
    }
    .metric-card{
      padding:16px;
      border-radius:14px;
      border:1px solid rgba(126,146,204,0.22);
      background:rgba(9,20,32,0.72);
      display:grid;
      gap:6px;
    }
    .metric-card-muted{opacity:0.78}
    .metric-label{font-size:0.78rem;color:var(--c8-text-muted);font-weight:800}
    .metric-value{font-size:1.16rem;letter-spacing:-0.03em}
    .metric-meta{color:#aac0d2;font-size:0.8rem}
    .overview-wrap{margin-top:18px}
    .overview-card{padding:18px}
    .card-head,.section-head{display:flex;justify-content:space-between;gap:16px;align-items:flex-start;margin-bottom:14px}
    .card-head p,.section-head p{margin:0;color:var(--c8-text-muted);line-height:1.5}
    .overview-table{width:100%;border-collapse:collapse}
    .overview-table th,.overview-table td{
      padding:12px 10px;
      border-bottom:1px solid rgba(148,163,184,0.16);
      text-align:left;
    }
    .overview-table th{color:#9eb7cd;width:32%}
    .chart-grid{
      margin-top:18px;
      display:grid;
      grid-template-columns:1.1fr 1.1fr;
      gap:16px;
    }
    .chart-card{padding:16px;overflow:hidden}
    .chart-card-full{grid-column:1 / -1}
    .chart-body{min-height:320px}
    .chart-body.chart-pies{min-height:640px}
    .detail-shell{margin-top:18px;padding:18px}
    .detail-toolbar{display:flex;flex-wrap:wrap;gap:10px;margin-bottom:16px}
    .detail-toolbar button{
      background:rgba(18,33,55,0.88);
      border-color:rgba(126,146,204,0.44);
      color:#d7e6ff;
      padding:10px 16px;
    }
    .detail-toolbar button.active{
      background:linear-gradient(180deg, rgba(33,52,105,0.92), rgba(18,31,66,0.96));
      border-color:rgba(91,140,255,0.46);
    }
    .panel-box{display:none}
    .panel-box.visible{display:block}
    .detail-summary{
      background:rgba(51,65,85,0.45);
      border:1px solid rgba(129,140,248,0.25);
      border-radius:10px;
      padding:10px 14px;
      margin:0 0 14px;
      font-size:0.9rem;
      line-height:1.55;
      color:#cbd5e1;
    }
    .detail-summary strong{color:#e2e8f0;margin-right:8px;font-weight:600}
    .detail-summary.muted{opacity:0.9;color:#94a3b8}
    .detail-summary .sum-item{white-space:nowrap}
    .detail-summary .k{color:#94a3b8;font-size:0.86em;margin-right:4px}
    .detail-summary .v{color:#f1f5f9;font-weight:600}
    .detail-summary .sum-sep{margin:0 6px;color:#64748b}
    .table-wrap{overflow-x:auto}
    .caption{color:#94a3b8;font-size:0.88rem;margin:8px 0 12px;line-height:1.45}
    .empty-msg{color:#94a3b8;margin:8px 0}
    table.detail-table{border-collapse:collapse;width:100%;font-size:0.88rem}
    table.detail-table th,table.detail-table td{
      padding:8px 10px;
      text-align:left;
      border-bottom:1px solid rgba(148,163,184,0.18);
      white-space:nowrap;
    }
    table.detail-table th{color:#94a3b8;font-weight:600}
    table.detail-table th.sortable-th{cursor:pointer;user-select:none;position:relative}
    table.detail-table th.sortable-th:hover{background:rgba(99,102,241,0.14);color:#e2e8f0}
    table.detail-table th.sort-asc::after{content:' ▲';font-size:0.55em;margin-left:5px;opacity:0.9;vertical-align:middle}
    table.detail-table th.sort-desc::after{content:' ▼';font-size:0.55em;margin-left:5px;opacity:0.9;vertical-align:middle}
    table.detail-table tr:hover td{background:rgba(51,65,85,0.35)}
    .stress-details{
      margin-top:20px;
      background:rgba(30,41,59,0.45);
      border-radius:12px;
      padding:12px 16px;
      border:1px solid rgba(148,163,184,0.2);
    }
    .stress-details summary{cursor:pointer;color:#e2e8f0;font-weight:700;padding:6px 0}
    .footer-note{margin-top:20px;color:#7f95a9;font-size:0.82rem;line-height:1.55}
    @media (max-width: 1120px){
      .controls-grid,.metrics-grid,.ops-status-rail,.chart-grid{grid-template-columns:repeat(2, minmax(0, 1fr))}
      .hero{grid-template-columns:1fr}
    }
    @media (max-width: 720px){
      .top-header{grid-template-columns:1fr;gap:8px}
      .controls-grid,.metrics-grid,.ops-status-rail,.chart-grid{grid-template-columns:1fr}
      .container{padding:16px 12px 32px}
      .hero h1{font-size:1.55rem}
    }
  </style>
</head>
<body>
  <div class="container">
    <header class="top-header card">
      <div>
        <div class="brand-subtitle">Crypto8 Inspired Monitor</div>
        <div class="brand-title">BFut Risk Monitor</div>
      </div>
      <div></div>
      <div class="header-meta" id="generated-at">로딩 중</div>
    </header>

    <section class="hero card">
      <div>
        <p class="section-eyebrow">Dashboard</p>
        <h1>바이낸스 선물 리스크를 웹에서 조회하고 바로 조정합니다</h1>
        <p>기존 <code>risk_report.py</code> 계산 로직을 그대로 사용하면서, Crypto8 스타일의 카드/상황판/패널 구조로 재배치했습니다. 충격 범위, 프리셋, 미체결 체결 반영 여부를 바꿔가며 계정 상태를 즉시 다시 계산할 수 있습니다.</p>
      </div>
      <div class="hero-notes">
        <div class="hero-note"><span>Mode</span><strong>조회 중심 웹 모니터링</strong></div>
        <div class="hero-note"><span>Controls</span><strong>Shock sweep · preset · auto refresh</strong></div>
        <div class="hero-note"><span>Output</span><strong>요약 카드 · 차트 · 세부 표 · HTML export</strong></div>
      </div>
    </section>

    <section class="controls-card card">
      <div class="controls-head">
        <div>
          <p class="section-eyebrow">Controls</p>
          <h2>조회 파라미터</h2>
        </div>
        <div class="button-row">
          <button id="refresh-btn" type="button">새로고침</button>
          <button id="export-btn" type="button" class="secondary">HTML 내보내기</button>
          <button id="reset-btn" type="button" class="ghost">기본값</button>
        </div>
      </div>
      <div class="controls-grid">
        <div class="field">
          <label for="shock-min">Shock Min (%)</label>
          <input id="shock-min" type="number" step="0.5" value="-40"/>
        </div>
        <div class="field">
          <label for="shock-max">Shock Max (%)</label>
          <input id="shock-max" type="number" step="0.5" value="15"/>
        </div>
        <div class="field">
          <label for="shock-step">Shock Step (%)</label>
          <input id="shock-step" type="number" step="0.5" value="2.5"/>
        </div>
        <div class="field">
          <label for="preset">Preset Shock (%)</label>
          <input id="preset" type="number" step="0.5" value="0"/>
        </div>
        <div class="field">
          <label for="refresh-interval">Auto Refresh</label>
          <select id="refresh-interval">
            <option value="0">꺼짐</option>
            <option value="15">15초</option>
            <option value="30">30초</option>
            <option value="60">60초</option>
            <option value="120">120초</option>
          </select>
        </div>
        <div class="field">
          <label>미체결 체결 반영</label>
          <label class="checkline"><input id="simulate-fills" type="checkbox"/> 가격 충격 시 리밋 주문 체결 가정</label>
        </div>
      </div>
      <div class="status-line" id="status-line">데이터를 불러오는 중입니다.</div>
    </section>

    <section id="ops-rail"></section>
    <section class="metrics-grid" id="metrics-grid"></section>
    <section class="overview-wrap" id="overview-wrap"></section>

    <section class="chart-grid">
      <article class="chart-card card">
        <div class="section-head">
          <div><p class="section-eyebrow">Exposure</p><h2>심볼 비중</h2></div>
          <p>노셔널과 미실현손익 집중도를 함께 봅니다.</p>
        </div>
        <div class="chart-body chart-pies" id="chart-pies"></div>
      </article>
      <article class="chart-card card">
        <div class="section-head">
          <div><p class="section-eyebrow">Trades</p><h2>최근 체결 흐름</h2></div>
          <p>시간순 누적 순매수 금액입니다.</p>
        </div>
        <div class="chart-body" id="chart-trades"></div>
      </article>
      <article class="chart-card chart-card-full card">
        <div class="section-head">
          <div><p class="section-eyebrow">Stress</p><h2>가격 충격 민감도</h2></div>
          <p>담보 대비 손익 비율과 추정 순자산을 동시에 확인합니다.</p>
        </div>
        <div class="chart-body" id="chart-stress"></div>
      </article>
    </section>

    <section class="detail-shell card">
      <div class="section-head">
        <div><p class="section-eyebrow">Details</p><h2>세부 내역</h2></div>
        <p>헤더를 누르면 표를 정렬할 수 있습니다.</p>
      </div>
      <div class="detail-toolbar">
        <button type="button" class="active" data-panel="panel-pos">포지션</button>
        <button type="button" data-panel="panel-oo">미체결</button>
        <button type="button" data-panel="panel-tr">최근 체결</button>
        <button type="button" data-panel="panel-col">담보 자산</button>
        <button type="button" data-panel="panel-dist">주문-마크 거리</button>
      </div>
      <div id="panel-pos" class="panel-box visible">
        <h3>포지션 상세</h3>
        <div id="summary-pos"></div>
        <div class="table-wrap" id="table-pos"></div>
      </div>
      <div id="panel-oo" class="panel-box">
        <h3>미체결 주문</h3>
        <div id="summary-oo"></div>
        <div class="table-wrap" id="table-oo"></div>
      </div>
      <div id="panel-tr" class="panel-box">
        <h3>최근 약 3일 체결</h3>
        <div id="summary-tr"></div>
        <div class="table-wrap" id="table-tr"></div>
      </div>
      <div id="panel-col" class="panel-box">
        <h3>담보 자산</h3>
        <div id="summary-col"></div>
        <div class="table-wrap" id="table-col"></div>
      </div>
      <div id="panel-dist" class="panel-box">
        <h3>지정가 주문 vs 마크 (%)</h3>
        <div id="summary-dist"></div>
        <p class="caption">매수는 마크가 리밋보다 낮을수록 음수, 매도는 반대입니다.</p>
        <div class="table-wrap" id="table-dist"></div>
      </div>
      <details class="stress-details">
        <summary>시나리오별 수치 테이블 (전체 스윕)</summary>
        <div id="summary-stress"></div>
        <div class="table-wrap" id="table-stress"></div>
      </details>
      <p class="footer-note">본 화면은 참고용 단순 모델입니다. 실제 강제청산·증거금은 거래소 규칙, 교차/격리, 유지증거금률 등에 따라 달라질 수 있습니다.</p>
    </section>
  </div>

  <script>
    const state = { timerId: null, loading: false };

    function qs(id) { return document.getElementById(id); }

    function collectParams() {
      return {
        shock_min: qs("shock-min").value,
        shock_max: qs("shock-max").value,
        shock_step: qs("shock-step").value,
        preset: qs("preset").value,
        simulate_fills: qs("simulate-fills").checked ? "1" : "0"
      };
    }

    function buildQuery(params) {
      return new URLSearchParams(params).toString();
    }

    function setStatus(text, isError) {
      const el = qs("status-line");
      el.textContent = text;
      el.style.color = isError ? "#ff9bb9" : "";
    }

    function showPanel(id) {
      document.querySelectorAll(".panel-box").forEach((el) => el.classList.remove("visible"));
      document.querySelectorAll(".detail-toolbar button").forEach((el) => el.classList.remove("active"));
      const panel = document.getElementById(id);
      if (panel) panel.classList.add("visible");
      const button = document.querySelector('.detail-toolbar button[data-panel="' + id + '"]');
      if (button) button.classList.add("active");
    }

    function cellSortVal(td) {
      if (!td) return { t: "s", v: "" };
      const raw = td.textContent.replace(/\\u00a0/g, " ").trim();
      if (raw === "" || raw === "—" || raw === "-") return { t: "s", v: raw.toLowerCase() };
      let s = raw.replace(/,/g, "").trim();
      if (s.charAt(0) === "+") s = s.substring(1);
      if (/^-?(?:\\d+\\.?\\d*|\\.\\d+)(?:[eE][+-]?\\d+)?$/.test(s)) {
        const n = parseFloat(s);
        if (!Number.isNaN(n) && Number.isFinite(n)) return { t: "n", v: n };
      }
      return { t: "s", v: raw.toLowerCase() };
    }

    function compareRows(ax, bx, dir) {
      const a = ax.sortKey;
      const b = bx.sortKey;
      if (a.t === "n" && b.t === "n") return dir * (a.v - b.v);
      if (a.t === "n") return -dir;
      if (b.t === "n") return dir;
      if (a.v < b.v) return -dir;
      if (a.v > b.v) return dir;
      return 0;
    }

    function initSortableTables() {
      document.querySelectorAll("table.detail-table").forEach((table) => {
        const headRow = table.querySelector("thead tr");
        const tbody = table.querySelector("tbody");
        if (!headRow || !tbody) return;
        const ths = headRow.querySelectorAll("th");
        ths.forEach((th, colIdx) => {
          th.classList.add("sortable-th");
          th.title = "클릭: 오름차순 / 다시 클릭: 내림차순";
          th.onclick = (ev) => {
            ev.preventDefault();
            const curCol = table.getAttribute("data-sort-col");
            const curDir = table.getAttribute("data-sort-dir");
            let dir = 1;
            if (String(colIdx) === curCol) dir = curDir === "asc" ? -1 : 1;
            table.setAttribute("data-sort-col", String(colIdx));
            table.setAttribute("data-sort-dir", dir === 1 ? "asc" : "desc");
            ths.forEach((node) => node.classList.remove("sort-asc", "sort-desc"));
            th.classList.add(dir === 1 ? "sort-asc" : "sort-desc");
            const rows = Array.from(tbody.querySelectorAll("tr"));
            const keyed = rows.map((tr) => ({ tr, sortKey: cellSortVal(tr.children[colIdx]) }));
            keyed.sort((x, y) => compareRows(x, y, dir));
            keyed.forEach((item) => tbody.appendChild(item.tr));
          };
        });
      });
    }

    function applyPayload(payload) {
      qs("generated-at").textContent = "업데이트 " + payload.generated_at;
      qs("metrics-grid").innerHTML = payload.html.metrics;
      qs("ops-rail").innerHTML = payload.html.ops_rail;
      qs("overview-wrap").innerHTML = payload.html.overview;
      qs("chart-pies").innerHTML = payload.charts.pies;
      qs("chart-trades").innerHTML = payload.charts.trades;
      qs("chart-stress").innerHTML = payload.charts.stress;
      qs("summary-pos").innerHTML = payload.html.positions_summary;
      qs("summary-oo").innerHTML = payload.html.orders_summary;
      qs("summary-tr").innerHTML = payload.html.trades_summary;
      qs("summary-col").innerHTML = payload.html.collateral_summary;
      qs("summary-dist").innerHTML = payload.html.distance_summary;
      qs("summary-stress").innerHTML = payload.html.stress_summary;
      qs("table-pos").innerHTML = payload.html.positions_table;
      qs("table-oo").innerHTML = payload.html.orders_table;
      qs("table-tr").innerHTML = payload.html.trades_table;
      qs("table-col").innerHTML = payload.html.collateral_table;
      qs("table-dist").innerHTML = payload.html.distance_table;
      qs("table-stress").innerHTML = payload.html.stress_table;
      initSortableTables();
      setStatus("정상 업데이트 완료", false);
    }

    async function loadSnapshot() {
      if (state.loading) return;
      state.loading = true;
      setStatus("데이터를 조회하는 중입니다...", false);
      try {
        const resp = await fetch("/api/snapshot?" + buildQuery(collectParams()), { cache: "no-store" });
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.error || "데이터를 불러오지 못했습니다.");
        applyPayload(data);
      } catch (err) {
        setStatus((err && err.message) ? err.message : "알 수 없는 오류가 발생했습니다.", true);
      } finally {
        state.loading = false;
      }
    }

    function updateAutoRefresh() {
      if (state.timerId) {
        clearInterval(state.timerId);
        state.timerId = null;
      }
      const sec = Number(qs("refresh-interval").value || "0");
      if (sec > 0) {
        state.timerId = setInterval(loadSnapshot, sec * 1000);
      }
    }

    function resetControls() {
      qs("shock-min").value = "-40";
      qs("shock-max").value = "15";
      qs("shock-step").value = "2.5";
      qs("preset").value = "0";
      qs("simulate-fills").checked = false;
      qs("refresh-interval").value = "0";
      updateAutoRefresh();
      loadSnapshot();
    }

    document.querySelectorAll(".detail-toolbar button").forEach((button) => {
      button.addEventListener("click", () => showPanel(button.dataset.panel));
    });
    qs("refresh-btn").addEventListener("click", loadSnapshot);
    qs("reset-btn").addEventListener("click", resetControls);
    qs("export-btn").addEventListener("click", () => {
      window.open("/dashboard.html?" + buildQuery(collectParams()), "_blank", "noopener,noreferrer");
    });
    ["shock-min", "shock-max", "shock-step", "preset", "simulate-fills"].forEach((id) => {
      qs(id).addEventListener("change", loadSnapshot);
    });
    qs("refresh-interval").addEventListener("change", updateAutoRefresh);

    loadSnapshot();
  </script>
</body>
</html>
"""


class _RiskWebHandler(BaseHTTPRequestHandler):
    server_version = "BFutRiskMonitor/1.0"

    def _write_bytes(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _write_json(self, code: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._write_bytes(code, body, "application/json; charset=utf-8")

    def _query_options(self) -> argparse.Namespace:
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        base = getattr(self.server, "base_options", None)
        return _make_runtime_options(
            api_key=getattr(base, "api_key", None),
            api_secret=getattr(base, "api_secret", None),
            binance_key_file=getattr(base, "binance_key_file", None),
            simulate_fills=_coerce_bool(
                query.get("simulate_fills", [str(int(bool(getattr(base, "simulate_fills", False))))])[0],
                bool(getattr(base, "simulate_fills", False)),
            ),
            shock_min=_coerce_float(query.get("shock_min", [str(getattr(base, "shock_min", -40.0))])[0], getattr(base, "shock_min", -40.0)),
            shock_max=_coerce_float(query.get("shock_max", [str(getattr(base, "shock_max", 15.0))])[0], getattr(base, "shock_max", 15.0)),
            shock_step=_coerce_float(query.get("shock_step", [str(getattr(base, "shock_step", 2.5))])[0], getattr(base, "shock_step", 2.5)),
            preset=_coerce_float(query.get("preset", [str(getattr(base, "preset", 0.0))])[0], getattr(base, "preset", 0.0)),
        )

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/healthz":
                self._write_json(200, {"ok": True, "service": "bfut-risk-monitor"})
                return
            if parsed.path in {"/", "/index.html"}:
                self._write_bytes(200, _build_web_app_html().encode("utf-8"), "text/html; charset=utf-8")
                return
            if parsed.path == "/api/snapshot":
                self._write_json(200, _build_web_payload(self._query_options()))
                return
            if parsed.path == "/dashboard.html":
                html_doc = _build_web_payload(self._query_options())["dashboard_html"]
                self._write_bytes(200, html_doc.encode("utf-8"), "text/html; charset=utf-8")
                return
            self._write_json(404, {"error": f"Unknown path: {parsed.path}"})
        except Exception as exc:
            self._write_json(500, {"error": str(exc)})

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stdout.write("[risk-web] " + (fmt % args) + "\n")


def serve_risk_monitor(host: str, port: int, base_options: Optional[argparse.Namespace] = None, open_browser: bool = False) -> None:
    server = ThreadingHTTPServer((host, int(port)), _RiskWebHandler)
    server.base_options = base_options or argparse.Namespace()  # type: ignore[attr-defined]
    url = f"http://{host}:{int(port)}"
    print(f"[웹] BFut Risk Monitor 실행: {url}")
    print("[웹] 종료하려면 Ctrl+C")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[웹] 종료합니다.")
    finally:
        server.server_close()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Binance USDT-M futures risk & stress CLI report")
    p.add_argument("--api-key", default=None, help="Override BINANCE_API_KEY (지정 시 키 파일보다 우선)")
    p.add_argument("--api-secret", default=None, help="Override BINANCE_API_SECRET")
    p.add_argument(
        "--binance-key-file",
        "--binance-rtf",
        dest="binance_key_file",
        default=str(DEFAULT_BINANCE_CREDENTIALS_PATH),
        metavar="PATH",
        help="API Key(1번째 줄)·Secret(2번째 줄) .txt(권장) 또는 .rtf. 기본: %(default)s",
    )
    p.add_argument(
        "--simulate-fills",
        action="store_true",
        help="민감도에 미체결(리밋) 체결 시뮬 포함 (대시보드 체크박스와 동일)",
    )
    p.add_argument("--shock-min", type=float, default=-40.0)
    p.add_argument("--shock-max", type=float, default=15.0)
    p.add_argument("--shock-step", type=float, default=2.5)
    p.add_argument(
        "--preset",
        type=float,
        default=0.0,
        help="단일 시나리오 충격 %% (예: -10, 5). 대시보드 프리셋과 동일한 의미",
    )
    p.add_argument(
        "--export-html",
        metavar="PATH",
        default=None,
        help="대시보드 HTML(스트레스 + 도넛2 + 누적체결)을 지정 경로에 저장",
    )
    p.add_argument(
        "--no-browser",
        action="store_true",
        help="HTML을 기본 브라우저로 열지 않음",
    )
    p.add_argument(
        "--show-tables",
        action="store_true",
        help="포지션·미체결·체결·담보·주문-마크 거리 테이블을 stdout에 출력",
    )
    p.add_argument(
        "--stress-csv",
        metavar="PATH",
        default=None,
        help="스윕 전체 stress_df를 CSV로 저장",
    )
    p.add_argument(
        "--serve",
        action="store_true",
        help="Crypto8 스타일 웹 모니터링 서버 실행",
    )
    p.add_argument(
        "--host",
        default=os.environ.get("HOST", "127.0.0.1"),
        help="웹 서버 바인드 주소 (기본: %(default)s)",
    )
    p.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("PORT", "8765")),
        help="웹 서버 포트 (기본: %(default)s)",
    )
    p.add_argument(
        "--open-browser",
        action="store_true",
        help="웹 서버 실행 시 기본 브라우저 자동 열기",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if args.serve:
        serve_risk_monitor(args.host, args.port, base_options=args, open_browser=args.open_browser)
        return
    ak, sk = _resolve_api_credentials(args)
    client = get_client(api_key=ak, api_secret=sk)
    snap = load_full_snapshot(client)
    summary = snap["summary"]
    pos_df = snap["pos_df"]
    oo_df = snap["oo_df"]
    tr_df = snap["tr_df"]
    positions = snap["positions"]
    open_orders = snap["open_orders"]
    marks = snap["mark_prices"]
    collateral = snap["collateral"]

    shocks: list[float] = []
    x = float(args.shock_min)
    while x <= float(args.shock_max) + 1e-9:
        shocks.append(round(x, 6))
        x += float(args.shock_step)
    stress_df = build_stress_curve(shocks, positions, open_orders, marks, collateral, args.simulate_fills)
    sc = stress_scenario(
        float(args.preset), positions, open_orders, marks, collateral, args.simulate_fills
    )

    html_doc = build_dashboard_html_document(
        stress_df, pos_df, tr_df, oo_df, collateral, open_orders, marks
    )
    export_path: Optional[Path] = None
    if args.export_html:
        export_path = Path(args.export_html).expanduser().resolve()
        _write_html_sync(export_path, html_doc)

    if not args.no_browser:
        if export_path is not None:
            _open_html_file(export_path)
            print(f"\n[그래프] 브라우저에서 열었습니다: {export_path}")
        else:
            tmp_path = open_dashboard_in_browser(html_doc)
            print(f"\n[그래프] 브라우저에서 열었습니다 (임시): {tmp_path}")

    if args.export_html:
        print(f"\n[저장] HTML: {export_path}")

    lines = [
        "=== 계정 요약 ===",
        f"지갑 잔고(totalWalletBalance):     {summary.total_wallet:,.2f} USDT",
        f"미실현손익:                       {summary.total_unrealized:,.2f} USDT",
        f"마진 잔고:                         {summary.total_margin_balance:,.2f} USDT",
        f"사용 가능:                         {summary.available_balance:,.2f} USDT",
        f"유지증거금 여유:                   {snap['maint_buffer']:,.2f} USDT",
        f"유지증거금 비중(총마진 대비):      {snap['margin_ratio_pct']:.2f}%",
        "",
        "=== 집중도 · 요약 ===",
        f"노출 포지션: {len(positions)}개 | 미체결: {len(open_orders)}개",
        f"HHI(노셔널 집중도): {snap['hhi']:.3f}",
        f"7일 실현손익 합(조회 심볼): {sum(snap['realized_7d'].values()):,.2f} USDT",
        "",
        f"=== 시나리오 프리셋 ({args.preset:+.1f}%) ===",
        f"총 포지션 가치:     {sc['total_position_value']:,.0f}",
        f"미실현손익 합:          {sc['total_unrealized_pnl']:,.0f}",
        f"담보(USDT환산):    {sc['collateral_usdt_equiv']:,.0f}",
        f"위험도(단순):       {sc['risk_label']}",
        "",
    ]
    print("\n".join(lines))

    print("=== 스트레스 스윕 (일부) ===")
    preview = stress_df.copy()
    print(preview.to_string(index=False, max_rows=20))

    if args.stress_csv:
        stress_df.to_csv(args.stress_csv, index=False)
        print(f"\n[저장] stress CSV: {args.stress_csv}")

    if args.show_tables:
        print("\n--- 포지션 ---")
        if pos_df.empty:
            print("(없음)")
        else:
            print(pos_df.to_string(index=False))
        print("\n--- 미체결 ---")
        if oo_df.empty:
            print("(없음)")
        else:
            print(oo_df.to_string(index=False))
            if "est_value_usdt" in oo_df.columns:
                print(f"추정 명목가치 합(마크 기준): {oo_df['est_value_usdt'].sum():,.2f} USDT")
        print("\n--- 최근 약 3일 체결 ---")
        if tr_df.empty:
            print("(없음)")
        else:
            print(tr_df.to_string(index=False))
        print("\n--- 담보 자산 ---")
        rows = []
        for name, info in collateral.items():
            rows.append(
                {
                    "asset": name,
                    "wallet": info["wallet_balance"],
                    "price": info["current_price"],
                    "value_usdt": info["total_value"],
                    "upl_asset": info["unrealized_pnl"],
                }
            )
        cdf = pd.DataFrame(rows)
        if cdf.empty:
            print("(없음)")
        else:
            print(cdf.to_string(index=False))
        print("\n--- 주문-마크 거리(%) ---")
        dist_df = order_distance_stats(open_orders, marks)
        if dist_df.empty:
            print("(분석 가능한 리밋 주문 없음)")
        else:
            print(dist_df.to_string(index=False))

    print(
        "\n참고: 실제 강제청산·증거금은 거래소 규칙·교차/격리·MMR 등으로 달라질 수 있습니다."
    )


if __name__ == "__main__":
    main()
