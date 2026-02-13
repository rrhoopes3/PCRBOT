import sys
import os
import time
import logging
import logging.handlers
import csv
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional

import json
import requests

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal, QUrl
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QLabel, QComboBox, QListWidget, QLineEdit,
    QPushButton, QAbstractItemView, QFrame, QSplitter, QSizePolicy,
    QCheckBox, QDoubleSpinBox, QFileDialog, QSlider,
)
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtMultimediaWidgets import QVideoWidget

# --- Logging setup ---
LOG_FILE = os.path.join(
    os.path.dirname(sys.executable) if getattr(sys, 'frozen', False)
    else os.path.dirname(os.path.abspath(__file__)),
    'pcr_dashboard.log'
)
logger = logging.getLogger('PCRDashboard')
logger.setLevel(logging.DEBUG)
_file_handler = logging.handlers.RotatingFileHandler(
    LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding='utf-8'
)
_file_handler.setFormatter(logging.Formatter(
    '%(asctime)s | %(levelname)-8s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S'
))
logger.addHandler(_file_handler)
_console_handler = logging.StreamHandler()
_console_handler.setLevel(logging.INFO)
_console_handler.setFormatter(logging.Formatter('%(levelname)-8s | %(message)s'))
logger.addHandler(_console_handler)

PRESET_TICKERS = [
    "SPX", "NDX", "RUT", "OEX", "XSP", "VIX",
    "SPY", "QQQ", "IWM", "DIA",
    "XLE", "XLF", "XLK", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC",
]
MAX_HISTORY = 100
REFRESH_INTERVAL_MS = 60_000
NUM_3D_EXPIRIES = 12  # Number of expiries to display in 3D chart

# --- Tradier API config ---
TRADIER_BASE_URL = "https://api.tradier.com/v1"
TRADIER_HEADERS = {"Accept": "application/json"}

CONFIG_FILE = os.path.join(
    os.path.dirname(sys.executable) if getattr(sys, 'frozen', False)
    else os.path.dirname(os.path.abspath(__file__)),
    'pcr_config.json'
)

SESSIONS_DIR = os.path.join(
    os.path.dirname(sys.executable) if getattr(sys, 'frozen', False)
    else os.path.dirname(os.path.abspath(__file__)),
    'sessions'
)

# --- Theme definitions ---
DARK_THEME = {
    'name': 'dark',
    'bg': '#1e1e1e', 'bg2': '#2a2a2a', 'bg3': '#252525',
    'fg': '#ddd', 'fg2': '#aaa', 'fg3': '#888', 'fg4': '#666',
    'border': '#444', 'accent': '#1a73e8', 'green': '#388e3c',
    'red': '#ff6b6b', 'grid': '#333', 'plotly_template': 'plotly_dark',
}
LIGHT_THEME = {
    'name': 'light',
    'bg': '#f5f5f5', 'bg2': '#ffffff', 'bg3': '#e8e8e8',
    'fg': '#222', 'fg2': '#555', 'fg3': '#777', 'fg4': '#999',
    'border': '#ccc', 'accent': '#1a73e8', 'green': '#2e7d32',
    'red': '#d32f2f', 'grid': '#ddd', 'plotly_template': 'plotly_white',
}


def load_config() -> dict:
    """Load config from JSON file next to the exe/script."""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            logger.exception("Failed to load config file")
    return {}


def save_config(cfg: dict):
    """Save config to JSON file."""
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(cfg, f, indent=2)
        logger.info("Config saved to %s", CONFIG_FILE)
    except Exception:
        logger.exception("Failed to save config file")


def tradier_get(endpoint: str, params: dict, api_key: str) -> dict | None:
    """Make a GET request to the Tradier API."""
    headers = {**TRADIER_HEADERS, "Authorization": f"Bearer {api_key}"}
    url = f"{TRADIER_BASE_URL}{endpoint}"
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.HTTPError as e:
        logger.error("Tradier HTTP error %s for %s: %s", resp.status_code, url, e)
        return None
    except Exception:
        logger.exception("Tradier request failed for %s", url)
        return None


def resource_path(filename):
    """Get path to bundled resource (works for dev and PyInstaller)."""
    if getattr(sys, 'frozen', False):
        base = sys._MEIPASS
    else:
        base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, filename)


@dataclass
class DataPoint:
    vol_ratio: float
    oi_ratio: float
    put_vol: int
    call_vol: int
    put_oi: int
    call_oi: int
    expiry: str
    timestamp: str


class DataFetcher(QThread):
    """Background thread that fetches real-time options data from Tradier API."""
    data_ready = pyqtSignal(str, str, object)       # ticker, expiry, DataPoint|None
    expiries_ready = pyqtSignal(str, list)           # ticker, [expiry_strings]
    all_done = pyqtSignal()
    status_update = pyqtSignal(str)                  # status message

    def __init__(self, parent=None):
        super().__init__(parent)
        self.tasks: list[tuple[str, str]] = []       # (ticker, expiry)
        self.fetch_expiries_for: list[str] = []      # tickers needing expiry lookup
        self.api_key: str = ""

    def set_tasks(self, tasks, fetch_expiries_for=None):
        self.tasks = tasks
        self.fetch_expiries_for = fetch_expiries_for or []

    def run(self):
        if not self.api_key:
            logger.error("No Tradier API key configured")
            self.status_update.emit("ERROR: No API key — enter key in sidebar")
            self.all_done.emit()
            return

        logger.info("DataFetcher started — %d expiry lookups, %d data tasks",
                     len(self.fetch_expiries_for), len(self.tasks))

        # First fetch expiries for new tickers
        for ticker in self.fetch_expiries_for:
            if self.isInterruptionRequested():
                logger.info("DataFetcher interrupted during expiry fetch")
                return
            self.status_update.emit(f"Fetching expiries for {ticker}...")
            data = tradier_get("/markets/options/expirations",
                               {"symbol": ticker, "includeAllRoots": "true"},
                               self.api_key)
            if data and "expirations" in data and data["expirations"]:
                dates = data["expirations"].get("date", [])
                if isinstance(dates, str):
                    dates = [dates]
                expiries = dates[:20]
                logger.info("Expiries for %s: %s", ticker, expiries)
                self.expiries_ready.emit(ticker, expiries)
            else:
                logger.warning("No expiries returned for %s: %s", ticker, data)
                self.expiries_ready.emit(ticker, [])
            time.sleep(0.2)

        # Then fetch option chain data for each ticker/expiry pair
        # Use a single timestamp for the entire refresh cycle so all expiries align on the time axis
        cycle_timestamp = datetime.now().strftime('%H:%M:%S')
        for ticker, expiry in self.tasks:
            if self.isInterruptionRequested():
                logger.info("DataFetcher interrupted during data fetch")
                return
            self.status_update.emit(f"Fetching {ticker} ({expiry})...")
            data = tradier_get("/markets/options/chains",
                               {"symbol": ticker, "expiration": expiry, "greeks": "false"},
                               self.api_key)
            if not data or "options" not in data or not data["options"]:
                logger.warning("No chain data for %s/%s: %s", ticker, expiry, data)
                self.data_ready.emit(ticker, expiry, None)
                continue

            options = data["options"].get("option", [])
            if isinstance(options, dict):
                options = [options]
            if not options:
                logger.warning("Empty options list for %s/%s", ticker, expiry)
                self.data_ready.emit(ticker, expiry, None)
                continue

            put_vol = sum(o.get("volume", 0) or 0 for o in options if o.get("option_type") == "put")
            call_vol = sum(o.get("volume", 0) or 0 for o in options if o.get("option_type") == "call")
            put_oi = sum(o.get("open_interest", 0) or 0 for o in options if o.get("option_type") == "put")
            call_oi = sum(o.get("open_interest", 0) or 0 for o in options if o.get("option_type") == "call")

            vol_ratio = put_vol / call_vol if call_vol > 0 else np.nan
            oi_ratio = put_oi / call_oi if call_oi > 0 else np.nan

            dp = DataPoint(
                vol_ratio=vol_ratio, oi_ratio=oi_ratio,
                put_vol=int(put_vol), call_vol=int(call_vol),
                put_oi=int(put_oi), call_oi=int(call_oi),
                expiry=expiry,
                timestamp=cycle_timestamp,
            )
            logger.info("Data OK for %s/%s — vol_ratio=%.3f oi_ratio=%.3f put_vol=%d call_vol=%d",
                        ticker, expiry, vol_ratio, oi_ratio, dp.put_vol, dp.call_vol)
            self.data_ready.emit(ticker, expiry, dp)
            time.sleep(0.2)

        logger.info("DataFetcher refresh cycle complete")
        self.status_update.emit("Refresh complete.")
        self.all_done.emit()


class PlotlyChartWidget(QWebEngineView):
    """Renders Plotly 3D charts via WebGL — interactive, rotatable, zoomable."""

    LAYER_COLORS = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#aec7e8', '#ffbb78',
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(350)
        self.theme = DARK_THEME
        self._show_placeholder()

    def _show_placeholder(self):
        t = self.theme
        self.setHtml(f"<html><body style='background:{t['bg']};color:{t['fg2']};display:flex;"
                     "align-items:center;justify-content:center;height:100vh;margin:0;'>"
                     "<p>Waiting for data...</p></body></html>")

    def update_chart_3d(self, histories: dict[str, list], ticker: str, expiries: list[str],
                        connect_expiries: bool = False, comparison_data: dict = None):
        """Render a 3D layered chart: X=time (floor), Y=expiry (floor depth), Z=ratio (vertical).
        comparison_data: optional dict of {ticker: {key: [DataPoint]}} for multi-ticker overlay."""
        if not expiries or not histories:
            return
        t = self.theme
        fig = go.Figure()

        # Build list of ticker datasets to render
        datasets = [(ticker, histories, self.LAYER_COLORS)]
        if comparison_data:
            comp_palettes = [
                ['#e6194b', '#f58231', '#ffe119', '#bfef45'],
                ['#3cb44b', '#42d4f4', '#4363d8', '#911eb4'],
            ]
            for ci, (comp_ticker, comp_hist) in enumerate(comparison_data.items()):
                datasets.append((comp_ticker, comp_hist, comp_palettes[ci % len(comp_palettes)]))

        active_indices = []
        expiry_data = {}

        for ds_ticker, ds_hist, palette in datasets:
            prefix = f"{ds_ticker}: " if len(datasets) > 1 else ""
            for i, expiry in enumerate(expiries[:NUM_3D_EXPIRIES]):
                key = f"{ds_ticker}_{expiry}"
                points = ds_hist.get(key, [])
                if not points:
                    continue

                if ds_ticker == ticker:
                    active_indices.append((i, expiry))
                    ts_map = {}
                    for j, p in enumerate(points):
                        ts_map[p.timestamp] = (p.vol_ratio, p.oi_ratio)
                    expiry_data[i] = ts_map

                timestamps = [p.timestamp for p in points]
                vol_ratios = [p.vol_ratio for p in points]
                oi_ratios = [p.oi_ratio for p in points]
                y_pos = [i] * len(points)
                color = palette[i % len(palette)]
                short_exp = expiry[5:] if len(expiry) > 5 else expiry

                fig.add_trace(go.Scatter3d(
                    x=timestamps, y=y_pos, z=vol_ratios,
                    mode='lines+markers',
                    line=dict(color=color, width=4),
                    marker=dict(size=3, color=color),
                    name=f"{prefix}{short_exp} Vol",
                    legendgroup=f"{ds_ticker}_{expiry}",
                ))
                fig.add_trace(go.Scatter3d(
                    x=timestamps, y=y_pos, z=oi_ratios,
                    mode='lines+markers',
                    line=dict(color=color, width=3, dash='dash'),
                    marker=dict(size=2, color=color, symbol='diamond'),
                    name=f"{prefix}{short_exp} OI",
                    legendgroup=f"{ds_ticker}_{expiry}",
                ))

        # Connector lines
        if connect_expiries and len(active_indices) >= 2:
            all_timestamps = set()
            for i, _ in active_indices:
                if i in expiry_data:
                    all_timestamps.update(expiry_data[i].keys())
            for ts in sorted(all_timestamps):
                vol_y, vol_z, oi_y, oi_z = [], [], [], []
                for i, _ in active_indices:
                    if i in expiry_data and ts in expiry_data[i]:
                        vr, oir = expiry_data[i][ts]
                        if pd.notna(vr):
                            vol_y.append(i); vol_z.append(vr)
                        if pd.notna(oir):
                            oi_y.append(i); oi_z.append(oir)
                if len(vol_y) >= 2:
                    fig.add_trace(go.Scatter3d(
                        x=[ts]*len(vol_y), y=vol_y, z=vol_z, mode='lines',
                        line=dict(color='rgba(255,255,255,0.3)', width=2),
                        showlegend=False, hoverinfo='skip'))
                if len(oi_y) >= 2:
                    fig.add_trace(go.Scatter3d(
                        x=[ts]*len(oi_y), y=oi_y, z=oi_z, mode='lines',
                        line=dict(color='rgba(255,200,100,0.25)', width=1.5, dash='dot'),
                        showlegend=False, hoverinfo='skip'))

        active_expiries = [e for e in expiries[:NUM_3D_EXPIRIES]
                           if f"{ticker}_{e}" in histories and histories[f"{ticker}_{e}"]]
        y_tickvals = list(range(len(active_expiries)))
        y_ticktext = [e[5:] for e in active_expiries]

        fig.update_layout(
            template=t['plotly_template'],
            title=dict(text=f"{ticker} — Multi-Expiry 3D Ratios", font=dict(size=16)),
            margin=dict(l=0, r=0, t=40, b=0),
            scene=dict(
                xaxis=dict(title='Time', showgrid=True, gridcolor=t['grid']),
                yaxis=dict(title='Expiry', tickvals=y_tickvals, ticktext=y_ticktext,
                           showgrid=True, gridcolor=t['grid']),
                zaxis=dict(title='Ratio', showgrid=True, gridcolor=t['grid']),
                bgcolor=t['bg'],
                camera=dict(eye=dict(x=2.2, y=-2.0, z=1.0)),
            ),
            legend=dict(font=dict(size=10), bgcolor=f"rgba(30,30,30,0.8)",
                        bordercolor=t['border'], borderwidth=1),
            paper_bgcolor=t['bg'],
        )
        html = fig.to_html(include_plotlyjs='cdn', full_html=True)
        self.setHtml(html)


class HeatmapWidget(QWebEngineView):
    """2D heatmap: rows=expiries, columns=timestamps, color=vol ratio."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(250)
        self.theme = DARK_THEME
        self.setHtml("<html><body style='background:#1e1e1e;color:#aaa;display:flex;"
                     "align-items:center;justify-content:center;height:100%;margin:0;'>"
                     "<p>Heatmap will appear after multiple data points...</p></body></html>")

    def update_heatmap(self, histories: dict[str, list], ticker: str, expiries: list[str]):
        if not expiries or not histories:
            return
        t = self.theme
        # Build matrix: rows=expiries, cols=timestamps
        all_timestamps = []
        for expiry in expiries[:NUM_3D_EXPIRIES]:
            key = f"{ticker}_{expiry}"
            for p in histories.get(key, []):
                if p.timestamp not in all_timestamps:
                    all_timestamps.append(p.timestamp)
        if not all_timestamps:
            return

        z_data = []
        y_labels = []
        for expiry in expiries[:NUM_3D_EXPIRIES]:
            key = f"{ticker}_{expiry}"
            points = histories.get(key, [])
            if not points:
                continue
            ts_map = {p.timestamp: p.vol_ratio for p in points}
            row = [ts_map.get(ts, np.nan) for ts in all_timestamps]
            z_data.append(row)
            y_labels.append(expiry[5:] if len(expiry) > 5 else expiry)

        if not z_data:
            return

        fig = go.Figure(data=go.Heatmap(
            z=z_data, x=all_timestamps, y=y_labels,
            colorscale='RdYlGn_r',  # Red=high PCR (bearish), Green=low (bullish)
            colorbar=dict(title='Vol PCR'),
            hoverongaps=False,
        ))
        fig.update_layout(
            template=t['plotly_template'],
            title=dict(text=f"{ticker} — PCR Heatmap (Vol Ratio)", font=dict(size=14)),
            xaxis=dict(title='Time'), yaxis=dict(title='Expiry'),
            margin=dict(l=60, r=20, t=40, b=40),
            paper_bgcolor=t['bg'], plot_bgcolor=t['bg'],
        )
        self.setHtml(fig.to_html(include_plotlyjs='cdn', full_html=True))


class VideoIndicator(QWidget):
    """Plays bearish/bullish/neutral video + audio based on sentiment.
    Click to toggle mute/unmute. Sits in the metrics row (upper right)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Video player (always muted — audio comes from separate MP3 player)
        self.setFixedHeight(140)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.video_widget = QVideoWidget()
        self.video_widget.setFixedSize(250, 140)
        self.video_widget.setStyleSheet("border: 2px solid #ff6b6b;")
        self.video_widget.setCursor(Qt.CursorShape.PointingHandCursor)
        self.player = QMediaPlayer()
        self.video_audio = QAudioOutput()
        self.video_audio.setVolume(0)  # Video audio always muted
        self.player.setAudioOutput(self.video_audio)
        self.player.setVideoOutput(self.video_widget)
        self.player.setLoops(QMediaPlayer.Loops.Infinite)

        # Audio player for MP3 files (separate from video)
        self.audio_player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.audio_output.setVolume(0)  # Start muted
        self.audio_player.setAudioOutput(self.audio_output)
        self.audio_player.setLoops(QMediaPlayer.Loops.Infinite)

        layout.addWidget(self.video_widget)

        # Mute indicator overlay
        self.mute_label = QLabel("\U0001f507", self.video_widget)
        self.mute_label.setStyleSheet(
            "color: #ff6b6b; background-color: rgba(0,0,0,180); "
            "font-size: 14px; padding: 2px 5px; border-radius: 3px;"
        )
        self.mute_label.move(4, 4)
        self.mute_label.show()

        # Click handling via event filter on video widget
        self.video_widget.installEventFilter(self)

        # State
        self._current_video = ""
        self._current_audio = ""
        self._muted = True
        self._audio_volume = 0.5

    def eventFilter(self, obj, event):
        if obj is self.video_widget and event.type() == event.Type.MouseButtonPress:
            self.toggle_mute()
            return True
        return super().eventFilter(obj, event)

    def toggle_mute(self):
        self._muted = not self._muted
        if self._muted:
            self.audio_output.setVolume(0)
            logger.debug("Audio muted")
        else:
            self.audio_output.setVolume(self._audio_volume)
            logger.debug("Audio unmuted (volume=%.2f)", self._audio_volume)
        self._update_mute_visual()

    def pause_audio(self):
        """Pause audio when tab is not visible."""
        self.audio_player.pause()

    def resume_audio(self):
        """Resume audio when tab becomes visible (only if unmuted)."""
        if self._current_audio:
            self.audio_player.play()

    def _update_mute_visual(self):
        if self._muted:
            self.mute_label.setText("\U0001f507")
            self.mute_label.setStyleSheet(
                "color: #ff6b6b; background-color: rgba(0,0,0,180); "
                "font-size: 14px; padding: 2px 5px; border-radius: 3px;"
            )
            self.video_widget.setStyleSheet("border: 2px solid #ff6b6b;")
        else:
            self.mute_label.setText("\U0001f50a")
            self.mute_label.setStyleSheet(
                "color: #4caf50; background-color: rgba(0,0,0,180); "
                "font-size: 14px; padding: 2px 5px; border-radius: 3px;"
            )
            self.video_widget.setStyleSheet("border: 2px solid #4caf50;")

    def update_sentiment(self, vol_ratio: float):
        if pd.isna(vol_ratio):
            self.player.stop()
            self.audio_player.stop()
            self._current_video = ""
            self._current_audio = ""
            return

        if vol_ratio > 1:
            video_file = 'bearish.mp4'
            audio_file = 'bearish.m4a'
        elif vol_ratio < 1:
            video_file = 'bullish.mp4'
            audio_file = 'bullish.m4a'
        else:
            video_file = 'neutral.mp4'
            audio_file = 'neutral.m4a'

        # Update video
        if video_file != self._current_video:
            path = resource_path(video_file)
            if os.path.exists(path):
                logger.debug("Playing %s: %s", video_file, path)
                self.player.stop()
                self.player.setSource(QUrl.fromLocalFile(path))
                self.player.play()
                self._current_video = video_file
            else:
                logger.warning("%s not found at %s", video_file, path)
                self.player.stop()
                self._current_video = ""

        # Update audio MP3
        if audio_file != self._current_audio:
            audio_path = resource_path(audio_file)
            if os.path.exists(audio_path):
                logger.debug("Playing audio %s: %s", audio_file, audio_path)
                self.audio_player.stop()
                self.audio_player.setSource(QUrl.fromLocalFile(audio_path))
                self.audio_player.play()
                self._current_audio = audio_file
            else:
                logger.debug("Audio %s not found (will load when available)", audio_file)
                self.audio_player.stop()
                self._current_audio = ""


class MetricsPanel(QWidget):
    """Displays 4 metric cards + video indicator in a horizontal row."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.metric_labels = {}
        for name in ['Vol Ratio', 'OI Ratio', 'Put Vol', 'Call Vol']:
            frame = QFrame()
            frame.setStyleSheet("""
                QFrame {
                    background-color: #2a2a2a;
                    border: 1px solid #444;
                    border-radius: 8px;
                    padding: 10px;
                }
            """)
            fl = QVBoxLayout(frame)
            title = QLabel(name)
            title.setStyleSheet("color: #aaa; font-size: 12px;")
            title.setAlignment(Qt.AlignmentFlag.AlignCenter)
            value = QLabel("--")
            value.setStyleSheet("color: #fff; font-size: 22px; font-weight: bold;")
            value.setAlignment(Qt.AlignmentFlag.AlignCenter)
            fl.addWidget(title)
            fl.addWidget(value)
            self.metric_labels[name] = value
            layout.addWidget(frame)

        # Video indicator in the metrics row (upper right)
        self.video_indicator = VideoIndicator()
        layout.addWidget(self.video_indicator)

    def update_metrics(self, dp: DataPoint):
        self.metric_labels['Vol Ratio'].setText(f"{dp.vol_ratio:.2f}" if pd.notna(dp.vol_ratio) else "--")
        self.metric_labels['OI Ratio'].setText(f"{dp.oi_ratio:.2f}" if pd.notna(dp.oi_ratio) else "--")
        self.metric_labels['Put Vol'].setText(f"{dp.put_vol:,}")
        self.metric_labels['Call Vol'].setText(f"{dp.call_vol:,}")
        self.video_indicator.update_sentiment(dp.vol_ratio)


class TickerTab(QWidget):
    """One tab per ticker: expiry selector, metrics, 3D chart, heatmap, alerts, export."""
    expiry_changed = pyqtSignal(str, str)  # ticker, new_expiry
    alert_triggered = pyqtSignal(str, str, float)  # ticker, direction, value

    def __init__(self, ticker: str, parent=None):
        super().__init__(parent)
        self.ticker = ticker
        self.current_expiry = ""
        self.all_expiries: list[str] = []
        self.comparison_tickers: list[str] = []  # For multi-ticker overlay
        self._last_histories = None
        self._alert_high = 1.5
        self._alert_low = 0.5
        self._alerts_enabled = False
        self._last_alert_direction = ""  # Prevent repeated alerts

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # === Row 1: Expiry selector + chart controls ===
        top = QHBoxLayout()
        top.addWidget(QLabel("Primary Expiry:"))
        self.expiry_combo = QComboBox()
        self.expiry_combo.setMinimumWidth(160)
        self.expiry_combo.currentTextChanged.connect(self._on_expiry_changed)
        top.addWidget(self.expiry_combo)
        self.expiry_label = QLabel("")
        self.expiry_label.setStyleSheet("color: #888;")
        top.addWidget(self.expiry_label)

        self.connect_checkbox = QCheckBox("Connect")
        self.connect_checkbox.setStyleSheet("color: #aaa; font-size: 11px;")
        self.connect_checkbox.setToolTip("Connect expiry layers at each timestamp")
        self.connect_checkbox.stateChanged.connect(self._on_connect_toggled)
        top.addWidget(self.connect_checkbox)

        # Fullscreen pop-out button
        self.popout_btn = QPushButton("Pop Out")
        self.popout_btn.setStyleSheet("padding: 2px 8px; font-size: 11px;")
        self.popout_btn.setToolTip("Open 3D chart in a separate window")
        self.popout_btn.clicked.connect(self._pop_out_chart)
        top.addWidget(self.popout_btn)

        # Export buttons
        self.export_csv_btn = QPushButton("CSV")
        self.export_csv_btn.setStyleSheet("padding: 2px 8px; font-size: 11px;")
        self.export_csv_btn.setToolTip("Export data to CSV")
        self.export_csv_btn.clicked.connect(self._export_csv)
        top.addWidget(self.export_csv_btn)

        self.export_png_btn = QPushButton("PNG")
        self.export_png_btn.setStyleSheet("padding: 2px 8px; font-size: 11px;")
        self.export_png_btn.setToolTip("Screenshot the 3D chart")
        self.export_png_btn.clicked.connect(self._export_png)
        top.addWidget(self.export_png_btn)

        top.addStretch()
        layout.addLayout(top)

        # === Row 2: Alerts ===
        alert_row = QHBoxLayout()
        self.alert_checkbox = QCheckBox("Alerts")
        self.alert_checkbox.setStyleSheet("color: #aaa; font-size: 11px;")
        self.alert_checkbox.setToolTip("Flash when PCR crosses thresholds")
        self.alert_checkbox.stateChanged.connect(self._on_alerts_toggled)
        alert_row.addWidget(self.alert_checkbox)

        alert_row.addWidget(QLabel("High:"))
        self.alert_high_spin = QDoubleSpinBox()
        self.alert_high_spin.setRange(0.1, 10.0)
        self.alert_high_spin.setSingleStep(0.1)
        self.alert_high_spin.setValue(1.5)
        self.alert_high_spin.setFixedWidth(65)
        self.alert_high_spin.setStyleSheet("font-size: 11px;")
        self.alert_high_spin.valueChanged.connect(lambda v: setattr(self, '_alert_high', v))
        alert_row.addWidget(self.alert_high_spin)

        alert_row.addWidget(QLabel("Low:"))
        self.alert_low_spin = QDoubleSpinBox()
        self.alert_low_spin.setRange(0.01, 10.0)
        self.alert_low_spin.setSingleStep(0.1)
        self.alert_low_spin.setValue(0.5)
        self.alert_low_spin.setFixedWidth(65)
        self.alert_low_spin.setStyleSheet("font-size: 11px;")
        self.alert_low_spin.valueChanged.connect(lambda v: setattr(self, '_alert_low', v))
        alert_row.addWidget(self.alert_low_spin)

        self.alert_status = QLabel("")
        self.alert_status.setStyleSheet("font-size: 11px; font-weight: bold;")
        alert_row.addWidget(self.alert_status)

        # Video size slider
        alert_row.addStretch()
        alert_row.addWidget(QLabel("Vid:"))
        self.video_slider = QSlider(Qt.Orientation.Horizontal)
        self.video_slider.setRange(100, 400)
        self.video_slider.setValue(250)
        self.video_slider.setFixedWidth(80)
        self.video_slider.setToolTip("Resize video indicator")
        self.video_slider.valueChanged.connect(self._on_video_resize)
        alert_row.addWidget(self.video_slider)

        layout.addLayout(alert_row)

        # === Metrics + video indicator ===
        self.metrics = MetricsPanel()
        layout.addWidget(self.metrics)

        # === Chart area: sub-tabs for 3D and Heatmap ===
        self.chart_tabs = QTabWidget()
        self.chart_tabs.setStyleSheet("QTabBar::tab { padding: 4px 12px; font-size: 11px; }")

        self.chart = PlotlyChartWidget()
        self.chart.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.chart_tabs.addTab(self.chart, "3D Chart")

        self.heatmap = HeatmapWidget()
        self.heatmap.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.chart_tabs.addTab(self.heatmap, "Heatmap")

        layout.addWidget(self.chart_tabs, stretch=1)

        # Error label
        self.error_label = QLabel("")
        self.error_label.setStyleSheet("color: #ff6b6b; font-size: 14px; padding: 20px;")
        self.error_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.error_label.hide()
        layout.addWidget(self.error_label)

        # Pop-out window reference
        self._popout_window = None

    def set_expiries(self, expiries: list[str]):
        self.all_expiries = expiries
        self.expiry_combo.blockSignals(True)
        self.expiry_combo.clear()
        if expiries:
            self.expiry_combo.addItems(expiries)
            self.current_expiry = expiries[0]
        self.expiry_combo.blockSignals(False)

    def _on_expiry_changed(self, text):
        if text and text != self.current_expiry:
            self.current_expiry = text
            self.expiry_changed.emit(self.ticker, text)

    def get_chart_expiries(self) -> list[str]:
        if not self.all_expiries or not self.current_expiry:
            return []
        try:
            idx = self.all_expiries.index(self.current_expiry)
        except ValueError:
            idx = 0
        return self.all_expiries[idx:idx + 1 + NUM_3D_EXPIRIES]

    def _on_connect_toggled(self, state):
        chart_expiries = self.get_chart_expiries()
        if chart_expiries and self._last_histories:
            self.chart.update_chart_3d(
                self._last_histories, self.ticker, chart_expiries,
                connect_expiries=self.connect_checkbox.isChecked())

    def _on_alerts_toggled(self, state):
        self._alerts_enabled = bool(state)
        if not state:
            self.alert_status.setText("")
            self._last_alert_direction = ""

    def _check_alerts(self, vol_ratio: float):
        """Check if PCR crosses alert thresholds."""
        if not self._alerts_enabled or pd.isna(vol_ratio):
            return
        direction = ""
        if vol_ratio >= self._alert_high:
            direction = "HIGH"
            self.alert_status.setText(f"ALERT: PCR {vol_ratio:.2f} >= {self._alert_high}")
            self.alert_status.setStyleSheet("color: #ff6b6b; font-size: 11px; font-weight: bold;")
        elif vol_ratio <= self._alert_low:
            direction = "LOW"
            self.alert_status.setText(f"ALERT: PCR {vol_ratio:.2f} <= {self._alert_low}")
            self.alert_status.setStyleSheet("color: #4caf50; font-size: 11px; font-weight: bold;")
        else:
            self.alert_status.setText(f"PCR {vol_ratio:.2f} — normal")
            self.alert_status.setStyleSheet("color: #888; font-size: 11px; font-weight: normal;")
            self._last_alert_direction = ""
            return

        if direction and direction != self._last_alert_direction:
            self._last_alert_direction = direction
            self.alert_triggered.emit(self.ticker, direction, vol_ratio)

    def _on_video_resize(self, value):
        h = int(value * 140 / 250)
        self.metrics.video_indicator.video_widget.setFixedSize(value, h)
        self.metrics.video_indicator.setFixedHeight(h)

    def _pop_out_chart(self):
        """Open the 3D chart in a separate resizable window."""
        if self._popout_window:
            self._popout_window.close()
        self._popout_window = QMainWindow()
        self._popout_window.setWindowTitle(f"{self.ticker} — 3D Chart (Pop Out)")
        self._popout_window.resize(1000, 700)
        popout_chart = PlotlyChartWidget()
        popout_chart.theme = self.chart.theme
        self._popout_window.setCentralWidget(popout_chart)
        self._popout_window.show()
        # Render same data
        if self._last_histories:
            chart_expiries = self.get_chart_expiries()
            popout_chart.update_chart_3d(
                self._last_histories, self.ticker, chart_expiries,
                connect_expiries=self.connect_checkbox.isChecked())

    def _export_csv(self):
        """Export all history data for this ticker to CSV."""
        if not self._last_histories:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export CSV", f"{self.ticker}_pcr_data.csv", "CSV Files (*.csv)")
        if not path:
            return
        rows = []
        for key, points in self._last_histories.items():
            if key.startswith(f"{self.ticker}_"):
                for p in points:
                    rows.append(asdict(p))
        if rows:
            with open(path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            logger.info("Exported %d rows to %s", len(rows), path)

    def _export_png(self):
        """Screenshot the current 3D chart via grab()."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Export PNG", f"{self.ticker}_chart.png", "PNG Files (*.png)")
        if not path:
            return
        pixmap = self.chart.grab()
        pixmap.save(path)
        logger.info("Chart screenshot saved to %s", path)

    def update_data(self, dp: DataPoint, histories: dict[str, list]):
        self.error_label.hide()
        self.chart.show()
        self.metrics.update_metrics(dp)
        self.expiry_label.setText(f"Expiry: {dp.expiry}")
        self._last_histories = histories
        self._check_alerts(dp.vol_ratio)
        chart_expiries = self.get_chart_expiries()
        self.chart.update_chart_3d(histories, self.ticker, chart_expiries,
                                   connect_expiries=self.connect_checkbox.isChecked())
        # Also update heatmap
        self.heatmap.update_heatmap(histories, self.ticker, chart_expiries)

    def show_error(self, msg: str):
        self.error_label.setText(msg)
        self.error_label.show()

    def apply_theme(self, theme: dict):
        """Apply theme to chart widgets."""
        self.chart.theme = theme
        self.heatmap.theme = theme


class SidebarWidget(QWidget):
    """Left sidebar for ticker selection, API key config, theme, and sessions."""
    tickers_changed = pyqtSignal(list)
    api_key_changed = pyqtSignal(str)
    theme_toggled = pyqtSignal()       # Toggle dark/light
    save_session = pyqtSignal()
    load_session = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(200)
        layout = QVBoxLayout(self)

        title = QLabel("PCR Dashboard")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #fff; padding: 8px 0;")
        layout.addWidget(title)

        # API Key section
        layout.addWidget(QLabel("Tradier API Key:"))
        self.api_key_input = QLineEdit()
        self.api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key_input.setPlaceholderText("Enter API key...")
        self.api_key_input.setStyleSheet(
            "background-color: #2a2a2a; color: #ddd; border: 1px solid #444;"
            "border-radius: 4px; padding: 4px;"
        )
        layout.addWidget(self.api_key_input)

        save_key_btn = QPushButton("Save Key")
        save_key_btn.setStyleSheet(
            "background-color: #388e3c; color: white; padding: 4px; border-radius: 4px;"
        )
        save_key_btn.clicked.connect(self._on_save_key)
        layout.addWidget(save_key_btn)

        self.key_status = QLabel("")
        self.key_status.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(self.key_status)

        # Load saved key
        cfg = load_config()
        saved_key = cfg.get("tradier_api_key", "")
        if saved_key:
            self.api_key_input.setText(saved_key)
            self.key_status.setText("Key loaded from config")
            self.key_status.setStyleSheet("color: #388e3c; font-size: 11px;")

        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color: #444;")
        layout.addWidget(sep)

        # --- Theme toggle + Session buttons ---
        controls_row = QHBoxLayout()
        self.theme_btn = QPushButton("\u263e Dark")
        self.theme_btn.setStyleSheet(
            "background-color: #333; color: #ddd; padding: 4px 6px;"
            "border-radius: 4px; font-size: 11px;"
        )
        self.theme_btn.setToolTip("Toggle dark / light theme")
        self.theme_btn.clicked.connect(self.theme_toggled.emit)
        controls_row.addWidget(self.theme_btn)

        save_sess_btn = QPushButton("\U0001f4be Save")
        save_sess_btn.setStyleSheet(
            "background-color: #333; color: #ddd; padding: 4px 6px;"
            "border-radius: 4px; font-size: 11px;"
        )
        save_sess_btn.setToolTip("Save session data to disk")
        save_sess_btn.clicked.connect(self.save_session.emit)
        controls_row.addWidget(save_sess_btn)

        load_sess_btn = QPushButton("\U0001f4c2 Load")
        load_sess_btn.setStyleSheet(
            "background-color: #333; color: #ddd; padding: 4px 6px;"
            "border-radius: 4px; font-size: 11px;"
        )
        load_sess_btn.setToolTip("Load a saved session")
        load_sess_btn.clicked.connect(self.load_session.emit)
        controls_row.addWidget(load_sess_btn)

        layout.addLayout(controls_row)

        # Separator
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.HLine)
        sep2.setStyleSheet("color: #444;")
        layout.addWidget(sep2)

        layout.addWidget(QLabel("Presets:"))
        self.preset_list = QListWidget()
        self.preset_list.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        self.preset_list.addItems(PRESET_TICKERS)
        self.preset_list.setStyleSheet("""
            QListWidget {
                background-color: #2a2a2a; color: #ddd; border: 1px solid #444;
                border-radius: 4px; font-size: 13px;
            }
            QListWidget::item:selected {
                background-color: #1a73e8; color: white;
            }
        """)
        layout.addWidget(self.preset_list, stretch=1)

        # Select SPY by default (reliable for options)
        for i in range(self.preset_list.count()):
            item = self.preset_list.item(i)
            if item.text() == "SPY":
                item.setSelected(True)

        layout.addWidget(QLabel("Custom Ticker:"))
        self.custom_input = QLineEdit()
        self.custom_input.setPlaceholderText("e.g. AAPL")
        self.custom_input.setStyleSheet(
            "background-color: #2a2a2a; color: #ddd; border: 1px solid #444;"
            "border-radius: 4px; padding: 4px;"
        )
        layout.addWidget(self.custom_input)

        add_btn = QPushButton("Add Custom")
        add_btn.setStyleSheet(
            "background-color: #1a73e8; color: white; padding: 6px; border-radius: 4px;"
        )
        add_btn.clicked.connect(self._on_add_custom)
        layout.addWidget(add_btn)

        refresh_btn = QPushButton("Refresh Now")
        refresh_btn.setStyleSheet(
            "background-color: #444; color: white; padding: 6px; border-radius: 4px;"
        )
        refresh_btn.clicked.connect(self._emit_tickers)
        layout.addWidget(refresh_btn)

        self.preset_list.itemSelectionChanged.connect(self._emit_tickers)

    def update_theme_button(self, theme: dict):
        """Update theme button text/icon to reflect current theme."""
        if theme['name'] == 'dark':
            self.theme_btn.setText("\u2600 Light")
            self.theme_btn.setToolTip("Switch to light theme")
        else:
            self.theme_btn.setText("\u263e Dark")
            self.theme_btn.setToolTip("Switch to dark theme")

    def _on_save_key(self):
        key = self.api_key_input.text().strip()
        if key:
            cfg = load_config()
            cfg["tradier_api_key"] = key
            save_config(cfg)
            self.key_status.setText("Key saved!")
            self.key_status.setStyleSheet("color: #388e3c; font-size: 11px;")
            self.api_key_changed.emit(key)
            logger.info("API key saved and applied")
        else:
            self.key_status.setText("Key is empty")
            self.key_status.setStyleSheet("color: #d32f2f; font-size: 11px;")

    def get_api_key(self) -> str:
        return self.api_key_input.text().strip()

    def _on_add_custom(self):
        text = self.custom_input.text().strip().upper()
        if text:
            # Add to list if not already present
            existing = [self.preset_list.item(i).text() for i in range(self.preset_list.count())]
            if text not in existing:
                self.preset_list.addItem(text)
                item = self.preset_list.item(self.preset_list.count() - 1)
                item.setSelected(True)
            self.custom_input.clear()
            self._emit_tickers()

    def _emit_tickers(self):
        selected = [item.text() for item in self.preset_list.selectedItems()]
        if selected:
            self.tickers_changed.emit(selected)

    def get_selected_tickers(self) -> list[str]:
        return [item.text() for item in self.preset_list.selectedItems()]


class MainWindow(QMainWindow):
    """Main application window with theme, session save/load, and alert sounds."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Multi PCR Dashboard")
        icon_path = resource_path('app_icon.ico')
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        self.resize(1400, 900)

        # Theme state
        self.current_theme = DARK_THEME
        self._apply_global_theme()

        # Layout: splitter with sidebar on left, tabs on right
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.sidebar = SidebarWidget()
        splitter.addWidget(self.sidebar)

        self.tab_widget = QTabWidget()
        splitter.addWidget(self.tab_widget)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        self.setCentralWidget(splitter)

        # State
        self.histories: dict[str, list[DataPoint]] = {}
        self.ticker_tabs: dict[str, TickerTab] = {}
        self.active_tickers: list[str] = []

        # Alert sound player (uses system bell as fallback)
        self.alert_player = QMediaPlayer()
        self.alert_audio_out = QAudioOutput()
        self.alert_audio_out.setVolume(0.7)
        self.alert_player.setAudioOutput(self.alert_audio_out)

        # Data fetcher
        self.fetcher = DataFetcher()
        self.fetcher.api_key = self.sidebar.get_api_key()
        self.fetcher.data_ready.connect(self.on_data_ready)
        self.fetcher.expiries_ready.connect(self.on_expiries_ready)
        self.fetcher.status_update.connect(self.on_status_update)

        # Timers
        self.refresh_timer = QTimer(self)
        self.refresh_timer.timeout.connect(self.start_refresh)
        self.refresh_timer.start(REFRESH_INTERVAL_MS)

        self.countdown = REFRESH_INTERVAL_MS // 1000
        self.countdown_timer = QTimer(self)
        self.countdown_timer.timeout.connect(self._update_countdown)
        self.countdown_timer.start(1000)

        # Status bar
        self.status_label = QLabel("Starting...")
        self.countdown_label = QLabel("")
        self.statusBar().addWidget(self.status_label, 1)
        self.statusBar().addPermanentWidget(self.countdown_label)

        # Connect sidebar signals
        self.sidebar.tickers_changed.connect(self.on_tickers_changed)
        self.sidebar.api_key_changed.connect(self.on_api_key_changed)
        self.sidebar.theme_toggled.connect(self.toggle_theme)
        self.sidebar.save_session.connect(self.save_session)
        self.sidebar.load_session.connect(self.load_session)
        self.tab_widget.currentChanged.connect(self.on_tab_changed)

        # Initial load
        initial = self.sidebar.get_selected_tickers()
        if initial:
            self.on_tickers_changed(initial)

    # --- Theme ---
    def _apply_global_theme(self):
        t = self.current_theme
        self.setStyleSheet(f"""
            QMainWindow, QWidget {{
                background-color: {t['bg']};
                color: {t['fg']};
            }}
            QTabWidget::pane {{
                border: 1px solid {t['border']};
                background-color: {t['bg']};
            }}
            QTabBar::tab {{
                background-color: {t['bg2']};
                color: {t['fg2']};
                padding: 8px 16px;
                border: 1px solid {t['border']};
                border-bottom: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }}
            QTabBar::tab:selected {{
                background-color: {t['bg']};
                color: {t['fg']};
            }}
            QLabel {{
                color: {t['fg']};
            }}
            QComboBox {{
                background-color: {t['bg2']};
                color: {t['fg']};
                border: 1px solid {t['border']};
                border-radius: 4px;
                padding: 4px;
            }}
            QStatusBar {{
                background-color: {t['bg3']};
                color: {t['fg3']};
            }}
            QDoubleSpinBox {{
                background-color: {t['bg2']};
                color: {t['fg']};
                border: 1px solid {t['border']};
            }}
            QSlider::groove:horizontal {{
                background: {t['border']};
                height: 4px;
                border-radius: 2px;
            }}
            QSlider::handle:horizontal {{
                background: {t['accent']};
                width: 12px;
                height: 12px;
                margin: -4px 0;
                border-radius: 6px;
            }}
        """)

    def toggle_theme(self):
        if self.current_theme['name'] == 'dark':
            self.current_theme = LIGHT_THEME
        else:
            self.current_theme = DARK_THEME
        self._apply_global_theme()
        self.sidebar.update_theme_button(self.current_theme)
        # Apply to all open ticker tabs
        for tab in self.ticker_tabs.values():
            tab.apply_theme(self.current_theme)
        logger.info("Theme switched to %s", self.current_theme['name'])

    # --- Session save/load ---
    def save_session(self):
        """Save all history data to a JSON file."""
        os.makedirs(SESSIONS_DIR, exist_ok=True)
        default_name = datetime.now().strftime("session_%Y%m%d_%H%M%S.json")
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Session", os.path.join(SESSIONS_DIR, default_name),
            "JSON Files (*.json)")
        if not path:
            return
        session_data = {
            'tickers': self.active_tickers,
            'expiries': {},
            'histories': {},
        }
        for ticker, tab in self.ticker_tabs.items():
            session_data['expiries'][ticker] = tab.all_expiries
        for key, points in self.histories.items():
            session_data['histories'][key] = [asdict(p) for p in points]
        try:
            with open(path, 'w') as f:
                json.dump(session_data, f, indent=2)
            logger.info("Session saved to %s (%d history keys)", path, len(self.histories))
            self.status_label.setText(f"Session saved: {os.path.basename(path)}")
        except Exception:
            logger.exception("Failed to save session")
            self.status_label.setText("ERROR: Failed to save session")

    def load_session(self):
        """Load a previously saved session from JSON."""
        os.makedirs(SESSIONS_DIR, exist_ok=True)
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Session", SESSIONS_DIR, "JSON Files (*.json)")
        if not path:
            return
        try:
            with open(path, 'r') as f:
                session_data = json.load(f)
        except Exception:
            logger.exception("Failed to load session file")
            self.status_label.setText("ERROR: Failed to load session")
            return

        # Restore histories
        self.histories.clear()
        for key, points_raw in session_data.get('histories', {}).items():
            self.histories[key] = [DataPoint(**p) for p in points_raw]

        # Restore tickers — create tabs and set expiries
        tickers = session_data.get('tickers', [])
        if tickers:
            # Select matching tickers in sidebar
            for i in range(self.sidebar.preset_list.count()):
                item = self.sidebar.preset_list.item(i)
                item.setSelected(item.text() in tickers)

            # Create tabs
            self.on_tickers_changed(tickers)

            # Restore expiries and replay last data point to charts
            for ticker, tab in self.ticker_tabs.items():
                saved_expiries = session_data.get('expiries', {}).get(ticker, [])
                if saved_expiries:
                    tab.set_expiries(saved_expiries)
                # Find last data point for the primary expiry
                primary_key = f"{ticker}_{tab.current_expiry}"
                if primary_key in self.histories and self.histories[primary_key]:
                    last_dp = self.histories[primary_key][-1]
                    tab.update_data(last_dp, self.histories)

        logger.info("Session loaded from %s (%d history keys)", path, len(self.histories))
        self.status_label.setText(f"Session loaded: {os.path.basename(path)}")

    # --- Alert handling ---
    def _on_alert_triggered(self, ticker: str, direction: str, value: float):
        """Handle alert from a TickerTab — play a sound and show status."""
        logger.info("ALERT: %s PCR %s — value %.3f", ticker, direction, value)
        self.status_label.setText(
            f"\u26a0 ALERT: {ticker} PCR {direction} ({value:.2f})")
        # Try to play an alert sound — reuse bearish/bullish audio
        if direction == "HIGH":
            audio_file = resource_path('bearish.m4a')
        else:
            audio_file = resource_path('bullish.m4a')
        if os.path.exists(audio_file):
            self.alert_player.stop()
            self.alert_player.setSource(QUrl.fromLocalFile(audio_file))
            self.alert_player.play()
        else:
            # Fallback: system bell
            QApplication.beep()

    # --- Tab management ---
    def on_tab_changed(self, index: int):
        """Pause audio on all tabs except the active one."""
        for t, tab in self.ticker_tabs.items():
            if self.tab_widget.indexOf(tab) == index:
                tab.metrics.video_indicator.resume_audio()
            else:
                tab.metrics.video_indicator.pause_audio()

    def on_api_key_changed(self, key: str):
        """API key was saved — update fetcher and trigger refresh."""
        self.fetcher.api_key = key
        logger.info("API key updated, triggering refresh")
        self.start_refresh(fetch_expiries_for=self.active_tickers)

    def on_tickers_changed(self, tickers: list[str]):
        logger.info("Tickers changed: %s", tickers)
        # Remove tabs for deselected tickers
        for t in list(self.ticker_tabs.keys()):
            if t not in tickers:
                idx = self.tab_widget.indexOf(self.ticker_tabs[t])
                if idx >= 0:
                    self.tab_widget.removeTab(idx)
                del self.ticker_tabs[t]

        # Add tabs for new tickers
        new_tickers = []
        for t in tickers:
            if t not in self.ticker_tabs:
                tab = TickerTab(t)
                tab.expiry_changed.connect(self.on_expiry_changed)
                tab.alert_triggered.connect(self._on_alert_triggered)
                tab.apply_theme(self.current_theme)
                self.ticker_tabs[t] = tab
                self.tab_widget.addTab(tab, t)
                new_tickers.append(t)

        self.active_tickers = tickers
        self.start_refresh(fetch_expiries_for=new_tickers if new_tickers else tickers)

    def on_expiry_changed(self, ticker: str, expiry: str):
        """User changed the expiry combo box — fetch primary + next expiries."""
        logger.info("Expiry changed for %s -> %s", ticker, expiry)
        tab = self.ticker_tabs.get(ticker)
        if self.fetcher.isRunning():
            self.fetcher.requestInterruption()
            self.fetcher.wait(2000)
        if tab:
            tasks = [(ticker, e) for e in tab.get_chart_expiries()]
        else:
            tasks = [(ticker, expiry)]
        self.fetcher.set_tasks(tasks)
        self.fetcher.start()

    def start_refresh(self, fetch_expiries_for=None):
        self.countdown = REFRESH_INTERVAL_MS // 1000
        if self.fetcher.isRunning():
            self.fetcher.requestInterruption()
            self.fetcher.wait(2000)

        # Build task list: for each ticker, fetch primary + next expiries
        tasks = []
        for t in self.active_tickers:
            tab = self.ticker_tabs.get(t)
            if tab and tab.current_expiry:
                for exp in tab.get_chart_expiries():
                    tasks.append((t, exp))

        need_expiries = fetch_expiries_for or []
        # If tickers have no expiry yet, we need to fetch them
        for t in self.active_tickers:
            tab = self.ticker_tabs.get(t)
            if tab and not tab.current_expiry and t not in need_expiries:
                need_expiries.append(t)

        self.fetcher.set_tasks(tasks, fetch_expiries_for=need_expiries)
        self.fetcher.start()

    def on_data_ready(self, ticker: str, expiry: str, dp):
        tab = self.ticker_tabs.get(ticker)
        if not tab:
            logger.warning("Data arrived for unknown ticker %s (tab removed?)", ticker)
            return
        if dp is None:
            logger.warning("No data for %s/%s — showing error in UI", ticker, expiry)
            # Only show error if it's the primary expiry
            if expiry == tab.current_expiry:
                tab.show_error(f"No data for {ticker}/{expiry} (market closed or invalid?)")
            return

        key = f"{ticker}_{expiry}"
        if key not in self.histories:
            self.histories[key] = []
        self.histories[key].append(dp)
        self.histories[key] = self.histories[key][-MAX_HISTORY:]

        # Update the tab — metrics use the primary expiry's data, chart uses all
        if expiry == tab.current_expiry:
            tab.update_data(dp, self.histories)
        else:
            # Non-primary expiry arrived — just refresh the 3D chart
            tab._last_histories = self.histories
            chart_expiries = tab.get_chart_expiries()
            tab.chart.update_chart_3d(self.histories, ticker, chart_expiries,
                                      connect_expiries=tab.connect_checkbox.isChecked())

    def on_expiries_ready(self, ticker: str, expiries: list[str]):
        logger.info("Expiries ready for %s: %d options", ticker, len(expiries))
        tab = self.ticker_tabs.get(ticker)
        if tab:
            tab.set_expiries(expiries)
            # After setting expiries, trigger a data fetch for the first expiry
            if expiries:
                self.fetcher.set_tasks([(ticker, expiries[0])])
                if not self.fetcher.isRunning():
                    self.fetcher.start()

    def on_status_update(self, msg: str):
        self.status_label.setText(msg)

    def _update_countdown(self):
        self.countdown = max(0, self.countdown - 1)
        self.countdown_label.setText(f"Next refresh: {self.countdown}s")


def main():
    # PyInstaller + QtWebEngine compatibility
    if getattr(sys, 'frozen', False):
        os.environ['QTWEBENGINE_CHROMIUM_FLAGS'] = '--no-sandbox'
        os.chdir(os.path.dirname(sys.executable))
        try:
            import certifi
            os.environ['SSL_CERT_FILE'] = certifi.where()
            os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
        except ImportError:
            pass

    logger.info("=" * 60)
    logger.info("Multi PCR Dashboard starting (Python %s, frozen=%s)",
                sys.version.split()[0], getattr(sys, 'frozen', False))
    logger.info("Log file: %s", LOG_FILE)

    app = QApplication(sys.argv)
    app.setApplicationName("Multi PCR Dashboard")
    window = MainWindow()
    window.show()
    logger.info("Window shown — entering event loop")
    exit_code = app.exec()
    logger.info("Application exiting with code %d", exit_code)
    sys.exit(exit_code)


if __name__ == '__main__':
    if sys.platform == 'win32':
        import multiprocessing
        multiprocessing.freeze_support()
    main()
