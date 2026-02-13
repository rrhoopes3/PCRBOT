import sys
import os
import time
import logging
import logging.handlers
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import json
import requests

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal, QUrl
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QLabel, QComboBox, QListWidget, QLineEdit,
    QPushButton, QAbstractItemView, QFrame, QSplitter, QSizePolicy
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
NUM_3D_EXPIRIES = 5  # Number of expiries to display in 3D chart

# --- Tradier API config ---
TRADIER_BASE_URL = "https://api.tradier.com/v1"
TRADIER_HEADERS = {"Accept": "application/json"}

CONFIG_FILE = os.path.join(
    os.path.dirname(sys.executable) if getattr(sys, 'frozen', False)
    else os.path.dirname(os.path.abspath(__file__)),
    'pcr_config.json'
)


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
                expiries = dates[:10]
                logger.info("Expiries for %s: %s", ticker, expiries)
                self.expiries_ready.emit(ticker, expiries)
            else:
                logger.warning("No expiries returned for %s: %s", ticker, data)
                self.expiries_ready.emit(ticker, [])
            time.sleep(0.2)

        # Then fetch option chain data for each ticker/expiry pair
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
                timestamp=datetime.now().strftime('%H:%M:%S'),
            )
            logger.info("Data OK for %s/%s — vol_ratio=%.3f oi_ratio=%.3f put_vol=%d call_vol=%d",
                        ticker, expiry, vol_ratio, oi_ratio, dp.put_vol, dp.call_vol)
            self.data_ready.emit(ticker, expiry, dp)
            time.sleep(0.2)

        logger.info("DataFetcher refresh cycle complete")
        self.status_update.emit("Refresh complete.")
        self.all_done.emit()


class PlotlyChartWidget(QWebEngineView):
    """Renders Plotly charts — supports 3D multi-expiry layered view."""

    # Color palette for expiry layers
    LAYER_COLORS = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(350)
        self.setHtml("<html><body style='background:#1e1e1e;color:#aaa;display:flex;"
                     "align-items:center;justify-content:center;height:100vh;margin:0;'>"
                     "<p>Waiting for data...</p></body></html>")

    def update_chart_3d(self, histories: dict[str, list], ticker: str, expiries: list[str]):
        """Render a 3D layered chart: X=time, Y=expiry layer, Z=ratio.
        Each expiry gets its own 'lane' on the Y axis so they stack like a 3D cube."""
        if not expiries or not histories:
            return

        fig = go.Figure()

        for i, expiry in enumerate(expiries[:NUM_3D_EXPIRIES]):
            key = f"{ticker}_{expiry}"
            points = histories.get(key, [])
            if not points:
                continue

            timestamps = [p.timestamp for p in points]
            vol_ratios = [p.vol_ratio for p in points]
            oi_ratios = [p.oi_ratio for p in points]
            y_pos = [i] * len(points)  # Each expiry on its own Y lane
            color = self.LAYER_COLORS[i % len(self.LAYER_COLORS)]

            # Short label for legend
            short_exp = expiry[5:] if len(expiry) > 5 else expiry  # "02-14" from "2026-02-14"

            # Vol ratio line
            fig.add_trace(go.Scatter3d(
                x=timestamps, y=y_pos, z=vol_ratios,
                mode='lines+markers',
                line=dict(color=color, width=4),
                marker=dict(size=3, color=color),
                name=f"{short_exp} Vol",
                legendgroup=expiry,
            ))
            # OI ratio line — same Y lane, dashed
            fig.add_trace(go.Scatter3d(
                x=timestamps, y=y_pos, z=oi_ratios,
                mode='lines+markers',
                line=dict(color=color, width=3, dash='dash'),
                marker=dict(size=2, color=color, symbol='diamond'),
                name=f"{short_exp} OI",
                legendgroup=expiry,
            ))

        # Y-axis tick labels = expiry dates
        active_expiries = [e for e in expiries[:NUM_3D_EXPIRIES]
                           if f"{ticker}_{e}" in histories and histories[f"{ticker}_{e}"]]
        y_tickvals = list(range(len(active_expiries)))
        y_ticktext = [e[5:] for e in active_expiries]

        fig.update_layout(
            template='plotly_dark',
            title=dict(text=f"{ticker} — Multi-Expiry 3D Ratios", font=dict(size=16)),
            margin=dict(l=0, r=0, t=40, b=0),
            scene=dict(
                xaxis=dict(title='Time', showgrid=True, gridcolor='#333'),
                yaxis=dict(title='Expiry', tickvals=y_tickvals, ticktext=y_ticktext,
                           showgrid=True, gridcolor='#333'),
                zaxis=dict(title='Ratio', showgrid=True, gridcolor='#333'),
                bgcolor='#1e1e1e',
                camera=dict(
                    eye=dict(x=1.8, y=-1.4, z=0.8),  # Angled view to see all layers
                    up=dict(x=0, y=0, z=1),
                ),
            ),
            legend=dict(
                font=dict(size=10),
                bgcolor='rgba(30,30,30,0.8)',
                bordercolor='#444',
                borderwidth=1,
            ),
            paper_bgcolor='#1e1e1e',
        )

        html = fig.to_html(include_plotlyjs='cdn', full_html=True)
        self.setHtml(html)


class VideoIndicator(QWidget):
    """Plays bearish.mp4, bullish.mp4, or neutral.mp4 based on sentiment.
    Compact single-video widget designed to sit in the metrics row."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.video_widget = QVideoWidget()
        self.video_widget.setMinimumSize(140, 80)
        self.player = QMediaPlayer()
        self.audio = QAudioOutput()
        self.audio.setVolume(0)
        self.player.setAudioOutput(self.audio)
        self.player.setVideoOutput(self.video_widget)
        self.player.setLoops(QMediaPlayer.Loops.Infinite)

        layout.addWidget(self.video_widget)

        self._current_video = ""

    def update_sentiment(self, vol_ratio: float):
        if pd.isna(vol_ratio):
            self.player.stop()
            return

        if vol_ratio > 1:
            video_file = 'bearish.mp4'
        elif vol_ratio < 1:
            video_file = 'bullish.mp4'
        else:
            video_file = 'neutral.mp4'

        # Avoid restarting the same video
        if video_file == self._current_video:
            return

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
    """One tab per ticker: expiry selector, metrics, 3D chart, video indicator."""
    expiry_changed = pyqtSignal(str, str)  # ticker, new_expiry

    def __init__(self, ticker: str, parent=None):
        super().__init__(parent)
        self.ticker = ticker
        self.current_expiry = ""
        self.all_expiries: list[str] = []  # Full expiry list for 3D chart

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # Top row: expiry selector (metrics shown for selected expiry)
        top = QHBoxLayout()
        top.addWidget(QLabel("Primary Expiry:"))
        self.expiry_combo = QComboBox()
        self.expiry_combo.setMinimumWidth(160)
        self.expiry_combo.currentTextChanged.connect(self._on_expiry_changed)
        top.addWidget(self.expiry_combo)
        self.expiry_label = QLabel("")
        self.expiry_label.setStyleSheet("color: #888;")
        top.addWidget(self.expiry_label)
        # Info label about 3D
        info_label = QLabel(f"(3D chart shows nearest {NUM_3D_EXPIRIES} expiries)")
        info_label.setStyleSheet("color: #666; font-size: 11px;")
        top.addWidget(info_label)
        top.addStretch()
        layout.addLayout(top)

        # Metrics + video indicator (for the primary/selected expiry)
        self.metrics = MetricsPanel()
        layout.addWidget(self.metrics)

        # 3D Chart
        self.chart = PlotlyChartWidget()
        self.chart.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self.chart, stretch=1)

        # Error label (hidden by default)
        self.error_label = QLabel("")
        self.error_label.setStyleSheet("color: #ff6b6b; font-size: 14px; padding: 20px;")
        self.error_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.error_label.hide()
        layout.addWidget(self.error_label)

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
        """Return the primary expiry + the next NUM_3D_EXPIRIES after it."""
        if not self.all_expiries or not self.current_expiry:
            return []
        try:
            idx = self.all_expiries.index(self.current_expiry)
        except ValueError:
            idx = 0
        # Primary + next N
        return self.all_expiries[idx:idx + 1 + NUM_3D_EXPIRIES]

    def update_data(self, dp: DataPoint, histories: dict[str, list]):
        """Update metrics for the primary expiry + render the 3D multi-expiry chart."""
        self.error_label.hide()
        self.chart.show()
        self.metrics.update_metrics(dp)
        self.expiry_label.setText(f"Expiry: {dp.expiry}")
        # 3D chart: primary + next 5 expiries
        chart_expiries = self.get_chart_expiries()
        self.chart.update_chart_3d(histories, self.ticker, chart_expiries)

    def show_error(self, msg: str):
        self.error_label.setText(msg)
        self.error_label.show()


class SidebarWidget(QWidget):
    """Left sidebar for ticker selection and API key config."""
    tickers_changed = pyqtSignal(list)
    api_key_changed = pyqtSignal(str)

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
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Multi PCR Dashboard")
        self.resize(1400, 900)
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #1e1e1e;
                color: #ddd;
            }
            QTabWidget::pane {
                border: 1px solid #444;
                background-color: #1e1e1e;
            }
            QTabBar::tab {
                background-color: #2a2a2a;
                color: #aaa;
                padding: 8px 16px;
                border: 1px solid #444;
                border-bottom: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: #1e1e1e;
                color: #fff;
            }
            QLabel {
                color: #ddd;
            }
            QComboBox {
                background-color: #2a2a2a;
                color: #ddd;
                border: 1px solid #444;
                border-radius: 4px;
                padding: 4px;
            }
            QStatusBar {
                background-color: #252525;
                color: #888;
            }
        """)

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

        # Connect sidebar
        self.sidebar.tickers_changed.connect(self.on_tickers_changed)
        self.sidebar.api_key_changed.connect(self.on_api_key_changed)

        # Initial load
        initial = self.sidebar.get_selected_tickers()
        if initial:
            self.on_tickers_changed(initial)

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
                self.ticker_tabs[t] = tab
                self.tab_widget.addTab(tab, t)
                new_tickers.append(t)

        self.active_tickers = tickers
        self.start_refresh(fetch_expiries_for=new_tickers if new_tickers else tickers)

    def on_expiry_changed(self, ticker: str, expiry: str):
        """User changed the expiry combo box — fetch primary + next 5 expiries."""
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

        # Build task list: for each ticker, fetch primary + next 5 expiries
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
            chart_expiries = tab.get_chart_expiries()
            tab.chart.update_chart_3d(self.histories, ticker, chart_expiries)

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
        os.environ['QTWEBENGINE_CHROMIUM_FLAGS'] = '--no-sandbox --disable-gpu'
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
