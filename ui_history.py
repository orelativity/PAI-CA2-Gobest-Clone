import tkinter as tk
from tkinter import ttk

from .db import fetch_db_stats, fetch_recent_predictions, fetch_top_drivers


class HistoryFrame(ttk.Frame):
    def __init__(self, master):
        super().__init__(master, padding=12)
        self._build()

    def _build(self):
        ttk.Label(self, text="History (SQLite Database)", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(
            self,
            text="This tab displays persistent prediction history saved by the app across runs.",
            style="Sub.TLabel",
            wraplength=860,
        ).grid(row=1, column=0, sticky="w", pady=(2, 12))

        # stats card
        self.stats_card = ttk.LabelFrame(self, text="Database Summary", padding=12)
        self.stats_card.grid(row=2, column=0, sticky="ew")
        self.stats_text = ttk.Label(self.stats_card, text="", style="Hint.TLabel", wraplength=860)
        self.stats_text.grid(row=0, column=0, sticky="w")

        # two columns: recent + top drivers
        grid = ttk.Frame(self)
        grid.grid(row=3, column=0, sticky="nsew", pady=(12, 0))
        grid.columnconfigure(0, weight=1)
        grid.columnconfigure(1, weight=1)
        grid.rowconfigure(0, weight=1)

        self.recent_card = ttk.LabelFrame(grid, text="Recent Predictions", padding=10)
        self.recent_card.grid(row=0, column=0, sticky="nsew", padx=(0, 8))

        self.top_card = ttk.LabelFrame(grid, text="Top Drivers (by dangerous_rate)", padding=10)
        self.top_card.grid(row=0, column=1, sticky="nsew", padx=(8, 0))

        self.recent_box = tk.Text(self.recent_card, height=16, wrap="none")
        self.recent_box.pack(fill="both", expand=True)

        self.top_box = tk.Text(self.top_card, height=16, wrap="none")
        self.top_box.pack(fill="both", expand=True)

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(3, weight=1)

        self.refresh()

    def refresh(self):
        # stats
        s = fetch_db_stats()
        self.stats_text.config(
            text=(
                f"Total predictions stored: {s['total_preds']}\n"
                f"Unique drivers tracked: {s['total_drivers']}\n"
                f"Drivers with dangerous_rate ≥ 0.50: {s['high_risk_drivers']}"
            )
        )

        # recent predictions
        recent = fetch_recent_predictions(limit=12)
        self.recent_box.delete("1.0", "end")
        if not recent:
            self.recent_box.insert("end", "No history yet.\n\nRun Batch Prediction to generate results.\n")
        else:
            for created_at, bid, did, proba, label, thr in recent:
                self.recent_box.insert(
                    "end",
                    f"{created_at} | bookingID={bid} | driver={did} | p={proba:.3f} | pred={label} | thr={thr:.2f}\n"
                )

        # top drivers
        top = fetch_top_drivers(limit=10)
        self.top_box.delete("1.0", "end")
        if not top:
            self.top_box.insert("end", "No driver history yet.\n\nRun Batch Prediction to populate driver history.\n")
        else:
            for did, total, dang, rate, avg_harsh, last_updated in top:
                self.top_box.insert(
                    "end",
                    f"driver={did} | trips={total} | dangerous={dang} | rate={rate:.2f} | avg_harsh={avg_harsh:.2f}\n"
                )
