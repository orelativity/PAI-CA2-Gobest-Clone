import tkinter as tk
from tkinter import ttk, messagebox

from .db import fetch_driver_history


class RealtimeFrame(ttk.Frame):
    def __init__(self, master, app):
        super().__init__(master, padding=12)
        self.app = app

        self.booking_choice = tk.StringVar()
        self.threshold = tk.DoubleVar(value=0.50)

        self._build()

    def _build(self):
        ttk.Label(self, text="Single Trip Prediction", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(
            self,
            text=(
                "Purpose: Choose a bookingID from the last loaded batch. "
                "This tab explains the prediction and shows driver history from the SQLite database.\n\n"
                "If you see no bookingIDs, run Batch Prediction first OR click the 'Load bookingIDs' button."
            ),
            style="Sub.TLabel",
            wraplength=900,
            justify="left",
        ).grid(row=1, column=0, sticky="w", pady=(4, 12))

        # selection card
        sel = ttk.LabelFrame(self, text="Step 1: Select bookingID", padding=12)
        sel.grid(row=2, column=0, sticky="ew")
        sel.columnconfigure(1, weight=1)

        ttk.Label(sel, text="bookingID:").grid(row=0, column=0, sticky="w", padx=(0, 10))
        self.combo = ttk.Combobox(sel, textvariable=self.booking_choice, state="readonly", values=[])
        self.combo.grid(row=0, column=1, sticky="ew")
        ttk.Button(sel, text="Refresh list", command=self.refresh_booking_list).grid(row=0, column=2, padx=(10, 0))

        # threshold card
        thr = ttk.LabelFrame(self, text="Step 2: Threshold", padding=12)
        thr.grid(row=3, column=0, sticky="ew", pady=(12, 0))
        thr.columnconfigure(0, weight=1)

        ttk.Label(
            thr,
            text="Rule: probability ≥ threshold → DANGEROUS, else SAFE.",
            style="Hint.TLabel",
        ).grid(row=0, column=0, sticky="w", pady=(0, 8))

        slider = ttk.Scale(thr, from_=0.10, to=0.90, variable=self.threshold, orient="horizontal")
        slider.grid(row=1, column=0, sticky="ew")

        ticks = ttk.Frame(thr)
        ticks.grid(row=2, column=0, sticky="ew", pady=(4, 0))
        ticks.columnconfigure(tuple(range(9)), weight=1)
        for i, v in enumerate([0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]):
            ttk.Label(ticks, text=f"{v:.2f}", style="Hint.TLabel").grid(row=0, column=i, sticky="n")

        self.thr_label = ttk.Label(thr, text=f"Current threshold: {self.threshold.get():.2f}", style="Sub.TLabel")
        self.thr_label.grid(row=3, column=0, sticky="w", pady=(8, 0))
        self.threshold.trace_add("write", self._on_thresh_change)

        # action card
        act = ttk.LabelFrame(self, text="Step 3: View prediction", padding=12)
        act.grid(row=4, column=0, sticky="ew", pady=(12, 0))

        ttk.Button(act, text="Show Result", command=self.show_result).grid(row=0, column=0, sticky="w")

        # result + history area
        grid = ttk.Frame(self)
        grid.grid(row=5, column=0, sticky="nsew", pady=(12, 0))
        grid.columnconfigure(0, weight=1)
        grid.columnconfigure(1, weight=1)
        grid.rowconfigure(0, weight=1)

        self.result_card = ttk.LabelFrame(grid, text="Prediction Output", padding=12)
        self.result_card.grid(row=0, column=0, sticky="nsew", padx=(0, 8))

        self.history_card = ttk.LabelFrame(grid, text="Driver History (SQLite)", padding=12)
        self.history_card.grid(row=0, column=1, sticky="nsew", padx=(8, 0))

        self.result_text = tk.Text(self.result_card, height=14, wrap="word")
        self.result_text.pack(fill="both", expand=True)
        self.history_text = tk.Text(self.history_card, height=14, wrap="word")
        self.history_text.pack(fill="both", expand=True)

        self._placeholder_state()

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(5, weight=1)

    def _placeholder_state(self):
        self.result_text.delete("1.0", "end")
        self.result_text.insert("end", "No result yet.\n\nLoad a batch and pick a bookingID, then click 'Show Result'.\n")

        self.history_text.delete("1.0", "end")
        self.history_text.insert("end", "No history yet.\n\nRun Batch Prediction to populate stored driver history.\n")

    def _on_thresh_change(self, *_):
        self.thr_label.config(text=f"Current threshold: {float(self.threshold.get()):.2f}")

    def refresh_booking_list(self):
        if self.app.preds is None or len(self.app.preds) == 0:
            self.combo["values"] = []
            self.booking_choice.set("")
            self._placeholder_state()
            return

        bids = sorted(self.app.preds["bookingID"].astype(str).unique().tolist())
        self.combo["values"] = bids
        if bids and (self.booking_choice.get() not in bids):
            self.booking_choice.set(bids[0])

    def show_result(self):
        if self.app.preds is None or len(self.app.preds) == 0:
            messagebox.showwarning("No data", "Run Batch Prediction first.")
            return

        bid = self.booking_choice.get().strip()
        if not bid:
            messagebox.showwarning("No selection", "Please choose a bookingID first.")
            return

        df = self.app.preds.copy()
        row = df[df["bookingID"].astype(str) == bid]
        if row.empty:
            messagebox.showerror("Not found", f"bookingID {bid} not found.")
            return
        row = row.iloc[0]

        proba = float(row["pred_proba"])
        thr = float(self.threshold.get())
        pred = int(proba >= thr)
        label = "DANGEROUS" if pred == 1 else "SAFE"

        # prediction explanation
        self.result_text.delete("1.0", "end")
        self.result_text.insert(
            "end",
            (
                f"bookingID: {bid}\n"
                f"driver_id: {int(row['driver_id'])}\n\n"
                f"predicted probability (dangerous): {proba:.3f}\n"
                f"threshold: {thr:.2f}\n"
                f"decision: probability ≥ threshold → {label}\n\n"
                "Interpretation:\n"
                "- Higher probability means the trip’s engineered behaviour features resemble dangerous trips.\n"
                "- Threshold controls strictness: lower threshold = more sensitive; higher threshold = more conservative.\n"
            )
        )

        # driver history
        hist = fetch_driver_history(int(row["driver_id"]))
        self.history_text.delete("1.0", "end")

        if not hist:
            self.history_text.insert(
                "end",
                (
                    "No history for this driver yet.\n\n"
                    "After you run Batch Prediction a few times, the database will accumulate:\n"
                    "- total trips seen\n"
                    "- number of predicted dangerous trips\n"
                    "- long-run dangerous rate\n"
                )
            )
        else:
            total_trips, dangerous_trips, dangerous_rate, avg_harsh, last_updated = hist
            self.history_text.insert(
                "end",
                (
                    f"driver_id: {int(row['driver_id'])}\n"
                    f"total trips stored: {int(total_trips)}\n"
                    f"dangerous trips stored: {int(dangerous_trips)}\n"
                    f"dangerous rate: {float(dangerous_rate):.2f}\n"
                    f"avg harsh acceleration: {float(avg_harsh):.2f}\n"
                    f"last updated: {last_updated}\n\n"
                    "Note:\n"
                    "This history is derived from predictions made by this GUI over time (persistent across runs).\n"
                )
            )
