import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import pandas as pd

from .model_utils import predict_from_raw


class BatchFrame(ttk.Frame):
    def __init__(self, master, app):
        super().__init__(master, padding=12)
        self.app = app

        self.sensor_path = tk.StringVar()
        self.driver_path = tk.StringVar()
        self.safety_path = tk.StringVar()

        self.threshold = tk.DoubleVar(value=0.50)

        self._build()

    def _build(self):
        ttk.Label(self, text="Batch Prediction", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(
            self,
            text=(
                "Purpose: Upload the 3 raw tables (sensor_data, driver_data, safety_labels). "
                "The app will generate the same engineered features used during training, then run the saved "
                "XGBoost classifier to predict whether each trip is dangerous.\n\n"
                "Output: A CSV containing predictions per bookingID + probability score.\n"
                "History: Every run is saved into a local SQLite database for persistent history."
            ),
            style="Sub.TLabel",
            wraplength=900,
            justify="left",
        ).grid(row=1, column=0, sticky="w", pady=(4, 12))

        # files card
        files = ttk.LabelFrame(self, text="Step 1: Select raw CSV files", padding=12)
        files.grid(row=2, column=0, sticky="ew")
        files.columnconfigure(1, weight=1)

        self._file_row(files, 0, "sensor_data.csv (must include bookingID)", self.sensor_path)
        self._file_row(files, 1, "driver_data.csv (must include id)", self.driver_path)
        self._file_row(files, 2, "safety_labels.csv (bookingID + driver_id)", self.safety_path)

        # threshold card
        thr = ttk.LabelFrame(self, text="Step 2: Choose threshold (decision cutoff)", padding=12)
        thr.grid(row=3, column=0, sticky="ew", pady=(12, 0))
        thr.columnconfigure(0, weight=1)

        ttk.Label(
            thr,
            text=(
                "Model output is a probability of dangerous driving.\n"
                "Rule: probability ≥ threshold → predicted dangerous.\n"
                "Lower threshold increases recall (catches more dangerous trips) but may increase false alarms."
            ),
            style="Hint.TLabel",
            wraplength=880,
            justify="left",
        ).grid(row=0, column=0, sticky="w", pady=(0, 10))

        slider = ttk.Scale(thr, from_=0.10, to=0.90, variable=self.threshold, orient="horizontal")
        slider.grid(row=1, column=0, sticky="ew")

        # tick labels
        ticks = ttk.Frame(thr)
        ticks.grid(row=2, column=0, sticky="ew", pady=(4, 0))
        ticks.columnconfigure(tuple(range(9)), weight=1)

        for i, v in enumerate([0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]):
            ttk.Label(ticks, text=f"{v:.2f}", style="Hint.TLabel").grid(row=0, column=i, sticky="n")

        # live label + entry
        row = ttk.Frame(thr)
        row.grid(row=3, column=0, sticky="ew", pady=(8, 0))
        row.columnconfigure(1, weight=1)

        self.thr_label = ttk.Label(row, text=f"Current threshold: {self.threshold.get():.2f}", style="Sub.TLabel")
        self.thr_label.grid(row=0, column=0, sticky="w")

        self.thr_entry = ttk.Entry(row, width=8)
        self.thr_entry.grid(row=0, column=2, sticky="e", padx=(10, 0))
        self.thr_entry.insert(0, f"{self.threshold.get():.2f}")

        ttk.Button(row, text="Set", command=self._apply_threshold_entry).grid(row=0, column=3, sticky="e", padx=(6, 0))

        self.threshold.trace_add("write", self._on_threshold_change)

        # run card
        run = ttk.LabelFrame(self, text="Step 3: Run batch prediction", padding=12)
        run.grid(row=4, column=0, sticky="ew", pady=(12, 0))
        run.columnconfigure(0, weight=1)

        btn_row = ttk.Frame(run)
        btn_row.grid(row=0, column=0, sticky="w")

        ttk.Button(btn_row, text="Run Batch Prediction", command=self._run).grid(row=0, column=0, padx=(0, 10))
        ttk.Button(btn_row, text="Load bookingIDs into Single Trip tab", command=self._push_to_single).grid(row=0, column=1)

        self.status = ttk.Label(run, text="Status: waiting for input…", style="Hint.TLabel")
        self.status.grid(row=1, column=0, sticky="w", pady=(10, 0))

        self.grid_columnconfigure(0, weight=1)

    def _file_row(self, parent, r, label, var):
        ttk.Label(parent, text=label).grid(row=r, column=0, sticky="w", padx=(0, 10), pady=6)
        ttk.Entry(parent, textvariable=var).grid(row=r, column=1, sticky="ew", pady=6)
        ttk.Button(parent, text="Browse", command=lambda: self._browse(var)).grid(row=r, column=2, sticky="e", pady=6)

    def _browse(self, var):
        p = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if p:
            var.set(p)

    def _on_threshold_change(self, *_):
        v = float(self.threshold.get())
        self.thr_label.config(text=f"Current threshold: {v:.2f}")
        self.thr_entry.delete(0, "end")
        self.thr_entry.insert(0, f"{v:.2f}")

    def _apply_threshold_entry(self):
        try:
            v = float(self.thr_entry.get().strip())
            if not (0.10 <= v <= 0.90):
                raise ValueError
            self.threshold.set(v)
        except Exception:
            messagebox.showerror("Invalid threshold", "Enter a number between 0.10 and 0.90")

    def _run(self):
        sp = self.sensor_path.get().strip()
        dp = self.driver_path.get().strip()
        lp = self.safety_path.get().strip()

        if not sp or not dp or not lp:
            messagebox.showerror("Missing files", "Please select all 3 raw CSV files first.")
            return

        try:
            self.status.config(text="Status: loading CSV files…")
            self.update_idletasks()

            sensor_df = pd.read_csv(sp)
            driver_df = pd.read_csv(dp)
            safety_df = pd.read_csv(lp)

            self.status.config(text="Status: engineering features + predicting (XGBoost)…")
            self.update_idletasks()

            preds = predict_from_raw(sensor_df, driver_df, safety_df, threshold=float(self.threshold.get()))

            # store into App for single tab
            self.app.set_shared_data(sensor_df, driver_df, safety_df, preds)

            # prompt save
            out_path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv")],
                initialfile="batch_predictions.csv",
                title="Save prediction output CSV",
            )
            if out_path:
                preds.to_csv(out_path, index=False)

            pos = int((preds["pred_label"] == 1).sum())
            total = int(len(preds))
            self.status.config(text=f"Status: done. predicted dangerous: {pos}/{total}. history updated.")

            # refresh history tab
            self.app.refresh_history()

        except Exception as e:
            messagebox.showerror("Batch prediction failed", str(e))
            self.status.config(text="Status: error occurred. check your CSV columns.")

    def _push_to_single(self):
        if self.app.preds is None:
            messagebox.showwarning("No predictions loaded", "Run Batch Prediction first.")
            return
        self.app.push_to_single()
