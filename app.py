import tkinter as tk
from tkinter import ttk, messagebox

from .db import init_db, reset_db
from .ui_batch import BatchFrame
from .ui_realtime import RealtimeFrame
from .ui_history import HistoryFrame


class App(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("GoBest Dangerous Trip Detector (XGBoost)")
        self.geometry("1020x720")
        self.minsize(960, 640)

        init_db()

        # shared state
        self.sensor_df = None
        self.driver_df = None
        self.safety_df = None
        self.preds = None

        self._style()
        self._build()

    def _style(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except Exception:
            pass

        style.configure("Title.TLabel", font=("Segoe UI", 16, "bold"))
        style.configure("Sub.TLabel", font=("Segoe UI", 10))
        style.configure("Hint.TLabel", font=("Segoe UI", 9), foreground="#555555")

        style.configure("TButton", padding=6)
        style.configure("TEntry", padding=4)
        style.configure("TCombobox", padding=3)

    def _build(self):
        root = ttk.Frame(self, padding=12)
        root.pack(fill="both", expand=True)

        # top header
        header = ttk.Frame(root)
        header.pack(fill="x")

        ttk.Label(header, text="GoBest: Dangerous Trip Detector", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            header,
            text="Raw CSV → Feature Engineering → XGBoost → SQLite History → GUI",
            style="Hint.TLabel",
        ).pack(anchor="w", pady=(2, 0))

        # menu
        menubar = tk.Menu(self)
        tools = tk.Menu(menubar, tearoff=0)
        tools.add_command(label="Refresh History", command=self.refresh_history)
        tools.add_separator()
        tools.add_command(label="Reset DB (clear history)", command=self._reset_db_prompt)
        menubar.add_cascade(label="Tools", menu=tools)
        self.config(menu=menubar)

        # notebook
        self.tabs = ttk.Notebook(root)
        self.tabs.pack(fill="both", expand=True, pady=(12, 0))

        self.batch_tab = BatchFrame(self.tabs, app=self)
        self.single_tab = RealtimeFrame(self.tabs, app=self)
        self.history_tab = HistoryFrame(self.tabs)

        self.tabs.add(self.batch_tab, text="Batch Prediction")
        self.tabs.add(self.single_tab, text="Single Trip")
        self.tabs.add(self.history_tab, text="History")

        # status bar
        self.status = ttk.Label(root, text="Ready.", style="Hint.TLabel")
        self.status.pack(anchor="w", pady=(10, 0))

    def set_shared_data(self, sensor_df, driver_df, safety_df, preds):
        self.sensor_df = sensor_df
        self.driver_df = driver_df
        self.safety_df = safety_df
        self.preds = preds

        self.status.config(text=f"Loaded predictions for {len(preds)} trips. Single Trip tab is ready.")
        self.push_to_single()

    def push_to_single(self):
        self.single_tab.refresh_booking_list()

    def refresh_history(self):
        try:
            self.history_tab.refresh()
            self.status.config(text="History refreshed.")
        except Exception as e:
            messagebox.showerror("History refresh failed", str(e))

    def _reset_db_prompt(self):
        ok = messagebox.askyesno("Reset history", "This will clear ALL stored history. Continue?")
        if not ok:
            return
        reset_db()
        self.refresh_history()
        self.status.config(text="History cleared.")


def main():
    App().mainloop()


if __name__ == "__main__":
    main()
