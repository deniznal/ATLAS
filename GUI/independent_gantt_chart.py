#!/usr/bin/env python3

import json
import re
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.colors as mcolors

# ---------------- Config ----------------
FIGSIZE = (12, 7)
DEFAULT_OUTPUT_PNG = "gantt_export.png"

# rainbow colors for stages 1..5 (matplotlib colormap)
RAINBOW = plt.cm.get_cmap("rainbow", 5)
STAGE_COLORS = {i + 1: mcolors.to_hex(RAINBOW(i)) for i in range(5)}
DEFAULT_COLOR = "#888888"

# ---------------- Helpers ----------------
def load_json_file(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    expected = {"chamber", "station_id", "station_name", "test_name", "product_id", "start_time", "duration", "stage"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing fields in JSON: {missing}")
    if "sample_number" not in df.columns:
        df["sample_number"] = 0
    # normalize types
    df["product_id"] = df["product_id"].astype(str)
    df["sample_number"] = df["sample_number"].astype(str)
    df["station_id"] = df["station_id"].astype(str)
    df["station_name"] = df["station_name"].astype(str)
    df["chamber"] = df["chamber"].astype(str)
    df["start_time"] = pd.to_numeric(df["start_time"], errors="coerce")
    df["duration"] = pd.to_numeric(df["duration"], errors="coerce")
    df["stage"] = pd.to_numeric(df["stage"], errors="coerce").fillna(0).astype(int)
    df["end_time"] = df["start_time"] + df["duration"]
    return df

def extract_number_if_any(s):
    """Return numeric part if present, otherwise large number for sorting (keeps non-numeric last)."""
    m = re.search(r"\d+", str(s))
    return int(m.group()) if m else float("inf")

# ---------------- GUI Application ----------------
class GanttControlRoomV2:
    def __init__(self, root):
        self.root = root
        self.root.title("Gantt Control Room v2")
        self.root.geometry("1350x780")

        self.data_df = pd.DataFrame()
        # state containers
        self.chamber_vars = {}       # chamber -> BoolVar
        self.station_vars = {}       # (chamber, station_name) -> BoolVar
        self.product_vars = {}       # product_id (str) -> BoolVar
        self.sample_vars = {}        # (product_id, sample_str) -> BoolVar
        self.product_to_samples = {} # product_id -> list of sample strings

        # top controls
        top = ttk.Frame(root)
        top.pack(fill="x", padx=6, pady=6)

        ttk.Button(top, text="Load JSON File", command=self.on_load).pack(side="left", padx=4)
        ttk.Button(top, text="Reload File", command=self.on_reload).pack(side="left", padx=4)
        ttk.Button(top, text="Save Chart as PNG", command=self.on_save_png).pack(side="left", padx=4)

        # Sort/Group dropdown (Option A)
        ttk.Label(top, text="Sort/Group by:").pack(side="left", padx=(12,2))
        self.sort_mode_var = tk.StringVar(value="Product")
        sort_options = ["Product", "Station"]
        self.sort_combo = ttk.Combobox(top, values=sort_options, state="readonly", width=28, textvariable=self.sort_mode_var)
        self.sort_combo.pack(side="left")
        self.sort_combo.bind("<<ComboboxSelected>>", lambda e: None)  # placeholder


        ttk.Button(top, text="Refresh Chart", command=self.refresh_chart).pack(side="right", padx=6)

        # main layout: left filters, right chart
        main = ttk.Frame(root)
        main.pack(fill="both", expand=True, padx=6, pady=6)
        main.columnconfigure(0, weight=1)
        main.columnconfigure(1, weight=3)

        # ----------------- filters (left) -----------------
        filters = ttk.Frame(main)
        filters.grid(row=0, column=0, sticky="nswe", padx=(0,6))

        # Chambers section (scrollable)
        ttk.Label(filters, text="Chambers & Stations", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        ttk.Button(
    filters,
    text="Select / Deselect All Stations",
    command=self.toggle_all_stations
).pack(anchor="w", pady=(2, 6))

        chamber_container = ttk.Frame(filters)
        chamber_container.pack(fill="both", expand=False, pady=(4,8))

        self.chamber_canvas = tk.Canvas(chamber_container, height=300)
        self.chamber_scroll = ttk.Scrollbar(chamber_container, orient="vertical", command=self.chamber_canvas.yview)
        self.chamber_canvas.configure(yscrollcommand=self.chamber_scroll.set)
        self.chamber_scroll.pack(side="right", fill="y")
        self.chamber_canvas.pack(side="left", fill="both", expand=True)

        self.chamber_frame = ttk.Frame(self.chamber_canvas)
        self.chamber_canvas.create_window((0,0), window=self.chamber_frame, anchor="nw")
        self.chamber_frame.bind("<Configure>", lambda e: self.chamber_canvas.configure(scrollregion=self.chamber_canvas.bbox("all")))

        # Products & Samples section (scrollable)
        ttk.Label(filters, text="Products & Samples", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(6,0))
        ttk.Button(
    filters,
    text="Select / Deselect All Products",
    command=self.toggle_all_products
).pack(anchor="w", pady=(2, 6))

        prod_container = ttk.Frame(filters)
        prod_container.pack(fill="both", expand=False, pady=(4,8))

        self.prod_canvas = tk.Canvas(prod_container, height=360)
        self.prod_scroll = ttk.Scrollbar(prod_container, orient="vertical", command=self.prod_canvas.yview)
        self.prod_canvas.configure(yscrollcommand=self.prod_scroll.set)
        self.prod_scroll.pack(side="right", fill="y")
        self.prod_canvas.pack(side="left", fill="both", expand=True)

        self.prod_frame = ttk.Frame(self.prod_canvas)
        self.prod_canvas.create_window((0,0), window=self.prod_frame, anchor="nw")
        self.prod_frame.bind("<Configure>", lambda e: self.prod_canvas.configure(scrollregion=self.prod_canvas.bbox("all")))

        # info label
        self.info_label = ttk.Label(filters, text="No file loaded.")
        self.info_label.pack(anchor="w", pady=(6,0))

        # ----------------- chart (right) -----------------
        chart_frame = ttk.Frame(main)
        chart_frame.grid(row=0, column=1, sticky="nswe")
        self.fig, self.ax = plt.subplots(figsize=FIGSIZE)
        self.canvas = FigureCanvasTkAgg(self.fig, master=chart_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    # ---------- UI builders ----------
    def clear_filters_ui(self):
        for w in self.chamber_frame.winfo_children():
            w.destroy()
        for w in self.prod_frame.winfo_children():
            w.destroy()
        self.chamber_vars.clear()
        self.station_vars.clear()
        self.product_vars.clear()
        self.sample_vars.clear()
        self.product_to_samples.clear()

    def build_filters_ui(self):
        if self.data_df.empty:
            return
        self.clear_filters_ui()

        # Chambers (sorted by numeric part of chamber label if present)
        chambers_sorted = sorted(self.data_df["chamber"].unique(), key=extract_number_if_any)
        for chamber in chambers_sorted:
            ch_df = self.data_df[self.data_df["chamber"] == chamber]
            chamber_block = ttk.Frame(self.chamber_frame)
            chamber_block.pack(fill="x", anchor="w", pady=2)

            top = ttk.Frame(chamber_block)
            top.pack(fill="x", anchor="w")

            cvar = tk.BooleanVar(value=True)
            chk = ttk.Checkbutton(top, text=chamber, variable=cvar)
            chk.pack(side="left", anchor="w")
            self.chamber_vars[chamber] = cvar

            arrow = ttk.Button(top, text="►", width=2)
            arrow.pack(side="right", anchor="e")

            stations_container = ttk.Frame(chamber_block)
            # collapsed initially
            stations_container.pack_forget()

            # get station names sorted numerically by name
            station_names = sorted(ch_df["station_id"].unique(), key=extract_number_if_any)
            station_vars_local = []
            for sname in station_names:
                svar = tk.BooleanVar(value=True)
                cb = ttk.Checkbutton(stations_container, text=sname, variable=svar)
                cb.pack(anchor="w", padx=12)
                self.station_vars[(chamber, sname)] = svar
                station_vars_local.append(svar)

            # chamber checkbox toggles all its stations
            def make_ch_toggle(svars, cvar):
                def toggler(*args):
                    val = cvar.get()
                    for sv in svars:
                        sv.set(val)
                return toggler
            cvar.trace_add("write", make_ch_toggle(station_vars_local, cvar))

            # expand/collapse arrow
            def make_toggle(frame, btn):
                def toggle():
                    if frame.winfo_ismapped():
                        frame.pack_forget()
                        btn.config(text="►")
                    else:
                        frame.pack(fill="x", anchor="w", padx=6)
                        btn.config(text="▼")
                return toggle
            arrow.config(command=make_toggle(stations_container, arrow))

        # Products and their samples
        df = self.data_df
        products_sorted = sorted(df["product_id"].unique(), key=extract_number_if_any)
        for pid in products_sorted:
            p_block = ttk.Frame(self.prod_frame)
            p_block.pack(fill="x", anchor="w", pady=2)

            p_top = ttk.Frame(p_block)
            p_top.pack(fill="x", anchor="w")

            pvar = tk.BooleanVar(value=True)
            pchk = ttk.Checkbutton(p_top, text=f"Product {pid}", variable=pvar)
            pchk.pack(side="left", anchor="w")
            self.product_vars[pid] = pvar

            parrow = ttk.Button(p_top, text="►", width=2)
            parrow.pack(side="right", anchor="e")

            samples_container = ttk.Frame(p_block)
            samples_container.pack_forget()

            samples = sorted(df[df["product_id"] == pid]["sample_number"].unique(), key=extract_number_if_any)
            self.product_to_samples[pid] = samples
            sample_vars_local = []
            for s in samples:
                sval = tk.BooleanVar(value=True)
                cb = ttk.Checkbutton(samples_container, text=f"Sample {s}", variable=sval)
                cb.pack(anchor="w", padx=12)
                self.sample_vars[(pid, str(s))] = sval
                sample_vars_local.append((str(s), sval))

            # product checkbox toggles all samples
            def make_product_toggle(sample_pairs, pvar):
                def toggler(*args):
                    val = pvar.get()
                    for _, sv in sample_pairs:
                        sv.set(val)
                return toggler
            pvar.trace_add("write", make_product_toggle(sample_vars_local, pvar))

            # sample -> update product checkbox state (if any sample off, product off; if all on -> product on)
            def make_sample_observers(sample_pairs, pvar):
                def update_product(*args):
                    # if any sample unchecked -> product unchecked
                    all_on = all(sv.get() for _, sv in sample_pairs)
                    # avoid writing if already equal to reduce recursion
                    if pvar.get() != all_on:
                        pvar.set(all_on)
                return update_product
            sample_obs = make_sample_observers(sample_vars_local, pvar)
            for _, sv in sample_vars_local:
                sv.trace_add("write", sample_obs)

            # expand/collapse product samples
            def make_prod_toggle(frame, btn):
                def toggle():
                    if frame.winfo_ismapped():
                        frame.pack_forget()
                        btn.config(text="►")
                    else:
                        frame.pack(fill="x", anchor="w", padx=6)
                        btn.config(text="▼")
                return toggle
            parrow.config(command=make_prod_toggle(samples_container, parrow))

        # info label
        self.info_label.config(text=f"Loaded {len(self.data_df)} tasks — {len(products_sorted)} products")

    # ---------- File handlers ----------
    def on_load(self):
        path = filedialog.askopenfilename(title="Select schedule JSON file", filetypes=[("JSON files","*.json"),("All files","*.*")])
        if not path:
            return
        try:
            df = load_json_file(path)
        except Exception as e:
            messagebox.showerror("Failed to load", f"Failed to load JSON:\n{e}")
            return
        self.data_df = df
        self.filepath = path
        self.build_filters_ui()
        self.refresh_chart()

    def on_reload(self):
        if not getattr(self, "filepath", None):
            messagebox.showinfo("No file", "No file was previously loaded. Use 'Load JSON File'.")
            return
        try:
            df = load_json_file(self.filepath)
        except Exception as e:
            messagebox.showerror("Failed to reload", f"Failed to reload JSON:\n{e}")
            return
        self.data_df = df
        self.build_filters_ui()
        self.refresh_chart()

    # ---------- Filtering ----------
    def toggle_all_stations(self):
        if not self.station_vars:
            return

        # if any station unchecked → check all, else uncheck all
        target = not all(v.get() for v in self.station_vars.values())
    # set all stations
        for svar in self.station_vars.values():
            svar.set(target)

    # sync chambers
        for chamber, cvar in self.chamber_vars.items():
            station_states = [
                self.station_vars[(ch, s)].get()
                for (ch, s) in self.station_vars
                if ch == chamber
            ]
            if station_states:
                cvar.set(all(station_states))

    def toggle_all_products(self):
        if not self.product_vars:
            return

        # if any product unchecked → check all, else uncheck all
        target = not all(v.get() for v in self.product_vars.values())

        # set all products
        for pvar in self.product_vars.values():
            pvar.set(target)

        # set all samples explicitly
        for svar in self.sample_vars.values():
            svar.set(target)

    def get_selected_products_samples(self):
        """Return list of selected (product, sample_str) tuples."""
        selected = []
        for pid, pvar in self.product_vars.items():
            if not pvar.get():
                continue
            samples = self.product_to_samples.get(pid, [])
            for s in samples:
                sval = self.sample_vars.get((pid, str(s)))
                if sval and sval.get():
                    selected.append((pid, str(s)))
        return selected

    def get_allowed_stations(self):
        allowed = set()
        for (ch, sname), svar in self.station_vars.items():
            if svar.get():
                allowed.add((ch, sname))
        return allowed

    def get_filtered_df(self):
        if self.data_df.empty:
            return self.data_df.copy()
        df = self.data_df.copy()
        # products & samples selection: if a product is checked but a sample not, sample_vars handled earlier
        sel_pairs = set(self.get_selected_products_samples())
        if not sel_pairs:
            return df.iloc[0:0]  # empty
        df = df[df.apply(lambda r: (str(r["product_id"]), str(r["sample_number"])) in sel_pairs, axis=1)]
        # stations
        allowed_stations = self.get_allowed_stations()
        if not allowed_stations:
            return df.iloc[0:0]
        df = df[df.apply(lambda r: (r["chamber"], r["station_id"]) in allowed_stations, axis=1)]
        return df

    # ---------- Chart rendering ----------
    def refresh_chart(self):
        if self.data_df.empty:
            messagebox.showinfo("No data", "Load a JSON file first.")
            return

        df = self.get_filtered_df()
        if df.empty:
            messagebox.showwarning("No tasks", "No tasks match current filters.")
            self.ax.clear()
            self.canvas.draw()
            return

        # decide mode
        sort_mode = self.sort_mode_var.get()  # either "Product" or "Station"

        # helper for sorting columns that may be strings containing numbers
        # Prepare df sorted for plotting: chamber numeric, station numeric, start_time
        df = df.copy()
        df = df.sort_values(
            by=["chamber", "station_id", "start_time"],
            key=lambda col: col.map(lambda v: extract_number_if_any(v) if col.name in ("chamber", "station_id") else v)
        )

        self.ax.clear()

        if sort_mode == "Product":
            # one row per product+sample
            df["sample_row_key"] = df["product_id"].astype(str) + "___" + df["sample_number"].astype(str)
            unique_rows = sorted(df["sample_row_key"].unique(), key=lambda k: (extract_number_if_any(k.split("___")[0]), extract_number_if_any(k.split("___")[1])))
            y_positions = {k: i for i, k in enumerate(unique_rows)}
            for _, row in df.iterrows():
                key = f"{row['product_id']}___{row['sample_number']}"
                y = y_positions[key]
                stage_val = int(row.get("stage", 0)) if str(row.get("stage", "")).isdigit() else 0
                color = STAGE_COLORS.get(stage_val, DEFAULT_COLOR)
                self.ax.barh(y, row["duration"], left=row["start_time"], height=0.6, color=color, edgecolor="black")
                label = f"{row['test_name']} @ {row['chamber']} S{row['station_id']}"
                self.ax.text(row["start_time"] + row["duration"]/2, y, label, ha="center", va="center", fontsize=7, color="black")
            ylabels = [f"P{k.split('___')[0]} - S{k.split('___')[1]}" for k in unique_rows]
            self.ax.set_yticks(range(len(unique_rows)))
            self.ax.set_yticklabels(ylabels)
            self.ax.set_ylabel("Product - Sample")
        else:
            # Station mode: rows are station rows (chamber + station_id)
            # Build station_keys sorted by chamber numeric then station numeric
            station_keys_df = (
                df[["chamber", "station_id"]]
                .drop_duplicates()
                .sort_values(["chamber", "station_id"], key=lambda s: s.map(extract_number_if_any))
            )
            station_list = [f"{r['chamber']} | S{r['station_id']}" for _, r in station_keys_df.iterrows()]
            y_positions = {}
            for idx, (_, r) in enumerate(station_keys_df.iterrows()):
                y_positions[(r["chamber"], r["station_id"])] = idx

            for _, row in df.iterrows():
                key = (row["chamber"], row["station_id"])
                y = y_positions[key]
                stage_val = int(row.get("stage", 0)) if str(row.get("stage", "")).isdigit() else 0
                color = STAGE_COLORS.get(stage_val, DEFAULT_COLOR)
                self.ax.barh(y, row["duration"], left=row["start_time"], height=0.6, color=color, edgecolor="black")
                label = f"{row['test_name']} @ P{row['product_id']} S{row['sample_number']}"
                self.ax.text(row["start_time"] + row["duration"]/2, y, label, ha="center", va="center", fontsize=7, color="black")

            self.ax.set_yticks(range(len(station_list)))
            self.ax.set_yticklabels(station_list)
            self.ax.set_ylabel("Station")

        self.ax.set_xlabel("Time")
        self.ax.set_title("Gantt Chart - " + sort_mode)
        self.ax.grid(axis="x", linestyle="--", alpha=0.4)

        # legend for stages present
        stages_present = sorted(set(int(s) for s in df["stage"].dropna().unique() if str(s).isdigit()))
        if stages_present:
            handles = [plt.Line2D([0],[0], color=STAGE_COLORS.get(s, DEFAULT_COLOR), lw=8, label=f"Stage {s}") for s in stages_present]
            self.ax.legend(handles=handles, title="Stage", loc="upper right")

        plt.tight_layout()
        self.canvas.draw()

    # ---------- Save chart ----------
    def on_save_png(self):
        if self.data_df.empty:
            messagebox.showinfo("No data", "Load and render a chart first.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files","*.png"),("All files","*.*")], initialfile=DEFAULT_OUTPUT_PNG)
        if not path:
            return
        try:
            self.fig.savefig(path, dpi=150)
            messagebox.showinfo("Saved", f"Chart saved to {path}")
        except Exception as e:
            messagebox.showerror("Save failed", f"Failed to save PNG:\n{e}")

# ---------- Run ----------
def main():
    root = tk.Tk()
    app = GanttControlRoomV2(root)
    root.mainloop()

if __name__ == "__main__":
    main()
