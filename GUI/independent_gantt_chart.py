import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def parse_schedule(raw_text):
    """Parse the schedule text into a pandas DataFrame."""
    tasks = []
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    record = {}
    for line in lines:
        if line.startswith("Chamber - Station:"):
            if record:
                tasks.append(record)
                record = {}
            record["station"] = line.replace("Chamber - Station:", "").strip()
        elif line.startswith("Product:"):
            record["product"] = line.split(":", 1)[1].strip()
        elif line.startswith("Task:"):
            record["task"] = line.split(":", 1)[1].strip()
        elif line.startswith("Start:"):
            record["start"] = line.split(": ", 1)[1].strip()
        elif line.startswith("Duration:"):
            record["duration"] = line.split(": ", 1)[1].strip()
        elif line.startswith("End:"):
            record["end"] = line.split(": ", 1)[1].strip()
    if record:
        tasks.append(record)

    def to_minutes(t):
        h, m, s = map(int, t.split(":"))
        return h * 60 + m + s / 60

    df = pd.DataFrame(tasks)
    if df.empty:
        return df

    df["start_min"] = df["start"].apply(to_minutes)
    df["duration_min"] = df["duration"].apply(to_minutes)
    df["end_min"] = df["end"].apply(to_minutes)
    return df

class GanttApp:
    def __init__(self, root):
        self.root = root
        self.df = pd.DataFrame()
        self.station_vars = {}
        self.filtered_stations = set()

        root.title("Gantt Chart Viewer")
        root.geometry("1200x700")

        self.frame_left = ttk.Frame(root)
        self.frame_left.pack(side="left", fill="y", padx=5, pady=5)

        self.frame_right = ttk.Frame(root)
        self.frame_right.pack(side="right", fill="both", expand=True)

        ttk.Label(self.frame_left, text="Stations", font=("Arial", 12, "bold")).pack(pady=5)

        ttk.Button(self.frame_left, text="Load Schedule File", command=self.load_file).pack(pady=5)
        ttk.Button(self.frame_left, text="Export visible tasks (CSV)", command=self.export_csv).pack(pady=5)
        ttk.Button(self.frame_left, text="Show parsed table", command=self.show_table).pack(pady=5)

        self.station_container = ttk.Frame(self.frame_left)
        self.station_container.pack(fill="y", expand=True)

        btn_frame = ttk.Frame(self.frame_left)
        btn_frame.pack(pady=10)
        ttk.Button(btn_frame, text="Show All", command=self.show_all).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Hide All", command=self.hide_all).pack(side="left", padx=5)

        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame_right)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def load_file(self):
        path = filedialog.askopenfilename(
            title="Select schedule text file",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if not path:
            return
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()

        df = parse_schedule(raw)
        if df.empty:
            messagebox.showerror("Error", "No valid schedule data found.")
            return

        self.df = df
        self.filtered_stations = set(df["station"].unique())
        self.build_station_checkboxes()
        self.update_chart()
        messagebox.showinfo("Loaded", f"Loaded {len(df)} tasks from {path}")

    def build_station_checkboxes(self):
        for widget in self.station_container.winfo_children():
            widget.destroy()
        self.station_vars.clear()

        for st in sorted(self.df["station"].unique()):
            var = tk.BooleanVar(value=True)
            chk = ttk.Checkbutton(
                self.station_container, text=st, variable=var,
                command=self.update_chart
            )
            chk.pack(anchor="w")
            self.station_vars[st] = var

    def update_chart(self):
        if self.df.empty:
            return

        self.filtered_stations = {
            st for st, var in self.station_vars.items() if var.get()
        }
        subset = self.df[self.df["station"].isin(self.filtered_stations)]

        self.ax.clear()
        if subset.empty:
            self.ax.text(0.5, 0.5, "No stations selected", ha="center", va="center")
        else:
            ylabels = sorted(subset["station"].unique())
            y_pos = range(len(ylabels))
            for i, st in enumerate(ylabels):
                st_data = subset[subset["station"] == st]
                for _, row in st_data.iterrows():
                    self.ax.barh(
                        i, row["duration_min"],
                        left=row["start_min"],
                        height=0.4, align="center",
                        label=row["task"] if i == 0 else "",
                    )
                    self.ax.text(
                        row["start_min"] + row["duration_min"] / 2, i,
                        f"{row['task']} (P{row['product']})",
                        va="center", ha="center", fontsize=7
                    )
            self.ax.set_yticks(list(y_pos))
            self.ax.set_yticklabels(ylabels)
            self.ax.set_xlabel("Minutes")
            self.ax.set_title("Gantt Chart - Schedule Overview")
            self.ax.grid(True, axis="x", linestyle="--", alpha=0.5)

        self.fig.tight_layout()
        self.canvas.draw()

    def show_all(self):
        for var in self.station_vars.values():
            var.set(True)
        self.update_chart()

    def hide_all(self):
        for var in self.station_vars.values():
            var.set(False)
        self.update_chart()

    def export_csv(self):
        if self.df.empty:
            messagebox.showerror("Error", "No data to export.")
            return
        subset = self.df[self.df["station"].isin(self.filtered_stations)]
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")]
        )
        if path:
            subset.to_csv(path, index=False)
            messagebox.showinfo("Export", f"Exported {len(subset)} tasks to {path}")

    def show_table(self):
        if self.df.empty:
            messagebox.showerror("Error", "No data loaded.")
            return
        win = tk.Toplevel(self.root)
        win.title("Parsed Table")
        tree = ttk.Treeview(win, columns=list(self.df.columns), show="headings")
        for c in self.df.columns:
            tree.heading(c, text=c)
            tree.column(c, width=100)
        for _, row in self.df.iterrows():
            tree.insert("", "end", values=list(row))
        tree.pack(fill="both", expand=True)

if __name__ == "__main__":
    root = tk.Tk()
    app = GanttApp(root)
    root.mainloop()
