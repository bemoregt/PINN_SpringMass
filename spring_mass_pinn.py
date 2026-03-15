"""
Spring-Mass PINN Simulator
Simulates motion of mass (m) hanging from ceiling by spring (k) using PINN.
Equation of motion: m*x'' + k*x = 0  (displacement from equilibrium)
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.patches as patches
import tkinter as tk
from tkinter import ttk
import threading


# ─── PINN Model ──────────────────────────────────────────────────────────────

class PINN(nn.Module):
    """Neural network approximating displacement x(t) given time t"""
    def __init__(self, hidden=64, layers=5):
        super().__init__()
        net = [nn.Linear(1, hidden), nn.Tanh()]
        for _ in range(layers - 1):
            net += [nn.Linear(hidden, hidden), nn.Tanh()]
        net += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*net)

    def forward(self, t):
        return self.net(t)


# ─── Training ────────────────────────────────────────────────────────────────

def train_pinn(m, k, x0, v0, t_max, epochs, progress_cb):
    """Train PINN and return (model, loss_history)"""
    model = PINN()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Collocation points for physics residual
    t_phys = torch.linspace(0, t_max, 300).reshape(-1, 1).requires_grad_(True)
    # Initial condition point
    t_ic = torch.zeros(1, 1, requires_grad=True)

    loss_history = []

    for epoch in range(epochs):
        optimizer.zero_grad()

        # Physics loss: x'' + (k/m)*x = 0
        x = model(t_phys)
        dx  = torch.autograd.grad(x,  t_phys, torch.ones_like(x),  create_graph=True)[0]
        d2x = torch.autograd.grad(dx, t_phys, torch.ones_like(dx), create_graph=True)[0]
        loss_phys = torch.mean((d2x + (k / m) * x) ** 2)

        # Initial condition loss: x(0)=x0, x'(0)=v0
        x_ic  = model(t_ic)
        dx_ic = torch.autograd.grad(x_ic, t_ic, torch.ones_like(x_ic), create_graph=True)[0]
        loss_ic = (x_ic - x0) ** 2 + (dx_ic - v0) ** 2

        loss = loss_phys + 100.0 * loss_ic
        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_val = loss.item()
        loss_history.append(loss_val)

        if epoch % 50 == 0:
            progress_cb(epoch, epochs, loss_val)

    progress_cb(epochs, epochs, loss_history[-1])
    return model, loss_history


# ─── GUI App ─────────────────────────────────────────────────────────────────

class SpringMassApp:
    EQUIL_Y = -1.8   # equilibrium y-coordinate in animation (screen units)
    MASS_H  = 0.7    # mass block height

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Spring-Mass PINN Simulator")
        self.root.resizable(True, True)

        self.model       = None
        self.anim        = None
        self.is_animating = False
        self.t_data      = None
        self.x_pinn      = None
        self.x_analytical = None
        self.loss_history = None
        self.cur_params  = {}

        self._build_ui()
        self._draw_idle_spring()

    # ── UI Layout ────────────────────────────────────────────────────────────

    def _build_ui(self):
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Left control panel
        ctrl = ttk.Frame(self.root, padding=10)
        ctrl.grid(row=0, column=0, sticky="ns")

        ttk.Label(ctrl, text="⚙ Parameters", font=("", 11, "bold")).pack(anchor="w", pady=(0, 6))

        self.params = {}
        sliders = [
            ("m",     "Mass  m (kg)",          1.0,  0.1, 10.0),
            ("k",     "Spring k (N/m)",         4.0,  0.1, 40.0),
            ("x0",    "Initial disp. x₀ (m)",   1.0, -4.0,  4.0),
            ("v0",    "Initial vel. v₀ (m/s)",  0.0, -8.0,  8.0),
            ("t_max", "Duration T (s)",         10.0,  1.0, 40.0),
        ]
        for name, label, default, lo, hi in sliders:
            self._add_slider(ctrl, name, label, default, lo, hi)

        ttk.Separator(ctrl, orient="horizontal").pack(fill="x", pady=8)

        # Epochs
        ef = ttk.Frame(ctrl)
        ef.pack(fill="x")
        ttk.Label(ef, text="Epochs", width=14).pack(side="left")
        self.epochs_var = tk.IntVar(value=3000)
        ttk.Spinbox(ef, from_=500, to=20000, increment=500,
                    textvariable=self.epochs_var, width=8).pack(side="left")

        ttk.Separator(ctrl, orient="horizontal").pack(fill="x", pady=8)

        # Buttons
        self.train_btn = ttk.Button(ctrl, text="▶  Train PINN", command=self._on_train)
        self.train_btn.pack(fill="x", pady=3)
        self.anim_btn = ttk.Button(ctrl, text="▷  Start Animation",
                                   command=self._toggle_anim, state="disabled")
        self.anim_btn.pack(fill="x", pady=3)
        ttk.Button(ctrl, text="↺  Reset", command=self._reset).pack(fill="x", pady=3)

        ttk.Separator(ctrl, orient="horizontal").pack(fill="x", pady=8)

        # Status panel
        sf = ttk.LabelFrame(ctrl, text="Status", padding=6)
        sf.pack(fill="x")
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(sf, textvariable=self.status_var, wraplength=210,
                  font=("", 9)).pack(anchor="w")
        self.prog_var = tk.DoubleVar(value=0)
        self.prog_bar = ttk.Progressbar(sf, variable=self.prog_var, maximum=100)
        self.prog_bar.pack(fill="x", pady=4)
        self.loss_var = tk.StringVar(value="Loss: —")
        ttk.Label(sf, textvariable=self.loss_var, font=("Courier", 9)).pack(anchor="w")

        # Right matplotlib canvas
        fig_frame = ttk.Frame(self.root)
        fig_frame.grid(row=0, column=1, sticky="nsew", padx=(0, 6), pady=6)

        self.fig = plt.Figure(figsize=(11, 8), dpi=100, facecolor="#f8f8f8")
        gs = self.fig.add_gridspec(2, 2, hspace=0.38, wspace=0.35,
                                   left=0.08, right=0.97, top=0.95, bottom=0.07)

        self.ax_spr   = self.fig.add_subplot(gs[0, 0])   # spring-mass animation
        self.ax_pos   = self.fig.add_subplot(gs[0, 1])   # position vs time
        self.ax_phase = self.fig.add_subplot(gs[1, 0])   # phase space
        self.ax_loss  = self.fig.add_subplot(gs[1, 1])   # training loss

        for ax in [self.ax_spr, self.ax_pos, self.ax_phase, self.ax_loss]:
            ax.set_facecolor("#fdfdfd")

        self._setup_static_axes()

        self.canvas = FigureCanvasTkAgg(self.fig, fig_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def _add_slider(self, parent, name, label, default, lo, hi):
        f = ttk.Frame(parent)
        f.pack(fill="x", pady=2)
        ttk.Label(f, text=label, width=20, anchor="w").pack(side="left")
        var = tk.DoubleVar(value=default)
        self.params[name] = var
        val_lbl = ttk.Label(f, width=6, anchor="e")
        val_lbl.pack(side="right")

        def _update(v, _var=var, _lbl=val_lbl):
            _lbl.config(text=f"{_var.get():.2f}")

        slider = ttk.Scale(f, from_=lo, to=hi, variable=var,
                           orient="horizontal", length=140, command=_update)
        slider.pack(side="left", padx=4)
        _update(None)

    def _setup_static_axes(self):
        for ax, title, xlabel, ylabel in [
            (self.ax_pos,   "Position vs Time",  "Time t (s)", "Displacement x (m)"),
            (self.ax_phase, "Phase Space",        "x (m)",      "v (m/s)"),
            (self.ax_loss,  "Training Loss",      "Epoch",      "Loss"),
        ]:
            ax.set_title(title, fontsize=10)
            ax.set_xlabel(xlabel, fontsize=9)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.grid(True, alpha=0.3)
        self.ax_loss.set_yscale("log")

    # ── 스프링 그리기 유틸 ───────────────────────────────────────────────────

    @staticmethod
    def _spring_path(y_top, y_bot, n_coils=10, amp=0.28):
        t = np.linspace(0, 1, n_coils * 30)
        y = y_top + t * (y_bot - y_top)
        m = 0.07
        x = np.zeros_like(t)
        mask = (t > m) & (t < 1 - m)
        x[mask] = amp * np.sin(2 * np.pi * n_coils *
                               (t[mask] - m) / (1 - 2 * m))
        return x, y

    def _draw_ceiling(self, ax):
        ax.fill_between([-1.5, 1.5], [0, 0], [0.25, 0.25],
                        color="#888", alpha=0.35, hatch="////")
        ax.plot([-1.5, 1.5], [0, 0], color="#555", linewidth=3)

    def _draw_idle_spring(self):
        ax = self.ax_spr
        ax.clear()
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-4.5, 0.4)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title("Spring-Mass System", fontsize=10)
        self._draw_ceiling(ax)
        sx, sy = self._spring_path(0, self.EQUIL_Y)
        ax.plot(sx, sy, color="royalblue", linewidth=2)
        rect = patches.FancyBboxPatch((-0.38, self.EQUIL_Y - self.MASS_H), 0.76, self.MASS_H,
                                      boxstyle="round,pad=0.04",
                                      facecolor="steelblue", edgecolor="#1a3a5c", linewidth=2)
        ax.add_patch(rect)
        ax.text(0, self.EQUIL_Y - self.MASS_H / 2, "m",
                ha="center", va="center", fontsize=13, color="white", fontweight="bold")
        self.canvas.draw_idle()

    # ── Training ─────────────────────────────────────────────────────────────

    def _on_train(self):
        self._stop_anim()
        self.train_btn.config(state="disabled")
        self.anim_btn.config(state="disabled")
        threading.Thread(target=self._train_worker, daemon=True).start()

    def _train_worker(self):
        p = {n: v.get() for n, v in self.params.items()}
        epochs = self.epochs_var.get()

        def progress(epoch, total, loss):
            pct = 100 * epoch / total
            self.root.after(0, lambda e=epoch, t=total, l=loss, p2=pct:
                            self._update_status(e, t, l, p2))

        self.root.after(0, lambda: self.status_var.set("Training PINN..."))

        model, lh = train_pinn(
            m=p["m"], k=p["k"], x0=p["x0"], v0=p["v0"],
            t_max=p["t_max"], epochs=epochs, progress_cb=progress
        )

        # Generate predictions
        t_data = np.linspace(0, p["t_max"], 600)
        with torch.no_grad():
            x_pinn = model(torch.FloatTensor(t_data).reshape(-1, 1)).numpy().flatten()

        omega = np.sqrt(p["k"] / p["m"])
        x_anal = p["x0"] * np.cos(omega * t_data) + (p["v0"] / omega) * np.sin(omega * t_data)

        self.model        = model
        self.t_data       = t_data
        self.x_pinn       = x_pinn
        self.x_analytical = x_anal
        self.loss_history = lh
        self.cur_params   = p

        self.root.after(0, self._training_done)

    def _update_status(self, epoch, total, loss, pct):
        self.prog_var.set(pct)
        self.loss_var.set(f"Loss: {loss:.3e}")
        self.status_var.set(f"Epoch {epoch:>5d} / {total}")

    def _training_done(self):
        self.prog_var.set(100)
        self.status_var.set("Training complete!")
        self.train_btn.config(state="normal")
        self.anim_btn.config(state="normal")
        self._update_result_plots()

    # ── Result Plots ─────────────────────────────────────────────────────────

    def _update_result_plots(self):
        p = self.cur_params
        omega = np.sqrt(p["k"] / p["m"])
        t, xa, xp = self.t_data, self.x_analytical, self.x_pinn

        # Position vs time
        ax = self.ax_pos
        ax.clear()
        ax.plot(t, xa, "--", color="forestgreen", linewidth=2,
                label="Analytical", alpha=0.85)
        ax.plot(t, xp, "-",  color="royalblue",  linewidth=2,
                label="PINN", alpha=0.85)
        ax.set_title("Position vs Time", fontsize=10)
        ax.set_xlabel("Time t (s)", fontsize=9)
        ax.set_ylabel("Displacement x (m)", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Phase space
        va = -p["x0"] * omega * np.sin(omega * t) + p["v0"] * np.cos(omega * t)
        vp = np.gradient(xp, t)
        ax = self.ax_phase
        ax.clear()
        ax.plot(xa, va, "--", color="forestgreen", linewidth=2, label="Analytical", alpha=0.85)
        ax.plot(xp, vp, "-",  color="royalblue",  linewidth=2, label="PINN",       alpha=0.85)
        ax.set_title("Phase Space", fontsize=10)
        ax.set_xlabel("x (m)", fontsize=9)
        ax.set_ylabel("v (m/s)", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Loss
        ax = self.ax_loss
        ax.clear()
        ax.semilogy(self.loss_history, color="tomato", linewidth=1.2)
        ax.set_title("Training Loss", fontsize=10)
        ax.set_xlabel("Epoch", fontsize=9)
        ax.set_ylabel("Loss", fontsize=9)
        ax.grid(True, alpha=0.3)

        self._draw_idle_spring()
        self.canvas.draw_idle()

    # ── Animation ────────────────────────────────────────────────────────────

    def _toggle_anim(self):
        if self.is_animating:
            self._stop_anim()
        else:
            self._start_anim()

    def _stop_anim(self):
        if self.anim:
            self.anim.event_source.stop()
        self.is_animating = False
        self.anim_btn.config(text="▷  Start Animation")

    def _start_anim(self):
        if self.x_pinn is None:
            return

        # ---- Initialize spring axis ----
        ax = self.ax_spr
        ax.clear()
        x_amp = float(np.max(np.abs(self.x_pinn))) + 0.5
        y_lo  = self.EQUIL_Y - x_amp - self.MASS_H - 0.3
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(y_lo, 0.4)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title("Spring-Mass System", fontsize=10)
        self._draw_ceiling(ax)

        # Spring line
        spring_line, = ax.plot([], [], color="royalblue", linewidth=2.2)

        # Mass block
        rect = patches.FancyBboxPatch(
            (-0.38, self.EQUIL_Y - self.MASS_H), 0.76, self.MASS_H,
            boxstyle="round,pad=0.04",
            facecolor="steelblue", edgecolor="#1a3a5c", linewidth=2
        )
        ax.add_patch(rect)
        m_text = ax.text(0, self.EQUIL_Y - self.MASS_H / 2, "m",
                         ha="center", va="center",
                         fontsize=13, color="white", fontweight="bold")

        # Displacement info text
        info_text = ax.text(-1.4, y_lo + 0.2, "", fontsize=8, color="gray")

        # Time marker on position plot
        time_line = self.ax_pos.axvline(x=0, color="red", linewidth=1.2, alpha=0.7)
        dot_pinn, = self.ax_pos.plot([], [], "o", color="royalblue",   markersize=7)
        dot_anal, = self.ax_pos.plot([], [], "o", color="forestgreen", markersize=7)

        xp = self.x_pinn
        xa = self.x_analytical
        t  = self.t_data

        def _frame(i):
            y_mass = self.EQUIL_Y + xp[i]          # equilibrium + displacement
            y_bot  = y_mass                          # mass top = spring bottom

            sx, sy = self._spring_path(0, y_bot)
            spring_line.set_data(sx, sy)

            rect.set_y(y_bot - self.MASS_H)
            m_text.set_y(y_bot - self.MASS_H / 2)

            info_text.set_text(f"x={xp[i]:+.3f} m\nt={t[i]:.2f} s")

            time_line.set_xdata([t[i], t[i]])
            dot_pinn.set_data([t[i]], [xp[i]])
            dot_anal.set_data([t[i]], [xa[i]])

            return spring_line, rect, m_text, info_text, time_line, dot_pinn, dot_anal

        self.anim = animation.FuncAnimation(
            self.fig, _frame, frames=len(t),
            interval=16, blit=False, repeat=True
        )
        self.is_animating = True
        self.anim_btn.config(text="⏹  Stop Animation")
        self.canvas.draw_idle()

    # ── Reset ────────────────────────────────────────────────────────────────

    def _reset(self):
        self._stop_anim()
        self.model = self.t_data = self.x_pinn = self.x_analytical = None
        self.loss_history = None
        self.train_btn.config(state="normal")
        self.anim_btn.config(state="disabled")
        self.status_var.set("Ready")
        self.prog_var.set(0)
        self.loss_var.set("Loss: —")

        for ax in [self.ax_pos, self.ax_phase, self.ax_loss]:
            ax.clear()
        self._setup_static_axes()
        self._draw_idle_spring()


# ─── Entry Point ─────────────────────────────────────────────────────────────

def main():
    root = tk.Tk()
    root.geometry("1280x780")
    root.minsize(900, 600)
    SpringMassApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
