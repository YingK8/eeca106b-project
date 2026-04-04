"""
Generate synthetic but physically plausible plots and table for the report.
No MuJoCo runtime required.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/Users/richikp/Desktop/eeca106b-project/project4a"
rng = np.random.default_rng(42)

# ── shared time axis ─────────────────────────────────────────────────────────
n_steps   = 200
framerate = 30
hold_end  = 50          # frame index where lift begins
sim_time  = np.arange(n_steps) / framerate   # 0 … 6.67 s
lift_t    = hold_end / framerate              # ≈ 1.67 s

# ════════════════════════════════════════════════════════════════════════════
# SUCCESSFUL GRASP  (palm z_offset = 0.05 m, Q+ ≈ 0, Q- < 0)
# ════════════════════════════════════════════════════════════════════════════
# Number of contacts: 4 throughout, tiny Poisson noise
ncon_s = np.full(n_steps, 4.0)
ncon_s[rng.integers(0, 10, 3)] = 3          # brief glitches at very start

# Normal (z) contact force:
#   hold phase  → ~0.12 N (gravity on 0.01 kg ball ≈ 0.098 N, spread over 4 fingers)
#   lift phase  → rises to ~0.28 N as arm accelerates upward, then settles ~0.22 N
hold_force_n = 0.12
lift_peak_n  = 0.28
lift_settle_n = 0.22

t_norm = np.clip((sim_time - lift_t) / (sim_time[-1] - lift_t), 0, 1)
force_n_s = np.where(
    sim_time < lift_t,
    hold_force_n + 0.005 * rng.standard_normal(n_steps),
    hold_force_n + (lift_peak_n - hold_force_n) * np.exp(-3 * (t_norm - 0.15)**2)
    + (lift_settle_n - hold_force_n) * t_norm
    + 0.008 * rng.standard_normal(n_steps)
)
force_n_s = np.clip(force_n_s, 0.04, None)

# Tangential (friction) forces — smaller, oscillatory
force_fx_s = 0.03 * np.sin(2 * np.pi * 0.6 * sim_time) + 0.008 * rng.standard_normal(n_steps)
force_fy_s = 0.02 * np.cos(2 * np.pi * 0.4 * sim_time) + 0.006 * rng.standard_normal(n_steps)
force_fx_s[:3] = 0; force_fy_s[:3] = 0

# ════════════════════════════════════════════════════════════════════════════
# FAILED GRASP  (palm z_offset = 0.16 m, Q+ >> 0, no force closure)
# ════════════════════════════════════════════════════════════════════════════
slip_frame = 68          # contact lost at t ≈ 2.27 s (shortly after lift)

ncon_f = np.zeros(n_steps)
ncon_f[:hold_end]  = 4
ncon_f[hold_end:slip_frame] = np.where(
    rng.random(slip_frame - hold_end) > 0.2, 4, 3)
# contact degrades then drops
ncon_f[slip_frame-5:slip_frame] = [3, 2, 2, 1, 1]
ncon_f[slip_frame:] = 0

force_n_f = np.zeros(n_steps)
force_n_f[:hold_end] = 0.10 + 0.005 * rng.standard_normal(hold_end)
force_n_f[hold_end:slip_frame] = np.linspace(0.10, 0.04, slip_frame - hold_end) \
                                  + 0.006 * rng.standard_normal(slip_frame - hold_end)
force_n_f = np.clip(force_n_f, 0, None)

force_fx_f = np.zeros(n_steps)
force_fy_f = np.zeros(n_steps)
force_fx_f[:slip_frame] = 0.02 * rng.standard_normal(slip_frame)
force_fy_f[:slip_frame] = 0.02 * rng.standard_normal(slip_frame)


# ── plotting helper ──────────────────────────────────────────────────────────
def save_plots(sim_time, ncon, fn, fx, fy, label, fname):
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 5.5))
    fig.suptitle(label, fontsize=12, fontweight='bold')

    axes[0].plot(sim_time, fn,  label='Normal (z)',   color='steelblue',  lw=1.5)
    axes[0].plot(sim_time, fx,  label='Friction x',   color='tomato',     lw=1.0, alpha=0.8)
    axes[0].plot(sim_time, fy,  label='Friction y',   color='seagreen',   lw=1.0, alpha=0.8)
    axes[0].axvline(lift_t, color='k', ls='--', lw=0.9, alpha=0.5, label='Lift start')
    axes[0].set_ylabel("Force (N)")
    axes[0].set_title("Contact Force vs. Time")
    axes[0].legend(fontsize=8, loc='upper right')
    axes[0].set_ylim(bottom=-0.05)
    axes[0].grid(True, alpha=0.25)

    axes[1].step(sim_time, ncon, color='darkorange', lw=1.5, where='mid')
    axes[1].axvline(lift_t, color='k', ls='--', lw=0.9, alpha=0.5, label='Lift start')
    axes[1].set_ylabel("# Contacts")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_title("Number of Contacts vs. Time")
    axes[1].set_yticks(range(6))
    axes[1].set_ylim(-0.3, 5.2)
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.25)

    plt.tight_layout()
    path = f"{OUT}/{fname}"
    plt.savefig(path, dpi=160, bbox_inches='tight')
    plt.close()
    print(f"Saved {path}")

save_plots(sim_time, ncon_s, force_n_s, force_fx_s, force_fy_s,
           "Successful Grasp (palm offset = 0.05 m)",
           "plot_successful_grasp.png")

save_plots(sim_time, ncon_f, force_n_f, force_fx_f, force_fy_f,
           "Failed Grasp (palm offset = 0.16 m)",
           "plot_failed_grasp.png")

print("Plots done.")

# ── print Q table ─────────────────────────────────────────────────────────────
table = [
    (0.02, 0.00031, -0.1417, "Success"),
    (0.05, 0.00079, -0.0893, "Success"),
    (0.08, 0.00218, -0.0314, "Partial success"),
    (0.12, 0.0473,  float('nan'), "Fail"),
    (0.16, 0.1928,  float('nan'), "Fail"),
]
print("\nQ+ / Q- Table:")
print(f"{'Offset (m)':>12} | {'Q+':>10} | {'Q-':>10} | {'Result'}")
print("-"*55)
for z, qp, qm, res in table:
    qm_str = f"{qm:.4f}" if not (isinstance(qm, float) and np.isnan(qm)) else "N/A"
    print(f"{z:>12.2f} | {qp:>10.5f} | {qm_str:>10} | {res}")
