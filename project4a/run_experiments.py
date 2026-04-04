"""
Experiment script:
  1. Run grasp synthesis at 5 palm heights, collect Q+/Q- values
  2. Run control loop for a SUCCESSFUL grasp  → save contact-force & ncon plots
  3. Run control loop for a FAILED grasp      → save contact-force & ncon plots
"""

import os
os.environ["MUJOCO_GL"] = "glfw"

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mujoco as mj
from dm_control import mjcf
import importlib, sys

# ── make sure the project modules are importable ────────────────────────────
sys.path.insert(0, "/Users/richikp/Desktop/eeca106b-project/project4a")

import multifingered_ik, grasp_synthesis, AllegroHandEnv
importlib.reload(multifingered_ik)
importlib.reload(grasp_synthesis)
importlib.reload(AllegroHandEnv)

from multifingered_ik import LevenbergMarquardtIK
from grasp_synthesis import synthesize_grasp, optimize_necessary_condition, optimize_sufficient_condition, build_friction_cone, build_grasp_matrix
from AllegroHandEnv import AllegroHandEnvSphere

# ── MJCF paths ───────────────────────────────────────────────────────────────
BASE = "/Users/richikp/Desktop/eeca106b-project/project4a"
hand_path    = f"{BASE}/mujoco_menagerie/wonik_allegro/right_hand.xml"
sawyer_path  = f"{BASE}/mujoco_menagerie/rethink_robotics_sawyer/sawyer.xml"

ball_xml = """
<mujoco model="ball">
    <worldbody>
        <body name="ball_body" pos="1.0 -0.2 0.95">
            <geom name="ball_geom" mass="0.01" friction="1.5" type="sphere" size="0.05" rgba="1 0 0 1"
                  solref="0.06 1" solimp="0.9 0.95 0.003 0.5 2"/>
        </body>
    </worldbody>
</mujoco>
"""

table_xml = """
<mujoco model="table">
    <worldbody>
        <body name="table_body" pos="1.0 -0.2 0.45">
            <geom name="table_geom" mass="200000" friction="0.8" type="box" solref="0.01 0.5" size="0.3 0.6 0.45" rgba="0.798 0.71 0.469 1"/>
        </body>
    </worldbody>
</mujoco>
"""

# ── Build scene ───────────────────────────────────────────────────────────────
def build_scene():
    hand_model   = mjcf.from_path(hand_path)
    sawyer_model = mjcf.from_path(sawyer_path)
    ball_model   = mjcf.from_xml_string(ball_xml)
    table_model  = mjcf.from_xml_string(table_xml)

    # fingertip offsets
    for name, offset in [('ff_tip', 0.028), ('mf_tip', 0.028),
                          ('rf_tip', 0.028), ('th_tip', 0.044)]:
        short = name.replace('_tip', '')
        tip_body = hand_model.find('body', name)
        tip_body.add('body', name=f'{name}_rubber', pos=[0, 0, offset])
        hand_model.find('body', f'{name}_rubber').add(
            'geom', type='sphere', size=[0.012], rgba=[0, 0, 0, 0])

    arena = mjcf.RootElement()
    sawyer_site = sawyer_model.find('site', 'attachment_site')
    attachment_frame = arena.attach(ball_model)
    arena.attach(table_model)
    sawyer_site.attach(hand_model)
    arena.attach(sawyer_model)

    # minimal scene
    arena.worldbody.add('geom', type='plane', size=[10, 10, 10])
    arena.worldbody.add('camera', name='camera_1', pos=[-1, -1, 0.3], euler=[1.55, 2, 0])
    attachment_frame.add('joint', name='ball_joint', type='free', armature='5e-5')

    ball_body = ball_model.find('body', 'ball_body')
    ball_geom = ball_model.find('geom', 'ball_geom')
    return arena, ball_body, ball_geom


def make_physics(arena):
    physics = mjcf.Physics.from_mjcf_model(arena)
    physics.named.model.cam_pos['camera_1'] = [1.9, 0.3, 1.2]
    return physics


INITIAL_QPOS = [0, 0, 0, 0, 0, 0, 0, 0, -0.8, 0, 2, 0, -1.2, 3.2,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

def set_initial_configuration(physics):
    physics.data.qpos[:] = INITIAL_QPOS
    physics.forward()


def evaluate_IK(physics, target_positions, target_orientations, target_names):
    model = physics.model
    data  = physics.data
    n     = len(target_positions)
    jacp  = np.zeros((n, 3, model.nv))
    jacr  = np.zeros((n, 3, model.nv))
    ik    = LevenbergMarquardtIK(model, data, 0.5, 0.002, 0.5,
                                  jacp, jacr, 0.15, 200, physics)
    return ik.calculate(target_positions, target_orientations, target_names)


# ── helpers to evaluate Q+/Q- from a physics state ──────────────────────────
def evaluate_fc_metrics(physics, ball_center, ball_radius, q_h_slice,
                         friction_coeff=0.5, num_approx=8):
    """
    Given current physics state (must have >=4 ball contacts),
    return (Q_plus, Q_minus).  If not in contact return (inf, inf).
    """
    if physics.data.ncon < 4:
        return float('inf'), float('inf')

    from grasp_synthesis import build_friction_cone, build_grasp_matrix
    from grasp_synthesis import optimize_necessary_condition, optimize_sufficient_condition

    object_name = 'ball/ball_geom'
    env = AllegroHandEnvSphere(physics, ball_center, ball_radius,
                                q_h_slice, object_name)

    try:
        contact_frames, contact_positions = env.get_contact_normals_and_positions(
            physics.data.ptr.contact)
    except Exception:
        return float('inf'), float('inf')

    if len(contact_positions) < 4:
        return float('inf'), float('inf')

    directions_list = [build_friction_cone(cf, friction_coeff, num_approx)
                       for cf in contact_frames]
    G = build_grasp_matrix(contact_positions, directions_list, origin=ball_center)

    Q_plus  = optimize_necessary_condition(G, env)
    if Q_plus < 1e-4:
        Q_minus = optimize_sufficient_condition(G)
    else:
        Q_minus = float('nan')
    return Q_plus, Q_minus


# ════════════════════════════════════════════════════════════════════════════
# 1.  PALM-HEIGHT SWEEP
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("PALM HEIGHT SWEEP")
print("="*60)

palm_offsets = [0.02, 0.05, 0.08, 0.12, 0.16]
results = []

for z_offset in palm_offsets:
    print(f"\n--- z_offset = {z_offset:.2f} ---")
    arena, ball_body, ball_geom = build_scene()
    physics = make_physics(arena)
    set_initial_configuration(physics)

    ball_center = np.array(ball_body.pos, dtype=float)
    ball_radius = float(ball_geom.size[0])
    q_h_slice   = slice(14, 30)

    # Position palm above ball
    target_palm_pos = ball_center.copy()
    target_palm_pos[2] += z_offset
    palm_qpos = evaluate_IK(physics,
                             target_palm_pos.reshape(1, -1),
                             np.array([[0.71, 0, 0.71, 0]]),
                             ['sawyer/allegro_right/palm'])
    physics.data.qpos[:] = palm_qpos
    physics.data.qpos[14:] = 0
    physics.forward()

    q_h_init = physics.data.qpos[q_h_slice].copy()
    object_name = 'ball/ball_geom'
    allegro_env = AllegroHandEnvSphere(physics, ball_center, ball_radius,
                                        q_h_slice, object_name)
    fingertip_names = ['sawyer/allegro_right/ff_tip_rubber',
                        'sawyer/allegro_right/mf_tip_rubber',
                        'sawyer/allegro_right/rf_tip_rubber',
                        'sawyer/allegro_right/th_tip_rubber']

    force_closure_q_h = synthesize_grasp(allegro_env, q_h_init, fingertip_names,
                                          max_iters=2000, lr=0.5)

    # Evaluate metrics at the synthesized configuration
    Q_plus, Q_minus = evaluate_fc_metrics(physics, ball_center, ball_radius,
                                           q_h_slice)
    results.append((z_offset, Q_plus, Q_minus))
    print(f"  Q+ = {Q_plus:.6f}  |  Q- = {Q_minus}")

print("\n\nSUMMARY TABLE")
print(f"{'z_offset':>10} | {'Q+':>12} | {'Q-':>12}")
print("-"*40)
for z, qp, qm in results:
    qm_str = f"{qm:.6f}" if not (isinstance(qm, float) and np.isnan(qm)) else "N/A"
    print(f"{z:>10.2f} | {qp:>12.6f} | {qm_str:>12}")

np.save(f"{BASE}/height_sweep_results.npy",
        np.array([(z, qp, qm if not np.isnan(qm) else 999)
                  for z, qp, qm in results], dtype=float))


# ════════════════════════════════════════════════════════════════════════════
# helper: run control loop and return (sim_time, ncon, force)
# ════════════════════════════════════════════════════════════════════════════
def run_control_loop(physics, grasp_q, ball_body, n_steps=200):
    physics.reset()
    physics.data.qpos[:] = grasp_q
    physics.data.qvel[:] = 0
    physics.forward()

    grasp_arm_q  = grasp_q[7:14].copy()
    grasp_hand_q = grasp_q[14:30].copy()

    target_lift = np.array(ball_body.pos, dtype=float).copy()
    target_lift[2] += 0.25
    palm_lift_qpos = evaluate_IK(physics,
                                  target_lift.reshape(1, -1),
                                  np.array([[0.71, 0, 0.71, 0]]),
                                  ['sawyer/allegro_right/palm'])
    lift_arm_q = palm_lift_qpos[7:14].copy()

    physics.data.qpos[:] = grasp_q
    physics.data.qvel[:] = 0
    physics.forward()

    framerate  = 30
    hold_steps = 50
    sim_time   = np.zeros(n_steps)
    ncon_arr   = np.zeros(n_steps)
    force_arr  = np.zeros((n_steps, 3))
    forcetorque = np.zeros(6)

    for i in range(n_steps):
        while physics.data.time * framerate < i:
            physics.data.ctrl[7:23] = grasp_hand_q
            if i < hold_steps:
                arm_target = grasp_arm_q
            else:
                t = float(np.clip((i - hold_steps) /
                                   max(n_steps - hold_steps - 1, 1), 0, 1))
                arm_target = (1 - t) * grasp_arm_q + t * lift_arm_q
            physics.data.ctrl[:7] = arm_target

            sim_time[i]  = physics.data.time
            ncon_arr[i]  = physics.data.ncon
            if physics.data.ncon > 0:
                mj.mj_contactForce(physics.model.ptr, physics.data.ptr,
                                   0, forcetorque)
                force_arr[i] = forcetorque[:3]
            physics.step()

    return sim_time, ncon_arr, force_arr


def save_plots(sim_time, ncon_arr, force_arr, label, outdir):
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(9, 6))
    fig.suptitle(f"Grasp Result: {label}", fontsize=13, fontweight='bold')

    axes[0].plot(sim_time, force_arr[:, 0], label='Normal (z)', color='steelblue')
    axes[0].plot(sim_time, force_arr[:, 1], label='Friction x', color='tomato', alpha=0.7)
    axes[0].plot(sim_time, force_arr[:, 2], label='Friction y', color='seagreen', alpha=0.7)
    axes[0].axvline(sim_time[50] if len(sim_time) > 50 else 0,
                    color='k', linestyle='--', alpha=0.4, label='Lift start')
    axes[0].set_ylabel("Force (N)")
    axes[0].set_title("Contact Force vs. Time")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(sim_time, ncon_arr, color='darkorange', linewidth=1.5)
    axes[1].set_ylabel("# Contacts")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_title("Number of Contacts vs. Time")
    axes[1].set_yticks(range(int(ncon_arr.max()) + 2))
    axes[1].axvline(sim_time[50] if len(sim_time) > 50 else 0,
                    color='k', linestyle='--', alpha=0.4, label='Lift start')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = f"{outdir}/plot_{label.lower().replace(' ', '_')}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


# ════════════════════════════════════════════════════════════════════════════
# 2.  SUCCESSFUL GRASP  (z_offset = 0.05)
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("SUCCESSFUL GRASP  (z_offset = 0.05)")
print("="*60)

arena, ball_body, ball_geom = build_scene()
physics = make_physics(arena)
set_initial_configuration(physics)
ball_center = np.array(ball_body.pos, dtype=float)
ball_radius = float(ball_geom.size[0])
q_h_slice   = slice(14, 30)

target_palm_pos = ball_center.copy(); target_palm_pos[2] += 0.05
palm_qpos = evaluate_IK(physics, target_palm_pos.reshape(1,-1),
                         np.array([[0.71,0,0.71,0]]),
                         ['sawyer/allegro_right/palm'])
physics.data.qpos[:] = palm_qpos
physics.data.qpos[14:] = 0
physics.forward()

q_h_init    = physics.data.qpos[q_h_slice].copy()
object_name = 'ball/ball_geom'
allegro_env = AllegroHandEnvSphere(physics, ball_center, ball_radius, q_h_slice, object_name)
fingertip_names = ['sawyer/allegro_right/ff_tip_rubber',
                    'sawyer/allegro_right/mf_tip_rubber',
                    'sawyer/allegro_right/rf_tip_rubber',
                    'sawyer/allegro_right/th_tip_rubber']

force_closure_q_h = synthesize_grasp(allegro_env, q_h_init, fingertip_names,
                                      max_iters=2000, lr=0.5)
grasp_q_success = physics.data.qpos.copy()

sim_time_s, ncon_s, force_s = run_control_loop(physics, grasp_q_success, ball_body)
save_plots(sim_time_s, ncon_s, force_s, "Successful Grasp", BASE)


# ════════════════════════════════════════════════════════════════════════════
# 3.  FAILED GRASP  (z_offset = 0.16 — too high for force closure)
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("FAILED GRASP  (z_offset = 0.16)")
print("="*60)

arena, ball_body, ball_geom = build_scene()
physics = make_physics(arena)
set_initial_configuration(physics)
ball_center = np.array(ball_body.pos, dtype=float)
ball_radius = float(ball_geom.size[0])

target_palm_pos = ball_center.copy(); target_palm_pos[2] += 0.16
palm_qpos = evaluate_IK(physics, target_palm_pos.reshape(1,-1),
                         np.array([[0.71,0,0.71,0]]),
                         ['sawyer/allegro_right/palm'])
physics.data.qpos[:] = palm_qpos
physics.data.qpos[14:] = 0
physics.forward()

q_h_init    = physics.data.qpos[q_h_slice].copy()
allegro_env = AllegroHandEnvSphere(physics, ball_center, ball_radius, q_h_slice, object_name)

force_closure_q_h_fail = synthesize_grasp(allegro_env, q_h_init, fingertip_names,
                                           max_iters=2000, lr=0.5)
grasp_q_fail = physics.data.qpos.copy()

sim_time_f, ncon_f, force_f = run_control_loop(physics, grasp_q_fail, ball_body)
save_plots(sim_time_f, ncon_f, force_f, "Failed Grasp", BASE)

print("\nDone. All results saved.")
