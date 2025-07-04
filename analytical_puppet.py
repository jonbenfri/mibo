#!/usr/bin/env python3
"""
analytical_puppet.py – single bar with TWO strings per vertex (4 strings)

Version 2.7.3
──────────────────────────────────────────────────────────────
• Stores FuncAnimation in variable `ani`  → animation runs
• cache_frame_data=False removes Matplotlib warning
• Everything else (4 anchor dots, sliders, reset, slew-limit,
  runaway guard) same as v2.7.2
"""

import numpy as np
import sympy as sp
import sympy.physics.mechanics as me
from scipy.integrate import RK45
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button

# ---------------- parameters -------------------------------------------------
L, X0, BAR_Y   = 0.35, 0.20, 0.75
MASS           = 0.05
STRING_DAMP    = 0.02
SPOOL_RATE     = 0.35          # m·s⁻¹ spool speed
ALPHA_B = BETA_B = 15.0
FPS, DT        = 60, 1/60
MIN_LEN, MAX_LEN = 0.10, 1.00
VIEW_X, VIEW_Y   = 1.2, 1.4

# ---------------- symbolic dynamics -----------------------------------------
t = sp.symbols('t')
x, y, th       = me.dynamicsymbols('x y th')
u1, u2, u3     = me.dynamicsymbols('u1 u2 u3')

N = me.ReferenceFrame('N')
A = N.orientnew('A', 'Axis', (th, N.z))

O = me.Point('O')
P = me.Point('P');  P.set_pos(O, x*N.x + y*N.y);  P.set_vel(N, u1*N.x + u2*N.y)
H = P.locatenew('H', L*sp.cos(th)*N.x + L*sp.sin(th)*N.y)
H.v2pt_theory(P, N, A)

arm = me.RigidBody('arm', P, A, MASS, (me.inertia(N, 0, 0, 1e-3), P))

# anchor coordinates (4 dots)
q_init = np.array([0.0, 0.0, -0.4])
Px0 = q_init[0]
Hx0 = Px0 + L*np.cos(q_init[2])
anchors = {
    'P_L': (Px0 - X0/2, BAR_Y), 'P_R': (Px0 + X0/2, BAR_Y),
    'H_L': (Hx0 - X0/2, BAR_Y), 'H_R': (Hx0 + X0/2, BAR_Y)
}
points = {'P': P, 'H': H}

phi_list, L_syms = [], {}
for tag, (ax, ay) in anchors.items():
    joint, _ = tag.split('_')
    pt = points[joint]
    dist = sp.sqrt((pt.pos_from(O).dot(N.x)-ax)**2 +
                   (pt.pos_from(O).dot(N.y)-ay)**2)
    Ls = sp.symbols(f'L_{tag}')
    L_syms[tag] = Ls
    phi_list.append(dist - Ls)
phi = sp.Matrix(phi_list)

q = sp.Matrix([x, y, th])
u = sp.Matrix([u1, u2, u3])
km = me.KanesMethod(N, q_ind=q, u_ind=u,
                    kd_eqs=[u1 - x.diff(t), u2 - y.diff(t), u3 - th.diff(t)])
km.kanes_equations([arm], loads=[])

Mfn  = sp.lambdify(list(q)+list(u), km.mass_matrix, 'numpy')
Ffn  = sp.lambdify(list(q)+list(u), km.forcing,      'numpy')
Jfn  = sp.lambdify(list(q)+list(u)+list(L_syms.values()),
                   phi.jacobian(q), 'numpy')
PHfn = sp.lambdify(list(q)+list(u)+list(L_syms.values()), phi, 'numpy')

# ---------------- simulation helpers ----------------------------------------
def make_initial():
    q0 = q_init.copy()
    u0 = np.zeros(3)
    L0 = {}
    for tag, (ax, ay) in anchors.items():
        joint, _ = tag.split('_')
        px, py = (q0[0], q0[1]) if joint == 'P' else (
                  q0[0] + L*np.cos(q0[2]), q0[1] + L*np.sin(q0[2]))
        L0[tag] = float(np.hypot(px-ax, py-ay))
    return q0, u0, L0.copy(), L0.copy()

def rhs(_, s, Lcmd):
    qv, uv = s[:3], s[3:]
    vec = np.r_[qv, uv]

    # small viscous damping along strings
    for tag, (ax, ay) in anchors.items():
        joint, _ = tag.split('_')
        px, py = (qv[0], qv[1]) if joint == 'P' else (
                  qv[0] + L*np.cos(qv[2]), qv[1] + L*np.sin(qv[2]))
        vx, vy = (uv[0], uv[1]) if joint == 'P' else (
                  uv[0] - L*uv[2]*np.sin(qv[2]),
                  uv[1] + L*uv[2]*np.cos(qv[2]))
        rel = np.array([px-ax, py-ay]); n = np.linalg.norm(rel)
        if n > 1e-9:
            Lcmd[tag] = np.clip(
                Lcmd[tag] - STRING_DAMP * (rel/n).dot([vx, vy]) * DT,
                MIN_LEN, MAX_LEN
            )

    Lvals = [Lcmd[k] for k in L_syms]
    M  = Mfn(*vec).astype(float)
    Fv = Ffn(*vec).astype(float).flatten()
    J  = Jfn(*(vec.tolist() + Lvals)).astype(float)
    ph = PHfn(*(vec.tolist() + Lvals)).astype(float).flatten()
    gamma = -2*ALPHA_B*(J @ uv) - (BETA_B**2)*ph

    sol, *_ = np.linalg.lstsq(
        np.block([[M, J.T],
                  [J, np.zeros((4,4))]]),
        np.hstack([Fv, -gamma]), rcond=None)
    return np.hstack([uv, sol[:3]])

# initial state
q0, u0, L_target, L_cmd = make_initial()
state  = np.hstack([q0, u0])
solver = RK45(lambda t, s: rhs(t, s, L_cmd),
              0.0, state, np.inf, max_step=DT)

# ---------------- GUI & drawing --------------------------------------------
fig = plt.figure(figsize=(6, 6))
gs  = fig.add_gridspec(7, 1,
        height_ratios=[6] + [0.4]*4 + [0.2] + [0.4], hspace=0.3)

ax = fig.add_subplot(gs[0]); ax.set_aspect('equal')
ax.set_xlim(-VIEW_X, VIEW_X); ax.set_ylim(-0.6, VIEW_Y)
ax.scatter(*zip(*anchors.values()), color='k', zorder=3)  # four dots

arm_line, = ax.plot([], [], 'b-', lw=4)
str_lines = [ax.plot([], [], 'r--')[0] for _ in range(4)]

slider_axes = [fig.add_subplot(gs[i]) for i in range(1, 5)]
sliders = {}
for ax_, tag in zip(slider_axes, L_target):
    sliders[tag] = Slider(ax_, tag,
                          MIN_LEN, MAX_LEN,
                          valinit=L_target[tag])
def on_slider(_):
    for tag in L_target:
        L_target[tag] = sliders[tag].val
for s in sliders.values():
    s.on_changed(on_slider)

def reset(event=None):
    global state, solver, L_target, L_cmd
    q0, u0, L_target, L_cmd = make_initial()
    state  = np.hstack([q0, u0])
    solver = RK45(lambda t, s: rhs(t, s, L_cmd),
                  0.0, state, np.inf, max_step=DT)
    for tag, val in L_target.items():
        sliders[tag].set_val(val)

Button(fig.add_subplot(gs[-1]), "Reset",
       color='lightgray', hovercolor='0.85').on_clicked(reset)

def redraw():
    x, y, th = state[:3]
    Pxy = np.array([x, y])
    Hxy = Pxy + L * np.array([np.cos(th), np.sin(th)])
    arm_line.set_data([Pxy[0], Hxy[0]], [Pxy[1], Hxy[1]])

    joints = np.vstack([Pxy, Pxy, Hxy, Hxy])
    for ln, (ax_, ay_), j in zip(str_lines, anchors.values(), joints):
        ln.set_data([ax_, j[0]], [ay_, j[1]])

redraw()           # initial drawing

def step(_):
    global state
    # spool toward target
    for tag in L_cmd:
        L_cmd[tag] += np.clip(L_target[tag] - L_cmd[tag],
                              -SPOOL_RATE*DT,  SPOOL_RATE*DT)
    solver.step()
    state = solver.y
    # runaway guard
    if abs(state[0]) > VIEW_X or state[1] < -0.6 or state[1] > VIEW_Y:
        print("Runaway reset")
        reset(); redraw(); return
    redraw()

# store in variable `ani` so it isn't garbage collected
ani = FuncAnimation(fig, step,
                    interval=1000//FPS,
                    blit=False,
                    cache_frame_data=False)

plt.suptitle("Single bar – four strings  (v2.7.3)")
plt.show()

