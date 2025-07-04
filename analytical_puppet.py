#!/usr/bin/env python3
"""
analytical_puppet.py – planar arm with two strings per joint
Version 2.5.2  (grid-spec syntax fix + tidy layout)
"""

import numpy as np, sympy as sp, sympy.physics.mechanics as me
from scipy.integrate import RK45
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button

# ───────────────── parameters ─────────────────
L, X0, BAR_Y   = 0.35, 0.20, 0.75
MASS, G_DAMP   = 0.05, 2.0
STRING_DAMP    = 0.03
ALPHA_B = BETA_B = 15.0
FPS, DT        = 60, 1/60
MIN_LEN, MAX_LEN = 0.10, 1.00
VIEW_X, VIEW_Y = 0.7, 1.1

# ────── 1. symbolic model (unchanged) ──────
t = sp.symbols('t')
x, y, th = me.dynamicsymbols('x y th')
u1, u2, u3 = me.dynamicsymbols('u1 u2 u3')

N = me.ReferenceFrame('N')
A = N.orientnew('A', 'Axis', (th, N.z))

O = me.Point('O')
P = me.Point('P'); P.set_pos(O, x*N.x + y*N.y); P.set_vel(N, u1*N.x + u2*N.y)
H = P.locatenew('H', L*sp.cos(th)*N.x + L*sp.sin(th)*N.y)
H.v2pt_theory(P, N, A)

arm = me.RigidBody('arm', P, A, MASS, (me.inertia(N, 0, 0, 1e-3), P))

anchors = {'P_L': (-X0/2, BAR_Y), 'P_R': (X0/2, BAR_Y),
           'H_L': (-X0/2, BAR_Y), 'H_R': (X0/2, BAR_Y)}
points  = {'P': P, 'H': H}

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
Ffn  = sp.lambdify(list(q)+list(u), km.forcing, 'numpy')
Jfn  = sp.lambdify(list(q)+list(u)+list(L_syms.values()),
                   phi.jacobian(q), 'numpy')
PHfn = sp.lambdify(list(q)+list(u)+list(L_syms.values()), phi, 'numpy')

# ────── 2. helpers: initial state, RHS, reset ──────
def make_initial():
    q0 = np.array([0.0, 0.0, -0.4])
    u0 = np.zeros(3)
    L0 = {}
    for tag, (ax, ay) in anchors.items():
        joint, _ = tag.split('_')
        px, py = (q0[0], q0[1]) if joint == 'P' else \
                 (q0[0] + L*np.cos(q0[2]), q0[1] + L*np.sin(q0[2]))
        L0[tag] = float(np.hypot(px-ax, py-ay))
    return q0, u0, L0

def rhs(_, s, Ldict):
    qv, uv = s[:3], s[3:]
    vec = np.r_[qv, uv]

    # viscous damping along strings
    for tag, (ax, ay) in anchors.items():
        joint, _ = tag.split('_')
        px, py = (qv[0], qv[1]) if joint == 'P' else \
                 (qv[0] + L*np.cos(qv[2]), qv[1] + L*np.sin(qv[2]))
        vx, vy = (uv[0], uv[1]) if joint == 'P' else \
                 (uv[0] - L*uv[2]*np.sin(qv[2]),
                  uv[1] + L*uv[2]*np.cos(qv[2]))
        rel = np.array([px-ax, py-ay])
        n = np.linalg.norm(rel)
        if n > 1e-9:
            Ldict[tag] = np.clip(
                Ldict[tag] - STRING_DAMP * (rel/n).dot([vx, vy]),
                MIN_LEN, MAX_LEN
            )

    Lvals = [Ldict[k] for k in L_syms]
    M  = Mfn(*vec).astype(float)
    Fv = Ffn(*vec).astype(float).flatten()
    J  = Jfn(*(vec.tolist() + Lvals)).astype(float)
    ph = PHfn(*(vec.tolist() + Lvals)).astype(float).flatten()
    gamma = -2*ALPHA_B*(J @ uv) - (BETA_B**2)*ph
    sol, *_ = np.linalg.lstsq(
        np.block([[M, J.T],
                  [J, np.zeros((4, 4))]]),
        np.hstack([Fv, -gamma]), rcond=None)
    return np.hstack([uv, sol[:3]])

def reset(_=None):
    global solver, state, L_cmd
    q0, u0, L_cmd = make_initial()
    state = np.hstack([q0, u0])
    solver = RK45(lambda t, s: rhs(t, s, L_cmd),
                  0.0, state, np.inf, max_step=DT)
    for tag, val in L_cmd.items():
        sliders[tag].set_val(val)

# ────── 3. create initial solver BEFORE GUI ──────
q0, u0, L_cmd = make_initial()
state = np.hstack([q0, u0])
solver = RK45(lambda t, s: rhs(t, s, L_cmd),
              0.0, state, np.inf, max_step=DT)

# ────── 4. Matplotlib GUI ──────
fig = plt.figure(figsize=(6, 6))
# 1 animation row, 4 slider rows, 1 blank spacer, 1 reset row
gs = fig.add_gridspec(7, 1,
                      height_ratios=[6] + [0.4]*4 + [0.2] + [0.4],
                      hspace=0.3)

ax_anim = fig.add_subplot(gs[0])
ax_anim.set_aspect('equal')
ax_anim.set_xlim(-VIEW_X, VIEW_X)
ax_anim.set_ylim(-0.4, VIEW_Y)
for ax_, ay_ in anchors.values():
    ax_anim.plot(ax_, ay_, 'ko')

arm_ln, = ax_anim.plot([], [], 'b-', lw=4)
str_lns = [ax_anim.plot([], [], 'r--')[0] for _ in range(4)]
Aarr = np.array(list(anchors.values()))

slider_axes = [fig.add_subplot(gs[i]) for i in range(1, 5)]
slider_tags = list(L_cmd.keys())
sliders = {}
for ax_, tag in zip(slider_axes, slider_tags):
    sliders[tag] = Slider(ax_, f"len {tag}",
                          MIN_LEN, MAX_LEN, valinit=L_cmd[tag])

def on_slider(_):
    for tag in slider_tags:
        L_cmd[tag] = sliders[tag].val
for s in sliders.values():
    s.on_changed(on_slider)

# reset button
ax_btn = fig.add_subplot(gs[-1]); ax_btn.axis('off')
Button(ax_btn, "Reset",
       color='lightgray', hovercolor='0.85').on_clicked(reset)

def draw(_):
    global state
    solver.step()
    state = solver.y
    # runaway guard
    if abs(state[0]) > VIEW_X or state[1] < -0.4 or state[1] > VIEW_Y:
        reset(); return []
    x, y, th = state[:3]
    Pxy = np.array([x, y])
    Hxy = Pxy + L * np.array([np.cos(th), np.sin(th)])
    arm_ln.set_data([Pxy[0], Hxy[0]], [Pxy[1], Hxy[1]])
    joints = np.vstack([Pxy, Pxy, Hxy, Hxy])
    for ln, a, j in zip(str_lns, Aarr, joints):
        ln.set_data([a[0], j[0]], [a[1], j[1]])
    return [arm_ln, *str_lns]

ani = FuncAnimation(fig, draw,
                    interval=1000 // FPS,
                    blit=True,
                    cache_frame_data=False)

plt.suptitle("Two strings per joint – v2.5.2")
plt.show()

