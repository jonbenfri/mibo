#!/usr/bin/env python3
"""
pose_driven_puppet.py – single bar, four overhead strings
────────────────────────────────────────────────────────────
Sliders control the bar’s pose (Px, Py, θ).  For each pose we
compute the exact lengths to the four fixed anchors.

Px  : midpoint X position       (slider “X”)
Py  : midpoint Y position       (slider “Y”)
θ   : bar rotation (rad)        (slider “θ”)

All geometry updates instantly – no physics, no solver, no resets.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# --------- geometry constants
BAR_LEN = 0.35
ANCH_SP = 0.20
BAR_Y0  = 0.75                    # vertical position of anchors

# anchor layout: two above each vertex
anchors = {
    'P_L': (-ANCH_SP/2, BAR_Y0),
    'P_R': ( ANCH_SP/2, BAR_Y0),
    'H_L': (-ANCH_SP/2, BAR_Y0),
    'H_R': ( ANCH_SP/2, BAR_Y0)
}

# ---------------- matplotlib set-up
fig, ax = plt.subplots(figsize=(6,6))
plt.subplots_adjust(left=0.25, bottom=0.35)   # leave room for 3 sliders
ax.set_aspect('equal')
ax.set_xlim(-1.0, 1.0); ax.set_ylim(-0.3, 1.2)
ax.set_title("Bar pose → string lengths")

# draw static anchor dots
ax.scatter(*zip(*anchors.values()), c='k', zorder=3)

# dynamic artists
bar_line,  = ax.plot([], [], 'b-', lw=4)
str_lines   = [ax.plot([], [], 'r--')[0] for _ in range(4)]

# ---------------- sliders for Px, Py, θ
ax_x = plt.axes([0.25, 0.25, 0.65, 0.03]);   s_x = Slider(ax_x, "X", -0.8, 0.8,  valinit=0.0)
ax_y = plt.axes([0.25, 0.20, 0.65, 0.03]);   s_y = Slider(ax_y, "Y", -0.1, 0.8,  valinit=0.0)
ax_t = plt.axes([0.25, 0.15, 0.65, 0.03]);   s_t = Slider(ax_t, "θ (deg)", -90, 90, valinit=-23)

def update(_):
    # read pose
    Px = s_x.val
    Py = s_y.val
    th = np.deg2rad(s_t.val)
    # endpoints
    dx = 0.5*BAR_LEN*np.cos(th)
    dy = 0.5*BAR_LEN*np.sin(th)
    Px_v = (Px - dx, Py - dy)  # vertex P
    Hx_v = (Px + dx, Py + dy)  # vertex H
    # draw bar
    bar_line.set_data([Px_v[0], Hx_v[0]], [Px_v[1], Hx_v[1]])
    # string lines & lengths
    lengths = {}
    for ln, (tag,(ax_,ay_)) in zip(str_lines, anchors.items()):
        joint = tag[0]          # 'P' or 'H'
        vx, vy = Px_v if joint=='P' else Hx_v
        ln.set_data([ax_, vx], [ay_, vy])
        lengths[tag] = np.hypot(vx-ax_, vy-ay_)
    # print nicely
    print(f"Lengths  (m):  P_L={lengths['P_L']:.3f}  P_R={lengths['P_R']:.3f}  ",
          f"H_L={lengths['H_L']:.3f}  H_R={lengths['H_R']:.3f}", end='\r')
    fig.canvas.draw_idle()

# connect sliders
for s in (s_x, s_y, s_t):
    s.on_changed(update)

# initial draw
update(None)
plt.show()

