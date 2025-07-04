#!/usr/bin/env python3
"""
analytical_puppet.py
──────────────────────────────────────────────────────────────
Pose-driven stick figure hung from fixed torso midpoint
(two overhead strings per vertex).

Torso midpoint is the origin (0,0) and only rotates (slider “Torso°”).
Sliders pose each limb; every move updates the strings instantly.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# ─── link lengths (m) ───────────────────────────────────────
L_TORSO, L_NECK = 0.40, 0.10
L_UARM = L_LARM = 0.25
L_ULEG = L_LLEG = 0.35

# ─── anchor layout ──────────────────────────────────────────
ANCH_XSEP   = 0.25          # left/right spacing per vertex
ANCH_Y_TOP  = 1.2
DY_ANCH     = 0.15

anchor_y = {
    'HEAD':      ANCH_Y_TOP,
    'SHOULDER':  ANCH_Y_TOP - DY_ANCH,
    'ELBOW_L':   ANCH_Y_TOP - 2*DY_ANCH,
    'ELBOW_R':   ANCH_Y_TOP - 2*DY_ANCH,
    'HAND_L':    ANCH_Y_TOP - 3*DY_ANCH,
    'HAND_R':    ANCH_Y_TOP - 3*DY_ANCH,
    'HIP':       ANCH_Y_TOP - 2*DY_ANCH,
    'KNEE_L':    ANCH_Y_TOP - 3*DY_ANCH,
    'KNEE_R':    ANCH_Y_TOP - 3*DY_ANCH,
    'FOOT_L':    ANCH_Y_TOP - 4*DY_ANCH,
    'FOOT_R':    ANCH_Y_TOP - 4*DY_ANCH,
}

verts = ['HIP','SHOULDER','HEAD',
         'ELBOW_L','HAND_L','ELBOW_R','HAND_R',
         'KNEE_L','FOOT_L','KNEE_R','FOOT_R']

# helper: anchor x-coords for a vertex
def anchors_of(x):
    return x - ANCH_XSEP/2, x + ANCH_XSEP/2

# ─── forward kinematics (torso pivots about its midpoint = origin) ───
def fk(p):
    thT = np.deg2rad(p['torso'])          # torso orientation
    # shoulder and hip are equally distant from origin
    Sx =  +0.5*L_TORSO*np.sin(thT)
    Sy =  +0.5*L_TORSO*np.cos(thT)
    Px =  -0.5*L_TORSO*np.sin(thT)
    Py =  -0.5*L_TORSO*np.cos(thT)

    Hx, Hy =  Sx, Sy + L_NECK             # head directly above shoulder

    # left arm (angles measured from torso normal)
    shL = thT + np.deg2rad(p['sh_L'])
    ELx = Sx + L_UARM*np.sin(shL)
    ELy = Sy - L_UARM*np.cos(shL)
    eL  = shL + np.deg2rad(p['el_L'])
    HLx = ELx + L_LARM*np.sin(eL)
    HLy = ELy - L_LARM*np.cos(eL)

    # right arm
    shR = thT - np.deg2rad(p['sh_R'])
    ERx = Sx + L_UARM*np.sin(shR)
    ERy = Sy - L_UARM*np.cos(shR)
    eR  = shR - np.deg2rad(p['el_R'])
    HRx = ERx + L_LARM*np.sin(eR)
    HRy = ERy - L_LARM*np.cos(eR)

    # left leg (angles from torso axis)
    hipL = thT - np.deg2rad(p['hip_L'])
    KLx  = Px + L_ULEG*np.sin(hipL)
    KLy  = Py - L_ULEG*np.cos(hipL)
    kL   = hipL - np.deg2rad(p['kn_L'])
    FLx  = KLx + L_LLEG*np.sin(kL)
    FLy  = KLy - L_LLEG*np.cos(kL)

    # right leg
    hipR = thT + np.deg2rad(p['hip_R'])
    KRx  = Px + L_ULEG*np.sin(hipR)
    KRy  = Py - L_ULEG*np.cos(hipR)
    kR   = hipR + np.deg2rad(p['kn_R'])
    FRx  = KRx + L_LLEG*np.sin(kR)
    FRy  = KRy - L_LLEG*np.cos(kR)

    return {
        'HIP':(Px,Py),          'SHOULDER':(Sx,Sy), 'HEAD':(Hx,Hy),
        'ELBOW_L':(ELx,ELy),    'HAND_L':(HLx,HLy),
        'ELBOW_R':(ERx,ERy),    'HAND_R':(HRx,HRy),
        'KNEE_L':(KLx,KLy),     'FOOT_L':(FLx,FLy),
        'KNEE_R':(KRx,KRy),     'FOOT_R':(FRx,FRy)
    }


# ─── matplotlib scene ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(7,7))
plt.subplots_adjust(left=0.28, bottom=0.05, right=0.98)
ax.set_aspect('equal')
ax.set_xlim(-1.1,1.1); ax.set_ylim(-0.6,1.3)
ax.set_title("Fixed-pivot puppet – string lengths in console")

# anchor dots
for v in verts:
    ay = anchor_y[v]
    a1,a2 = anchors_of(0)
    ax.scatter([a1,a2],[ay,ay],c='k',s=18)

# dynamic artists
lines = [ax.plot([],[],'b-',lw=4)[0] for _ in range(10)]   # skeleton
strings = [ax.plot([],[],'r--')[0] for _ in range(len(verts)*2)]

# ─── sliders ------------------------------------------------
cols, rows = 5, 2
dx, dy     = 0.13, 0.10          # slider width & height
x0, y0     = 0.02, 0.92          # top-left of grid
specs = [
    ("Torso°", -90,  90, -25),
    ("sh_L°", -120,   0, -40), ("el_L°", -120,   0, -50),
    ("sh_R°", -120,   0, -40), ("el_R°", -120,   0, -50),
    ("hip_L°", -60,  60,  20), ("kn_L°", -120,   0, -40),
    ("hip_R°", -60,  60,  20), ("kn_R°", -120,   0, -40),
]
sliders = {}
for i,(lab,lo,hi,val) in enumerate(specs):
    col, row = divmod(i, cols)
    ax_s = plt.axes([x0 + col*dx, y0 - row*dy, dx*0.9, 0.03])
    sliders[lab] = Slider(ax_s, lab, lo, hi, valinit=val)


# parameter holder
P=dict(torso = sliders["Torso°"].val,
       sh_L  = sliders["sh_L°"].val, el_L = sliders["el_L°"].val,
       sh_R  = sliders["sh_R°"].val, el_R = sliders["el_R°"].val,
       hip_L = sliders["hip_L°"].val, kn_L= sliders["kn_L°"].val,
       hip_R = sliders["hip_R°"].val, kn_R= sliders["kn_R°"].val)

def redraw(_=None):
    # update params from sliders
    P['torso'] = sliders["Torso°"].val
    P['sh_L'], P['el_L'] = sliders["sh_L°"].val, sliders["el_L°"].val
    P['sh_R'], P['el_R'] = sliders["sh_R°"].val, sliders["el_R°"].val
    P['hip_L'],P['kn_L'] = sliders["hip_L°"].val, sliders["kn_L°"].val
    P['hip_R'],P['kn_R'] = sliders["hip_R°"].val, sliders["kn_R°"].val

    V=fk(P)

    # skeleton segments
    segs=[ (V['HIP'],V['SHOULDER']),(V['SHOULDER'],V['HEAD']),
           (V['SHOULDER'],V['ELBOW_L']),(V['ELBOW_L'],V['HAND_L']),
           (V['SHOULDER'],V['ELBOW_R']),(V['ELBOW_R'],V['HAND_R']),
           (V['HIP'],V['KNEE_L']),(V['KNEE_L'],V['FOOT_L']),
           (V['HIP'],V['KNEE_R']),(V['KNEE_R'],V['FOOT_R']) ]
    for ln,(p0,p1) in zip(lines,segs):
        ln.set_data([p0[0],p1[0]],[p0[1],p1[1]])

    # strings + length print
    out=[]; idx=0
    for v in verts:
        vx,vy=V[v]; ay=anchor_y[v]; a1,a2=anchors_of(0)
        strings[idx].set_data([a1,vx],[ay,vy])
        strings[idx+1].set_data([a2,vx],[ay,vy])
        out.append(f"{v}:{np.hypot(vx-a1,vy-ay):.2f},{np.hypot(vx-a2,vy-ay):.2f}")
        idx+=2
    print(" | ".join(out),end='\r')
    fig.canvas.draw_idle()

for s in sliders.values(): s.on_changed(redraw)

redraw()
plt.show()

