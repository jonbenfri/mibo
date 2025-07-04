#!/usr/bin/env python3
"""
analytical_puppet.py  –  planar arm with two strings per joint
Version 2.2.5
-----------------------------------------------------------------
Fix: store FuncAnimation in variable `ani` so it isn’t garbage-collected
before plt.show(), therefore the arm & strings render.
"""

import numpy as np, sympy as sp, sympy.physics.mechanics as me
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ───────── parameters ─────────
L, X0, BAR_Y = 0.35, 0.20, 0.75
MASS         = 0.05
G_DAMP       = 2.0
STRING_DAMP  = 0.03
ALPHA_B = BETA_B = 15.0
T_MAX, FPS  = 5.0, 60

# ────── Symbolic model (unchanged from v2.2.4) ──────
t = sp.symbols('t')
x, y, th      = me.dynamicsymbols('x y th')
u1, u2, u3    = me.dynamicsymbols('u1 u2 u3')

N  = me.ReferenceFrame('N')
A  = N.orientnew('A', 'Axis', (th, N.z))

O  = me.Point('O')
P  = me.Point('P'); P.set_pos(O, x*N.x + y*N.y)
P.set_vel(N, u1*N.x + u2*N.y)

H  = P.locatenew('H',  L*sp.cos(th)*N.x + L*sp.sin(th)*N.y)
H.v2pt_theory(P, N, A)

arm = me.RigidBody('arm', P, A, MASS, (me.inertia(N,0,0,1e-3), P))

anchors = { 'P_L':(-X0/2, BAR_Y), 'P_R':( X0/2, BAR_Y),
            'H_L':(-X0/2, BAR_Y), 'H_R':( X0/2, BAR_Y) }
points   = { 'P':P, 'H':H }

phi_list, L_syms = [], {}
for tag,(ax,ay) in anchors.items():
    joint,_ = tag.split('_')
    pt      = points[joint]
    dist    = sp.sqrt((pt.pos_from(O).dot(N.x)-ax)**2 +
                      (pt.pos_from(O).dot(N.y)-ay)**2)
    Ls      = sp.symbols(f'L_{tag}')
    L_syms[tag]=Ls
    phi_list.append(dist - Ls)
phi = sp.Matrix(phi_list)

q = sp.Matrix([x,y,th])
u = sp.Matrix([u1,u2,u3])
kd_eqs = [u1-x.diff(t), u2-y.diff(t), u3-th.diff(t)]

km = me.KanesMethod(N,q_ind=q,u_ind=u,kd_eqs=kd_eqs)
km.kanes_equations([arm],loads=[])

M_sym = km.mass_matrix
F_sym = km.forcing
J_sym = phi.jacobian(q)

state_syms=list(q)+list(u)
L_symbols=list(L_syms.values())
lM   = sp.lambdify(state_syms,            M_sym,'numpy')
lF   = sp.lambdify(state_syms,            F_sym,'numpy')
lJ   = sp.lambdify(state_syms+L_symbols,  J_sym,'numpy')
lphi = sp.lambdify(state_syms+L_symbols,  phi,'numpy')

# ───── initial pose & taut lengths ─────
q0 = np.array([0.0, 0.0, -0.4]);  u0 = np.zeros(3)
L_cmd = {}
for tag,(ax,ay) in anchors.items():
    joint,_=tag.split('_')
    px,py = (q0[0],q0[1]) if joint=='P' else \
            (q0[0]+L*np.cos(q0[2]), q0[1]+L*np.sin(q0[2]))
    L_cmd[tag]=float(np.hypot(px-ax, py-ay))
y0 = np.hstack([q0,u0])

# ───── RHS (same as v2.2.4) ─────
def rhs(_,s):
    qv,uv=s[:3],s[3:]; vec=np.r_[qv,uv]
    # string viscous damping
    for tag,(ax,ay) in anchors.items():
        joint,_=tag.split('_')
        px,py=(qv[0],qv[1]) if joint=='P' else \
              (qv[0]+L*np.cos(qv[2]), qv[1]+L*np.sin(qv[2]))
        vx,vy=(uv[0],uv[1]) if joint=='P' else \
              (uv[0]-L*uv[2]*np.sin(qv[2]), uv[1]+L*uv[2]*np.cos(qv[2]))
        rel=np.array([px-ax,py-ay]); n=np.linalg.norm(rel)
        if n>1e-9:
            L_cmd[tag]-=STRING_DAMP*(rel/n).dot([vx,vy])
            L_cmd[tag]=max(0.01,L_cmd[tag])

    Lvals=[L_cmd[k] for k in L_syms]
    M   = lM(*vec).astype(float)
    Fv  = lF(*vec).astype(float).flatten()
    J   = lJ(*(vec.tolist()+Lvals)).astype(float)
    ph  = lphi(*(vec.tolist()+Lvals)).astype(float).flatten()
    gamma = -2*ALPHA_B*(J@uv) - (BETA_B**2)*ph
    KKT=np.block([[M,J.T],[J,np.zeros((4,4))]])
    sol,_ ,_,_=np.linalg.lstsq(KKT, np.hstack([Fv,-gamma]), rcond=None)
    return np.hstack([uv, sol[:3]])

sol=solve_ivp(rhs,(0,T_MAX),y0,max_step=0.01,rtol=1e-9,atol=1e-9)

# ───── animation ─────
fig,ax=plt.subplots(); ax.set_aspect('equal')
ax.set_xlim(-0.6,0.6); ax.set_ylim(-0.3,1.0)
for pos in anchors.values(): ax.plot(*pos,'ko')
arm_ln,=ax.plot([],[],'b-',lw=4)
str_lns=[ax.plot([],[],'r--')[0] for _ in range(4)]
Aarr=np.array(list(anchors.values()))
def animate(i):
    x,y,th=sol.y[0,i],sol.y[1,i],sol.y[2,i]
    Pxy=np.array([x,y]); Hxy=Pxy+L*np.array([np.cos(th),np.sin(th)])
    arm_ln.set_data([Pxy[0],Hxy[0]],[Pxy[1],Hxy[1]])
    joints=np.vstack([Pxy,Pxy,Hxy,Hxy])
    for ln,a,j in zip(str_lns,Aarr,joints):
        ln.set_data([a[0],j[0]],[a[1],j[1]])
    return [arm_ln,*str_lns]

ani = FuncAnimation(fig, animate, frames=len(sol.t),  # ← keep reference
                    interval=1000//FPS, blit=True)

plt.title("Two strings per joint – v2.2.5")
plt.show()

