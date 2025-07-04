#!/usr/bin/env python3
"""
robot_sim.py  – 2-D marionette simulator (pymunk + pygame_gui)
Version 1.5.0
----------------------------------------------------------------
• Shorter strings & limbs (70 % scale)
• Bodies heavier (0.05 kg) and global damping = 2 s⁻¹
• Per-string velocity damping gain = 0.03
Keys:
  E – toggle Exact-mode   (freeze / unfreeze non-left strings)
  W – toggle Left-hand wave
"""

__version__ = "1.5.0"

import math, sys, pygame, pymunk, pygame_gui
from   pygame_gui.core import ObjectID

# ───────────────────────── constants ──────────────────────────
WIDTH, HEIGHT, PPM   = 1000, 800, 300
FPS, DT              = 60, 1/60
GRAVITY              = (0, -9.81)
BODY_MASS, BODY_R    = 0.05, 0.015   # heavier links
GLOBAL_DAMPING       = -1.0           # s⁻¹
ANCHOR_Y             = 0.75          # anchors near top
SCALE                = 0.70          # limb length scale (shorter strings)
LEFT_WAVE_A, LEFT_WAVE_F = 0.05, 2.0 # m, Hz
STRING_DAMP_GAIN     = -0.01          # m / (m·s) damping

to_px = lambda v: (int(v.x*PPM + WIDTH/2),
                   int(HEIGHT/2 - v.y*PPM))

# ────────────────── pygame / pymunk init ──────────────────────
pygame.init()
screen, clock = pygame.display.set_mode((WIDTH, HEIGHT)), pygame.time.Clock()
manager       = pygame_gui.UIManager((WIDTH, HEIGHT))
space         = pymunk.Space(); space.gravity = GRAVITY; STATIC = space.static_body
space.iterations = 20          # let the solver work harder


# ───────────────────── anchors (Vec2d) ───────────────────────
A = {k: pymunk.Vec2d(*v) for k,v in {
    "head" :( 0.00, ANCHOR_Y),
    "handL":(-0.30*SCALE, ANCHOR_Y),
    "handR":( 0.30*SCALE, ANCHOR_Y),
    "footL":(-0.10*SCALE, ANCHOR_Y),
    "footR":( 0.10*SCALE, ANCHOR_Y)}.items()}

# ───────────── helper: build damped joint body ───────────────
def make_joint(pos):
    body = pymunk.Body(BODY_MASS, pymunk.moment_for_circle(BODY_MASS, 0, BODY_R))
    body.position = pos
    body.velocity_func = lambda b,g,d,dt: pymunk.Body.update_velocity(
        b, g, GLOBAL_DAMPING, dt)
    circ = pymunk.Circle(body, BODY_R); circ.filter = pymunk.ShapeFilter(group=1)
    space.add(body, circ)
    return body

# ───────────────────── joints & limbs ─────────────────────────
# scaled vertical coordinates (shorter puppet)
def y(val): return val * SCALE
j = { "head" : make_joint(( 0.0,         y(-0.10))),
      "should": make_joint(( 0.0,         y(-0.30))),
      "handL" : make_joint(( y(-0.20),    y(-0.35))),
      "handR" : make_joint(( y( 0.20),    y(-0.35))),
      "hip"   : make_joint(( 0.0,         y(-0.63))),
      "footL" : make_joint(( y(-0.10),    y(-0.84))),
      "footR" : make_joint(( y( 0.10),    y(-0.84))) }

limbs={}
def pin(n,a,b):
    pj = pymunk.PinJoint(j[a], j[b]); space.add(pj); limbs[n]=pj
for a,b in [("head","should"),("should","handL"),("should","handR"),
            ("should","hip"),  ("hip","footL"),   ("hip","footR")]:
    pin(f"{a}-{b}", a, b)

# ───────────────────── strings (SlideJoint) ───────────────────
strings, cmd_len = {}, {}
def add_string(k):
    L = A[k].get_distance(j[k].position)
    sj = pymunk.SlideJoint(STATIC, j[k], A[k], (0,0), L, L)
    sj.collide_bodies = False
    space.add(sj)
    strings[k] = sj
    cmd_len[k]  = L
for k in A: add_string(k)

# ───────────────────── GUI sliders ────────────────────────────
SL_W,x0,ypos = 260, 10, 10
ui={}
def slider(lbl,start,rng):
    global ypos
    ui[lbl]=pygame_gui.elements.UIHorizontalSlider(
        pygame.Rect((x0,ypos),(SL_W,20)),start,rng,manager,
        object_id=ObjectID(class_id="@slider",object_id=lbl))
    pygame_gui.elements.UILabel(pygame.Rect((x0+SL_W+6,ypos),(160,20)),
                                lbl,manager,
                                object_id=ObjectID(class_id="@label",object_id=lbl))
    ypos += 24
for k in A:                slider(f"len {k}",  cmd_len[k], (0, cmd_len[k]*2))
ypos += 8
for n,p in limbs.items():  slider(f"limb {n}", p.distance,
                                  (p.distance*0.5, p.distance*2))
ypos += 8
for k in A:                slider(f"X {k}", A[k].x, (-0.5, 0.5))

# ───────── exact-mode / wave toggles & UI refresh ────────────
frozen   = ["head","handR","footL","footR"]
exact_on = True
wave_on  = True
def refresh_ui():
    for k in frozen:
        ui[f"len {k}"].disable() if exact_on else ui[f"len {k}"].enable()
    ui["len handL"].disable() if wave_on else ui["len handL"].enable()
refresh_ui()

# ─────────── helpers: apply cmd_len & damping ────────────────
def enforce_len():
    for k,L in cmd_len.items():
        strings[k].min = strings[k].max = L

def damp_strings(dt):
    g = STRING_DAMP_GAIN
    for k in strings:
        rel = j[k].position - A[k]
        if rel.length < 1e-6: continue
        v_along = j[k].velocity.dot(rel.normalized())
        cmd_len[k] -= g * v_along * dt
        cmd_len[k]  = max(0.01, cmd_len[k])
        ΔL = -g * v_along * dt
        ΔL = max(-0.002, min(0.002, ΔL))   # clamp to ±2 mm
        cmd_len[k] += ΔL

def lock_frozen():
    for k in frozen:
        cmd_len[k] = A[k].get_distance(j[k].position)

# ───────────── drawing helper ────────────────────────────────
SEG=[("head","should"),("should","handL"),("should","handR"),
     ("should","hip"),  ("hip","footL"),   ("hip","footR")]
def draw():
    for a,b in SEG:
        pygame.draw.line(screen,(0,180,0),to_px(j[a].position),to_px(j[b].position),5)
    for k in A:
        pygame.draw.line(screen,(200,0,0),to_px(A[k]),to_px(j[k].position),2)

# ───────────────────── main loop ─────────────────────────────
t=0.0; running=True
while running:
    dt = clock.tick(FPS)/1000; t += dt
    for ev in pygame.event.get():
        if ev.type==pygame.QUIT: running=False
        if ev.type==pygame.KEYDOWN:
            if ev.key==pygame.K_e: exact_on=not exact_on; refresh_ui()
            if ev.key==pygame.K_w: wave_on =not wave_on ; refresh_ui()
        manager.process_events(ev)
    manager.update(dt)

    # left-hand command
    if wave_on:
        cmd_len["handL"] = 0.35 + LEFT_WAVE_A*math.sin(2*math.pi*LEFT_WAVE_F*t)
    elif ui["len handL"].is_enabled:
        cmd_len["handL"] = ui["len handL"].get_current_value()

    # sliders
    for lbl,s in ui.items():
        if not s.is_enabled: continue
        v = s.get_current_value()
        if lbl.startswith("len "):
            cmd_len[lbl[4:]] = v
        elif lbl.startswith("limb "):
            limbs[lbl[5:]].distance = v
        elif lbl.startswith("X "):
            k = lbl[2:]; A[k]=pymunk.Vec2d(v,A[k].y); strings[k].anchor_a=A[k]

    if exact_on: lock_frozen()
    damp_strings(DT)
    enforce_len()
    for _ in range(4):               # four small steps, same wall-clock pace
        space.step(DT/4)

    screen.fill((240,240,240)); draw(); manager.draw_ui(screen)
    pygame.display.set_caption(
        f"robot_sim v{__version__} | Exact:{'ON' if exact_on else 'OFF'} (E) "
        f"| Wave:{'ON' if wave_on else 'OFF'} (W)"
    )
    pygame.display.flip()

pygame.quit(); sys.exit()

