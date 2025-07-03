#!/usr/bin/env python3
"""
robot_sim.py  – 2-D marionette simulator (pymunk + pygame_gui)
Version 1.4.0
----------------------------------------------------------------
Keys
  E : toggle Exact-mode   (freeze / unfreeze the four non-left strings)
  W : toggle Left-arm wave
Sliders
  len  … : string exact length  (greyed when program controls the string)
  limb … : rigid limb length
  X    … : anchor horizontal position
New in 1.4.0
  • Per-string velocity-damping reduces oscillations even in Exact-mode.
"""

__version__ = "1.4.0"

import math, sys, pygame, pymunk, pygame_gui
from   pygame_gui.core import ObjectID

# ────────────── constants & helpers ──────────────
WIDTH, HEIGHT, PPM      = 1000, 800, 300
FPS, DT                 = 60, 1/60
GRAVITY, DAMPING        = (0, -9.81), 1.0
LEFT_WAVE_A, LEFT_WAVE_F = 0.05, 2.0         # m, Hz
STRING_DAMP_GAIN        = 0.10               # m per (m/s) per sec
to_px = lambda v: (int(v.x*PPM + WIDTH/2),
                   int(HEIGHT/2 - v.y*PPM))

# ───────────── pygame / pymunk set-up ─────────────
pygame.init()
screen, clock = (pygame.display.set_mode((WIDTH, HEIGHT)),
                 pygame.time.Clock())
manager = pygame_gui.UIManager((WIDTH, HEIGHT))
space   = pymunk.Space(); space.gravity = GRAVITY; STATIC = space.static_body

# ─────────────── anchor positions ────────────────
A = {k: pymunk.Vec2d(*v) for k,v in {
    "head" :( 0.00, 0.00),
    "handL":(-0.30, 0.00),
    "handR":( 0.30, 0.00),
    "footL":(-0.10, 0.00),
    "footR":( 0.10, 0.00)}.items()}

# ───────────── build joints & limbs ──────────────
def make_joint(pos):
    m,r = 0.02, 0.015
    body = pymunk.Body(m, pymunk.moment_for_circle(m,0,r)); body.position = pos
    body.velocity_func = lambda b,g,d,dt: pymunk.Body.update_velocity(b,g,DAMPING,dt)
    circ = pymunk.Circle(body,r); circ.filter = pymunk.ShapeFilter(group=1)
    space.add(body,circ); return body

j = {"head" :make_joint(( 0.0,-0.1)),
     "should":make_joint(( 0.0,-0.3)),
     "handL": make_joint((-0.2,-0.5)),
     "handR": make_joint(( 0.2,-0.5)),
     "hip"  :make_joint(( 0.0,-0.9)),
     "footL":make_joint((-0.1,-1.2)),
     "footR":make_joint(( 0.1,-1.2))}

limbs={}
def pin(name,a,b):
    pj=pymunk.PinJoint(j[a],j[b]); space.add(pj); limbs[name]=pj
for a,b in [("head","should"),("should","handL"),("should","handR"),
            ("should","hip"),  ("hip","footL"),   ("hip","footR")]:
    pin(f"{a}-{b}",a,b)

# ───────────── build strings (SlideJoint) ────────
strings, cmd_len = {}, {}            # cmd_len keeps current commanded length

def add_string(k,L):
    sj = pymunk.SlideJoint(STATIC, j[k], A[k], (0,0), L, L)  # min=max=L
    sj.collide_bodies = False
    space.add(sj)
    strings[k] = sj
    cmd_len[k]  = L

for k,L in {"head":0.35,"handL":0.50,"handR":0.50,
            "footL":0.70,"footR":0.70}.items():
    add_string(k,L)

# ──────────────── GUI sliders ────────────────────
SL_W,x0,y = 260,10,10
ui={}
def slider(lbl,start,rng):
    global y
    ui[lbl] = pygame_gui.elements.UIHorizontalSlider(
        pygame.Rect((x0,y),(SL_W,20)), start, rng, manager,
        object_id=ObjectID(class_id="@slider",object_id=lbl))
    pygame_gui.elements.UILabel(pygame.Rect((x0+SL_W+6,y),(160,20)),
                                lbl, manager,
                                object_id=ObjectID(class_id="@label",object_id=lbl))
    y += 24

for k in A:                slider(f"len {k}",  cmd_len[k], (0,2))
y += 8
for n,p in limbs.items():  slider(f"limb {n}", p.distance,
                                  (p.distance*0.3, p.distance*2))
y += 8
for k in A:                slider(f"X {k}",    A[k].x, (-0.5,0.5))

# ────────── exact-mode / wave toggles + UI state ──────────
frozen    = ["head","handR","footL","footR"]
exact_on  = True
wave_on   = True
def refresh_ui():
    for k in frozen:
        (ui[f"len {k}"].disable() if exact_on else
         ui[f"len {k}"].enable())
    (ui["len handL"].disable() if wave_on else ui["len handL"].enable())
refresh_ui()

# ────────── helpers: apply commands & damping ──────────────
def apply_commanded_lengths():
    for k,L in cmd_len.items():
        strings[k].min = strings[k].max = L

def damp_strings(dt):
    g = STRING_DAMP_GAIN
    for k, sj in strings.items():
        rel = j[k].position - A[k]
        if rel.length < 1e-6: continue
        diru = rel.normalized()
        v_along = j[k].velocity.dot(diru)      # scalar
        cmd_len[k] -= g * v_along * dt
        cmd_len[k] = max(0.01, cmd_len[k])     # keep positive

def lock_frozen_strings():
    for k in frozen:
        d = A[k].get_distance(j[k].position)
        cmd_len[k] = d

# ─────────────── drawing helper ─────────────────────────────
SEG=[("head","should"),("should","handL"),("should","handR"),
     ("should","hip"),  ("hip","footL"),   ("hip","footR")]

def draw():
    for a,b in SEG:
        pygame.draw.line(screen,(0,180,0),to_px(j[a].position),to_px(j[b].position),5)
    for k in A:
        pygame.draw.line(screen,(200,0,0),to_px(A[k]),to_px(j[k].position),2)

# ───────────────────── main loop ────────────────────────────
t = 0.0
running = True
while running:
    dt = clock.tick(FPS)/1000; t += dt
    for ev in pygame.event.get():
        if ev.type == pygame.QUIT:
            running = False
        if ev.type == pygame.KEYDOWN:
            if ev.key == pygame.K_e:
                exact_on = not exact_on; refresh_ui()
            if ev.key == pygame.K_w:
                wave_on  = not wave_on;  refresh_ui()
        manager.process_events(ev)
    manager.update(dt)

    # 1.  feed-forward left-hand wave or slider value
    if wave_on:
        cmd_len["handL"] = 0.50 + LEFT_WAVE_A*math.sin(2*math.pi*LEFT_WAVE_F*t)
    else:
        if ui["len handL"].is_enabled:
            cmd_len["handL"] = ui["len handL"].get_current_value()

    # 2.  sliders → commands
    for lbl,s in ui.items():
        if not s.is_enabled: continue
        v = s.get_current_value()
        if lbl.startswith("len "):   cmd_len[lbl[4:]] = v
        elif lbl.startswith("limb "): limbs[lbl[5:]].distance = v
        elif lbl.startswith("X "):
            k = lbl[2:]; A[k] = pymunk.Vec2d(v, A[k].y)
            strings[k].anchor_a = A[k]

    # 3.  apply exact lock (if ON) *before* damping
    if exact_on: lock_frozen_strings()

    # 4.  energy-damping tweak   (small ΔL against velocity)
    damp_strings(dt)

    # 5.  push commanded lengths to constraints
    apply_commanded_lengths()

    space.step(DT)

    screen.fill((240,240,240)); draw(); manager.draw_ui(screen)
    pygame.display.set_caption(
        f"robot_sim v{__version__} | Exact:{'ON' if exact_on else 'OFF'} (E) "
        f"| Wave:{'ON' if wave_on else 'OFF'} (W)"
    )
    pygame.display.flip()

pygame.quit(); sys.exit()

