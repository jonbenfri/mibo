#!/usr/bin/env python3
"""
robot_sim.py  – 2-D marionette simulator (pymunk + pygame_gui)
Version 1.3.0
----------------------------------------------------------------
• All strings are now enforced as exact-length rods each frame
  (sj.min = sj.max = commanded length) – prevents droop when exact-mode off.
• E  – toggle exact-mode (freezes / unfreezes the four non-left strings)
• W  – toggle left-hand wave on/off
"""

__version__ = "1.3.0"

import math, sys, pygame, pymunk, pygame_gui
from   pygame_gui.core import ObjectID

# ─────────────────────────  constants  ──────────────────────────
WIDTH, HEIGHT, PPM = 1000, 800, 300
FPS, DT            = 60, 1 / 60
GRAVITY, DAMPING   = (0, -9.81), 1.0
LEFT_WAVE_A, LEFT_WAVE_F = 0.05, 2.0      # metres, Hz
to_px = lambda v: (int(v.x * PPM + WIDTH / 2),
                   int(HEIGHT / 2 - v.y * PPM))

# ──────────────────────  pygame / pymunk init  ──────────────────
pygame.init()
screen, clock = pygame.display.set_mode((WIDTH, HEIGHT)), pygame.time.Clock()
manager       = pygame_gui.UIManager((WIDTH, HEIGHT))
space         = pymunk.Space()
space.gravity = GRAVITY
STATIC        = space.static_body

# ─────────────────────────  anchors  ────────────────────────────
A = {k: pymunk.Vec2d(*v) for k, v in {
    "head" :( 0.00, 0.00),
    "handL":(-0.30, 0.00),
    "handR":( 0.30, 0.00),
    "footL":(-0.10, 0.00),
    "footR":( 0.10, 0.00)}.items()}

# ──────────────────  helper: create damped joint  ───────────────
def make_joint(pos):
    m, r  = 0.02, 0.015
    body  = pymunk.Body(m, pymunk.moment_for_circle(m, 0, r)); body.position = pos
    body.velocity_func = lambda b,g,d,dt: pymunk.Body.update_velocity(b, g, DAMPING, dt)
    circ  = pymunk.Circle(body, r); circ.filter = pymunk.ShapeFilter(group=1)
    space.add(body, circ)
    return body

# joints
j = {"head" : make_joint(( 0.0, -0.1)),
     "should": make_joint(( 0.0, -0.3)),
     "handL": make_joint((-0.2, -0.5)),
     "handR": make_joint(( 0.2, -0.5)),
     "hip"  : make_joint(( 0.0, -0.9)),
     "footL": make_joint((-0.1, -1.2)),
     "footR": make_joint(( 0.1, -1.2))}

# limbs (PinJoint)
limbs={}
def pin(n,a,b):
    pj = pymunk.PinJoint(j[a], j[b]); space.add(pj); limbs[n] = pj
for a,b in [("head","should"),("should","handL"),("should","handR"),
            ("should","hip"),  ("hip","footL"),   ("hip","footR")]:
    pin(f"{a}-{b}", a, b)

# strings (SlideJoint) – stored in dict
strings={}
def add_string(key, length):
    sj = pymunk.SlideJoint(STATIC, j[key], A[key], (0,0), length, length) # min=max=length
    sj.collide_bodies = False
    space.add(sj)
    strings[key] = sj
for k, L in {"head":0.35,"handL":0.50,"handR":0.50,"footL":0.70,"footR":0.70}.items():
    add_string(k, L)

# ─────────────────────────  GUI sliders  ────────────────────────
SL_W, x0, y = 260, 10, 10
ui={}
def slider(label, start, rng):
    global y
    ui[label] = pygame_gui.elements.UIHorizontalSlider(
        pygame.Rect((x0, y), (SL_W, 20)),
        start, rng, manager,
        object_id = ObjectID(class_id="@slider", object_id=label)
    )
    pygame_gui.elements.UILabel(
        pygame.Rect((x0 + SL_W + 6, y), (160, 20)),
        label, manager,
        object_id = ObjectID(class_id="@label", object_id=label)
    )
    y += 24

for k in A:               slider(f"len {k}",  strings[k].max, (0, 2))
y += 8
for name, pj in limbs.items():
    slider(f"limb {name}", pj.distance,
           (pj.distance * 0.3, pj.distance * 2))
y += 8
for k in A:               slider(f"X {k}", A[k].x, (-0.5, 0.5))

# ───────────────  exact-mode & wave toggles  ────────────────────
frozen    = ["head","handR","footL","footR"]
exact_on  = True
wave_on   = True

def lock_frozen():
    for k in frozen:
        d = A[k].get_distance(j[k].position)
        strings[k].min = strings[k].max = d

def refresh_slider_state():
    for k in frozen:
        ui[f"len {k}"].enable = (not exact_on)
    ui["len handL"].enable = (not wave_on)

refresh_slider_state()

# ─────────────────────  drawing helper  ─────────────────────────
SEG=[("head","should"),("should","handL"),("should","handR"),
     ("should","hip"),  ("hip","footL"),   ("hip","footR")]
def draw():
    for a,b in SEG:
        pygame.draw.line(screen,(0,180,0),to_px(j[a].position),to_px(j[b].position),5)
    for k in A:
        pygame.draw.line(screen,(200,0,0),to_px(A[k]),to_px(j[k].position),2)

# ─────────────────────────  main loop  ──────────────────────────
t=0.0; running=True
while running:
    dt = clock.tick(FPS) / 1000.0; t += dt
    for ev in pygame.event.get():
        if ev.type == pygame.QUIT:
            running = False
        if ev.type == pygame.KEYDOWN:
            if ev.key == pygame.K_e:
                exact_on = not exact_on; refresh_slider_state()
            if ev.key == pygame.K_w:
                wave_on  = not wave_on;  refresh_slider_state()
        manager.process_events(ev)
    manager.update(dt)

    # left-hand string command
    if wave_on:
        cmd = 0.50 + LEFT_WAVE_A * math.sin(2 * math.pi * LEFT_WAVE_F * t)
        strings["handL"].min = strings["handL"].max = cmd
    else:
        if ui["len handL"].enable:
            v = ui["len handL"].get_current_value()
            strings["handL"].min = strings["handL"].max = v

    # sliders → physics
    for label, s in ui.items():
        if not s.enable: continue
        v = s.get_current_value()
        if label.startswith("len "):
            k = label[4:]
            strings[k].min = strings[k].max = v
        elif label.startswith("limb "):
            limbs[label[5:]].distance = v
        elif label.startswith("X "):
            k = label[2:]
            A[k] = pymunk.Vec2d(v, A[k].y)
            strings[k].anchor_a = A[k]

    if exact_on: lock_frozen()
    space.step(DT)

    screen.fill((240,240,240))
    draw()
    manager.draw_ui(screen)
    pygame.display.set_caption(
        f"robot_sim v{__version__} | Exact:{'ON' if exact_on else 'OFF'} (E) | "
        f"Wave:{'ON' if wave_on else 'OFF'} (W)"
    )
    pygame.display.flip()

pygame.quit(); sys.exit()

