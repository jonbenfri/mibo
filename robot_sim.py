#!/usr/bin/env python3
"""
robot_sim.py  – 2-D marionette simulator (pymunk + pygame_gui)
Version 1.2.1
-------------------------------------------------------------
Keys
• E  – toggle exact-mode (freeze all but left hand)
• W  – toggle left-hand wave on/off

Sliders
• len …   – string lengths          (greyed when locked)
• limb …  – rigid limb lengths
• X …     – anchor X positions
"""

__version__ = "1.2.1"

import math, sys, pygame, pymunk, pygame_gui
from   pygame_gui.core import ObjectID

# ─────────────────────────  constants  ──────────────────────────
WIDTH, HEIGHT, PPM   = 1000, 800, 300
FPS, DT              = 60, 1/60
GRAVITY, DAMPING     = (0, -9.81), 1.0
LEFT_WAVE_A, LEFT_WAVE_F = 0.05, 2.0        # m, Hz
to_px = lambda v: (int(v.x*PPM + WIDTH/2), int(HEIGHT/2 - v.y*PPM))

# ───────────────────────  pygame / pymunk  ──────────────────────
pygame.init()
screen, clock = pygame.display.set_mode((WIDTH, HEIGHT)), pygame.time.Clock()
manager = pygame_gui.UIManager((WIDTH, HEIGHT))
space   = pymunk.Space(); space.gravity = GRAVITY; STATIC = space.static_body

# ──────────────────────────  anchors  ───────────────────────────
A = {k: pymunk.Vec2d(*v) for k, v in {
    "head": (0.00, 0.00),
    "handL":(-0.30, 0.00),
    "handR":( 0.30, 0.00),
    "footL":(-0.10, 0.00),
    "footR":( 0.10, 0.00)}.items()}

# ──────────────────────  helper: damped joint  ──────────────────
def make_joint(pos):
    m, r = 0.02, 0.015
    body = pymunk.Body(m, pymunk.moment_for_circle(m, 0, r)); body.position = pos
    body.velocity_func = lambda b,g,d,dt: pymunk.Body.update_velocity(b,g,DAMPING,dt)
    shp = pymunk.Circle(body, r); shp.filter = pymunk.ShapeFilter(group=1)
    space.add(body, shp)
    return body

# bodies
j = {"head":make_joint(( 0.0,-0.1)),
     "should":make_joint(( 0.0,-0.3)),
     "handL":make_joint((-0.2,-0.5)),
     "handR":make_joint(( 0.2,-0.5)),
     "hip":make_joint(( 0.0,-0.9)),
     "footL":make_joint((-0.1,-1.2)),
     "footR":make_joint(( 0.1,-1.2))}

# limbs (PinJoint)
limbs={}
def pin(n,a,b):
    pj = pymunk.PinJoint(j[a], j[b]); space.add(pj); limbs[n]=pj
for a,b in [("head","should"),("should","handL"),("should","handR"),
            ("should","hip"),  ("hip","footL"),   ("hip","footR")]:
    pin(f"{a}-{b}", a, b)

# strings (SlideJoint)  ← FIX lives here
strings={}
def sld(key, length):
    sj = pymunk.SlideJoint(STATIC, j[key], A[key], (0,0), 0.0, length)
    sj.collide_bodies = False
    space.add(sj)
    strings[key] = sj

for k,L in {"head":0.35,"handL":0.50,"handR":0.50,"footL":0.70,"footR":0.70}.items():
    sld(k, L)

# ─────────────────────────  gui sliders  ────────────────────────
SL_W, x0, y = 260, 10, 10
ui={}
def slider(lbl,start,rng):
    global y
    ui[lbl]=pygame_gui.elements.UIHorizontalSlider(
        pygame.Rect((x0,y),(SL_W,20)),start,rng,manager,
        object_id=ObjectID(class_id="@slider",object_id=lbl))
    pygame_gui.elements.UILabel(
        pygame.Rect((x0+SL_W+6,y),(160,20)),lbl,manager,
        object_id=ObjectID(class_id="@label",object_id=lbl))
    y+=24

for k in A:               slider(f"len {k}",  strings[k].max, (0,2))
y+=8
for n,p in limbs.items(): slider(f"limb {n}", p.distance, (p.distance*0.3, p.distance*2))
y+=8
for k in A:               slider(f"X {k}",    A[k].x, (-0.5,0.5))

# ───────────────  exact-mode & wave toggles  ────────────────────
frozen    = ["head","handR","footL","footR"]
exact_on  = True
wave_on   = True
def lock_frozen():
    for k in frozen:
        strings[k].max = A[k].get_distance(j[k].position)
def update_slider_state():
    for k in frozen:
        ui[f"len {k}"].enabled = (not exact_on)
    ui["len handL"].enabled = (not wave_on)
update_slider_state()

# ─────────────────────────  draw helper  ────────────────────────
SEG=[("head","should"),("should","handL"),("should","handR"),
     ("should","hip"), ("hip","footL"),   ("hip","footR")]
def draw():
    for a,b in SEG:
        pygame.draw.line(screen,(0,180,0),to_px(j[a].position),to_px(j[b].position),5)
    for k in A:
        pygame.draw.line(screen,(200,0,0),to_px(A[k]),to_px(j[k].position),2)

# ──────────────────────────  main loop  ─────────────────────────
t=0.0; running=True
while running:
    dt = clock.tick(FPS)/1000; t+=dt
    for event in pygame.event.get():
        if event.type==pygame.QUIT: running=False
        if event.type==pygame.KEYDOWN:
            if event.key==pygame.K_e:
                exact_on = not exact_on; update_slider_state()
            if event.key==pygame.K_w:
                wave_on  = not wave_on;  update_slider_state()
        manager.process_events(event)
    manager.update(dt)

    # left-hand drive
    if wave_on:
        strings["handL"].max = 0.50 + LEFT_WAVE_A*math.sin(2*math.pi*LEFT_WAVE_F*t)
    elif ui["len handL"].enabled:
        strings["handL"].max = ui["len handL"].get_current_value()

    # sliders → physics
    for lbl,s in ui.items():
        if not s.enable: continue
        v = s.get_current_value()
        if lbl.startswith("len "):
            strings[lbl[4:]].max = v
        elif lbl.startswith("limb "):
            limbs[lbl[5:]].distance = v
        elif lbl.startswith("X "):
            key = lbl[2:]; A[key] = pymunk.Vec2d(v, A[key].y); strings[key].anchor_a = A[key]

    if exact_on: lock_frozen()
    space.step(DT)

    screen.fill((240,240,240)); draw(); manager.draw_ui(screen)
    pygame.display.set_caption(
        f"robot_sim v{__version__} | Exact:{'ON' if exact_on else 'OFF'} (E) | "
        f"Wave:{'ON' if wave_on else 'OFF'} (W)"
    )
    pygame.display.flip()

pygame.quit(); sys.exit()

