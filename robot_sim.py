#!/usr/bin/env python3
"""
robot_sim.py  – 2-D marionette simulator (pymunk + pygame_gui)
Version 1.1.0
-------------------------------------------------------------
• Left arm waves; exact-mode (E) freezes all other strings.
• Sliders:
      – 5× string length
      – 6× limb length
      – 5× anchor X position
"""

__version__ = "1.1.0"

import math, sys, pygame, pymunk, pygame_gui

# ----------------------------------------------------------------------
# Basic constants
# ----------------------------------------------------------------------
WIDTH, HEIGHT, PPM = 1000, 800, 300
FPS, DT            = 60, 1/60
GRAVITY            = (0, -9.81)
DAMPING            = 1.0             # linear velocity damping 1/s
LEFT_WAVE_A        = 0.05            # metres
LEFT_WAVE_F        = 2.0             # Hz

to_px = lambda v: (int(v.x*PPM + WIDTH/2),
                   int(HEIGHT/2 - v.y*PPM))

# ----------------------------------------------------------------------
# Pygame / Pymunk setup
# ----------------------------------------------------------------------
pygame.init()
screen  = pygame.display.set_mode((WIDTH, HEIGHT))
clock   = pygame.time.Clock()
manager = pygame_gui.UIManager((WIDTH, HEIGHT))

space   = pymunk.Space()
space.gravity = GRAVITY
STATIC  = space.static_body

# ----------------------------------------------------------------------
# Anchors (Vec2d)
# ----------------------------------------------------------------------
A = {
    "head" : pymunk.Vec2d( 0.00, 0.00),
    "handL": pymunk.Vec2d(-0.30, 0.00),
    "handR": pymunk.Vec2d( 0.30, 0.00),
    "footL": pymunk.Vec2d(-0.10, 0.00),
    "footR": pymunk.Vec2d( 0.10, 0.00)
}

# ----------------------------------------------------------------------
# Helper to build a damped joint body
# ----------------------------------------------------------------------
def make_joint(pos):
    m, r = 0.02, 0.015
    body = pymunk.Body(m, pymunk.moment_for_circle(m, 0, r))
    body.position = pos
    body.velocity_func = (
        lambda b, g, d, dt: pymunk.Body.update_velocity(b, g, DAMPING, dt)
    )
    shape = pymunk.Circle(body, r)
    shape.filter = pymunk.ShapeFilter(group=1)   # self-collision off
    space.add(body, shape)
    return body

# ----------------------------------------------------------------------
# Build skeleton joints
# ----------------------------------------------------------------------
j = {
    "head"  : make_joint(( 0.00, -0.10)),
    "should": make_joint(( 0.00, -0.30)),
    "handL" : make_joint((-0.20, -0.50)),
    "handR" : make_joint(( 0.20, -0.50)),
    "hip"   : make_joint(( 0.00, -0.90)),
    "footL" : make_joint((-0.10, -1.20)),
    "footR" : make_joint(( 0.10, -1.20))
}

# Limbs (PinJoint)
limbs = {}
def pin(name, a, b):
    pj = pymunk.PinJoint(j[a], j[b])
    space.add(pj)
    limbs[name] = pj

for a, b in [("head","should"), ("should","handL"), ("should","handR"),
             ("should","hip"),   ("hip","footL"),    ("hip","footR")]:
    pin(f"{a}-{b}", a, b)

# Strings (SlideJoint)
strings = {}
def sld(key, length):
    sj = pymunk.SlideJoint(STATIC, j[key], A[key], (0, 0), 0.0, length)
    sj.collide_bodies = False
    space.add(sj)
    strings[key] = sj

for k, L in {"head":0.35, "handL":0.50, "handR":0.50,
             "footL":0.70, "footR":0.70}.items():
    sld(k, L)

# ----------------------------------------------------------------------
# GUI sliders (string length, limb length, anchor X only)
# ----------------------------------------------------------------------
SL_W, x0, y = 260, 10, 10
ui = {}
def slider(label, start, rng):
    global y
    ui[label] = pygame_gui.elements.UIHorizontalSlider(
        pygame.Rect((x0, y), (SL_W, 20)), start, rng, manager
    )
    pygame_gui.elements.UILabel(
        pygame.Rect((x0 + SL_W + 6, y), (150, 20)), label, manager
    )
    y += 24

for k in A:                   slider(f"len {k}", strings[k].max, (0.0, 2.0))
y += 8
for name, pj in limbs.items():
    base = pj.distance
    slider(f"limb {name}", base, (base * 0.3, base * 2.0))
y += 8
for k in A:                   slider(f"X {k}", A[k].x, (-0.5, 0.5))

# ----------------------------------------------------------------------
# Exact-mode toggle  (freeze all but left hand)
# ----------------------------------------------------------------------
frozen_keys = ["head", "handR", "footL", "footR"]
exact_mode  = True
def lock_frozen():
    for k in frozen_keys:
        strings[k].max = A[k].get_distance(j[k].position)

# ----------------------------------------------------------------------
# Drawing helper
# ----------------------------------------------------------------------
SEG = [("head","should"), ("should","handL"), ("should","handR"),
       ("should","hip"),  ("hip","footL"),    ("hip","footR")]
def draw_scene():
    for a, b in SEG:
        pygame.draw.line(screen, (0, 180, 0),
                         to_px(j[a].position), to_px(j[b].position), 5)
    for k in A:
        pygame.draw.line(screen, (200, 0, 0),
                         to_px(A[k]), to_px(j[k].position), 2)

# ----------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------
t = 0.0
running = True
while running:
    dt = clock.tick(FPS) / 1000.0
    t += dt

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_e:
            exact_mode = not exact_mode
        manager.process_events(event)

    manager.update(dt)

    # Left-hand waving motion
    strings["handL"].max = 0.50 + LEFT_WAVE_A * math.sin(2 * math.pi * LEFT_WAVE_F * t)

    # Apply GUI slider values
    for label, s in ui.items():
        v = s.get_current_value()
        if label.startswith("len "):
            strings[label[4:]].max = v
        elif label.startswith("limb "):
            limbs[label[5:]].distance = v
        elif label.startswith("X "):
            key = label[2:]
            old = A[key]
            new = pymunk.Vec2d(v, old.y)
            A[key] = new
            strings[key].anchor_a = new

    if exact_mode:
        lock_frozen()

    space.step(DT)

    screen.fill((240, 240, 240))
    draw_scene()
    manager.draw_ui(screen)
    pygame.display.set_caption(
        f"robot_sim v{__version__}  |  Exact-mode: {'ON' if exact_mode else 'OFF'}  (E toggles)"
    )
    pygame.display.flip()

pygame.quit()
sys.exit()

