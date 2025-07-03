"""
robot_sim.py  –  2-D servo-marionette with live sliders
-------------------------------------------------------
• Pymunk physics keeps limbs rigid (PinJoint) and strings limited (SlideJoint)
• pygame draws the figure
• pygame_gui provides on-screen sliders to change:
      – 5 string max lengths
      – 6 limb (PinJoint) lengths
"""

import math, sys, pygame, pymunk, pygame_gui
from   pymunk.pygame_util import DrawOptions

# --------------------------------------------------------------------------
# Window & physics constants
# --------------------------------------------------------------------------
WIDTH, HEIGHT = 900, 800            # pixels
PPM           = 300                 # pixels per metre
FPS           = 60
DT            = 1.0 / FPS
GRAVITY       = (0, -9.81)          # Chipmunk +Y up → negative is down

# --------------------------------------------------------------------------
# Pygame / pymunk / pygame_gui setup
# --------------------------------------------------------------------------
pygame.init()
screen   = pygame.display.set_mode((WIDTH, HEIGHT))
clock    = pygame.time.Clock()
options  = DrawOptions(screen)

manager  = pygame_gui.UIManager((WIDTH, HEIGHT))   # UI manager

space = pymunk.Space()
space.gravity = GRAVITY
STATIC = space.static_body

def to_px(vec):
    """Chipmunk world-coords → Pygame screen-coords (origin centre)."""
    return int(vec.x * PPM + WIDTH/2), int(HEIGHT/2 - vec.y * PPM)

# --------------------------------------------------------------------------
# Static “control-bar” anchors (Vec2d, not tuples!)
# --------------------------------------------------------------------------
A = {               # anchors in metres
    "head" : pymunk.Vec2d( 0.00,  0.00),
    "handL": pymunk.Vec2d(-0.30,  0.00),
    "handR": pymunk.Vec2d( 0.30,  0.00),
    "footL": pymunk.Vec2d(-0.10,  0.00),
    "footR": pymunk.Vec2d( 0.10,  0.00)
}

# --------------------------------------------------------------------------
# Helper: small circle body for each joint
# --------------------------------------------------------------------------
def make_joint(pos, mass=0.02, radius=0.015):
    mo = pymunk.moment_for_circle(mass, 0, radius)
    body = pymunk.Body(mass, mo)
    body.position = pos
    shape = pymunk.Circle(body, radius)
    shape.filter = pymunk.ShapeFilter(group=1)   # no self-collision
    space.add(body, shape)
    return body

# --------------------------------------------------------------------------
# Build stick-figure joints
# --------------------------------------------------------------------------
j = {}
j["head"]   = make_joint(( 0.00, -0.10))
j["should"] = make_joint(( 0.00, -0.30))
j["handL"]  = make_joint((-0.20, -0.50))
j["handR"]  = make_joint(( 0.20, -0.50))
j["hip"]    = make_joint(( 0.00, -0.90))
j["footL"]  = make_joint((-0.10, -1.20))
j["footR"]  = make_joint(( 0.10, -1.20))

# --------------------------------------------------------------------------
# Rigid limbs (PinJoint)  …save handles so we can edit .distance later
# --------------------------------------------------------------------------
limbs = {}
def add_limb(name, a, b):
    pj = pymunk.PinJoint(j[a], j[b])
    space.add(pj)
    limbs[name] = pj

add_limb("head-should",  "head",   "should")
add_limb("should-handL", "should", "handL")
add_limb("should-handR", "should", "handR")
add_limb("should-hip",   "should", "hip")
add_limb("hip-footL",    "hip",    "footL")
add_limb("hip-footR",    "hip",    "footR")

# --------------------------------------------------------------------------
# Strings (SlideJoint)  …handles kept in dict for live editing
# --------------------------------------------------------------------------
strings = {}
def add_string(name, joint_key, anchor_key, length):
    sj = pymunk.SlideJoint(STATIC, j[joint_key], A[anchor_key], (0,0), 0.0, length)
    sj.collide_bodies = False
    space.add(sj)
    strings[name] = sj

add_string("head",  "head",  "head",  0.35)
add_string("handL", "handL", "handL", 0.50)
add_string("handR", "handR", "handR", 0.50)
add_string("footL", "footL", "footL", 0.70)
add_string("footR", "footR", "footR", 0.70)

# --------------------------------------------------------------------------
# Build UI sliders
# --------------------------------------------------------------------------
slider_w = 220
pad      = 6
ui_x0    = 10
ui_y     = 10

ui_sliders = {}   # name → slider element

# 1) string sliders
for name in ["head", "handL", "handR", "footL", "footR"]:
    ui_sliders[name] = pygame_gui.elements.UIHorizontalSlider(
        relative_rect = pygame.Rect((ui_x0, ui_y), (slider_w, 20)),
        start_value   = strings[name].max,
        value_range   = (0.1, 1.0),
        manager       = manager
    )
    pygame_gui.elements.UILabel(pygame.Rect((ui_x0+slider_w+4, ui_y), (60,20)),
                                text=name, manager=manager)
    ui_y += 26

ui_y += 10  # blank line between groups

# 2) limb-length sliders
for name, pj in limbs.items():
    base_len = pj.distance
    ui_sliders[name] = pygame_gui.elements.UIHorizontalSlider(
        pygame.Rect((ui_x0, ui_y), (slider_w, 20)),
        start_value = base_len,
        value_range = (base_len*0.6, base_len*1.4),
        manager     = manager
    )
    pygame_gui.elements.UILabel(pygame.Rect((ui_x0+slider_w+4, ui_y), (90,20)),
                                text=name, manager=manager)
    ui_y += 26

# --------------------------------------------------------------------------
# Drawing helpers
# --------------------------------------------------------------------------
SEGMENTS = [("head","should"), ("should","handL"), ("should","handR"),
            ("should","hip"),  ("hip","footL"),   ("hip","footR")]

def draw_scene():
    # limbs (green)
    for a,b in SEGMENTS:
        pygame.draw.line(screen, (0,180,0), to_px(j[a].position),
                                           to_px(j[b].position), 5)
    # strings (red)
    for key, sj in strings.items():
        pygame.draw.line(screen, (200,0,0), to_px(A[key]),
                                            to_px(j[key].position), 2)

# --------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------
running = True
while running:
    time_delta = clock.tick(FPS)/1000.0

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        manager.process_events(event)

    manager.update(time_delta)

    # Update physics parameters from sliders
    for name, slider in ui_sliders.items():
        val = slider.get_current_value()
        if name in strings:
            strings[name].max = val
        else:                     # limb slider
            limbs[name].distance = val

    space.step(DT)

    # ----------------------------------------------------------------------
    # Render
    # ----------------------------------------------------------------------
    screen.fill((240,240,240))
    draw_scene()
    manager.draw_ui(screen)

    pygame.display.flip()

pygame.quit()
sys.exit()

