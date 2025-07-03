"""
robot_sim.py – 2-D servo-marionette using pymunk + pygame
---------------------------------------------------------
• Green segments = perfectly rigid limbs (PinJoint)
• Red dashed     = strings whose .max length you vary each frame
"""
import math, sys, pygame, pymunk
from   pymunk.pygame_util import DrawOptions

# ------------------------------------------------------------------
# Window / physics constants
# ------------------------------------------------------------------
WIDTH, HEIGHT = 700, 800          # pixels
PPM           = 300               # pixels per metre
FPS           = 60
DT            = 1.0 / FPS
GRAVITY       = (0, -9.81)        # Chipmunk +Y up

# ------------------------------------------------------------------
# Pygame & pymunk setup
# ------------------------------------------------------------------
pygame.init()
screen  = pygame.display.set_mode((WIDTH, HEIGHT))
clock   = pygame.time.Clock()
options = DrawOptions(screen)

space = pymunk.Space()
space.gravity = GRAVITY
static_body   = space.static_body    # needed for static anchors

def to_px(p):
    """Chipmunk coords → pygame pixels (origin centre, +Y up)."""
    return int(p.x * PPM + WIDTH / 2), int(HEIGHT / 2 - p.y * PPM)

# ------------------------------------------------------------------
# Static “control-bar” anchor points  (now Vec2d, not tuples!)
# ------------------------------------------------------------------
anchor_world = {
    "head" : pymunk.Vec2d( 0.00,  0.00),
    "handL": pymunk.Vec2d(-0.30,  0.00),
    "handR": pymunk.Vec2d( 0.30,  0.00),
    "footL": pymunk.Vec2d(-0.10,  0.00),
    "footR": pymunk.Vec2d( 0.10,  0.00),
}

# ------------------------------------------------------------------
# Helper to create a circular body for each joint
# ------------------------------------------------------------------
def make_joint(pos, mass=0.02, radius=0.015):
    moment = pymunk.moment_for_circle(mass, 0, radius)
    body   = pymunk.Body(mass, moment)
    body.position = pos
    shape  = pymunk.Circle(body, radius)
    shape.filter = pymunk.ShapeFilter(group=1)   # prevent self-collision
    space.add(body, shape)
    return body

# ------------------------------------------------------------------
# Build stick-figure joints
# ------------------------------------------------------------------
j = {}   # shorthand dict
j["head"]   = make_joint(( 0.00, -0.10))
j["should"] = make_joint(( 0.00, -0.30))
j["handL"]  = make_joint((-0.20, -0.50))
j["handR"]  = make_joint(( 0.20, -0.50))
j["hip"]    = make_joint(( 0.00, -0.90))
j["footL"]  = make_joint((-0.10, -1.20))
j["footR"]  = make_joint(( 0.10, -1.20))

# Rigid limbs (PinJoint keeps exact distance)
for a, b in [("head","should"), ("should","handL"), ("should","handR"),
             ("should","hip"),   ("hip","footL"),  ("hip","footR")]:
    space.add(pymunk.PinJoint(j[a], j[b]))

# ------------------------------------------------------------------
# Strings implemented as SlideJoint (min=0, max=L)
# We keep handles so we can change .max each frame like servo spools.
# ------------------------------------------------------------------
strings = {}
def add_string(name, anchor_key, initial_len):
    sj = pymunk.SlideJoint(static_body, j[name],
                           anchor_world[anchor_key], (0,0),
                           0.0, initial_len)      # min, max
    sj.collide_bodies = False
    space.add(sj)
    strings[name] = sj

add_string("handL","handL", 0.50)
add_string("handR","handR", 0.50)
add_string("footL","footL", 0.70)
add_string("footR","footR", 0.70)
add_string("head" ,"head" , 0.35)

# ------------------------------------------------------------------
# Simple “wave both hands” demo by modulating two string lengths
# ------------------------------------------------------------------
def update_servos(t):
    strings["handL"].max = 0.50 + 0.05*math.sin(t*2.0)
    strings["handR"].max = 0.50 + 0.05*math.cos(t*2.6)
    # To animate feet or head, adjust their .max similarly.

# ------------------------------------------------------------------
# Main loop
# ------------------------------------------------------------------
running, t = True, 0.0
SEGMENTS = [("head","should"), ("should","handL"), ("should","handR"),
            ("should","hip"),  ("hip","footL"),    ("hip","footR")]

while running:
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            running = False

    update_servos(t)
    space.step(DT)
    t += DT

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------
    screen.fill((240,240,240))

    # Draw rigid limbs (green)
    for a, b in SEGMENTS:
        pygame.draw.line(screen, (0,180,0), to_px(j[a].position),
                                           to_px(j[b].position), 5)
    # Draw strings (red dashed)
    for name, sj in strings.items():
        pygame.draw.line(screen, (200,0,0), to_px(anchor_world[name]),
                                            to_px(j[name].position), 2)

    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()
sys.exit()

