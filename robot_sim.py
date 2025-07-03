"""
robot_sim.py  – 2-D marionette with live sliders
================================================
• edit string lengths, limb lengths, and anchor positions in real time
• small velocity damping reduces jitter
"""

import sys, pygame, pymunk, pygame_gui
from   pymunk.pygame_util import DrawOptions

WIDTH, HEIGHT = 1000, 800
PPM           = 300
FPS           = 60
DT            = 1.0 / FPS
GRAVITY       = (0, -9.81)
DAMPING       = 0.9          # 1/s linear damping

pygame.init()
screen  = pygame.display.set_mode((WIDTH, HEIGHT))
clock   = pygame.time.Clock()
manager = pygame_gui.UIManager((WIDTH, HEIGHT))
space   = pymunk.Space()
space.gravity = GRAVITY
STATIC  = space.static_body

def to_px(v): return int(v.x*PPM + WIDTH/2), int(HEIGHT/2 - v.y*PPM)

# ------------------------------------------------------------------
# Anchors (Vec2d)
# ------------------------------------------------------------------
A = {
    "head" : pymunk.Vec2d( 0.00,  0.00),
    "handL": pymunk.Vec2d(-0.30,  0.00),
    "handR": pymunk.Vec2d( 0.30,  0.00),
    "footL": pymunk.Vec2d(-0.10,  0.00),
    "footR": pymunk.Vec2d( 0.10,  0.00)
}

# ------------------------------------------------------------------
# Helper to create a (damped) joint body
# ------------------------------------------------------------------
def make_joint(pos):
    body = pymunk.Body(0.02, pymunk.moment_for_circle(0.02, 0, 0.015))
    body.position = pos
    body.velocity_func = lambda b,g,d,dt: pymunk.Body.update_velocity(b, g, DAMPING, dt)
    shape = pymunk.Circle(body, 0.015)
    shape.filter = pymunk.ShapeFilter(group=1)
    space.add(body, shape)
    return body

# Joints
j = { "head"  : make_joint(( 0.00, -0.10)),
      "should": make_joint(( 0.00, -0.30)),
      "handL" : make_joint((-0.20, -0.50)),
      "handR" : make_joint(( 0.20, -0.50)),
      "hip"   : make_joint(( 0.00, -0.90)),
      "footL" : make_joint((-0.10, -1.20)),
      "footR" : make_joint(( 0.10, -1.20)) }

# Pin-joint limbs
limbs = {}
def add_limb(name,a,b):
    pj = pymunk.PinJoint(j[a], j[b]); space.add(pj); limbs[name]=pj
for a,b in [("head","should"),("should","handL"),("should","handR"),
            ("should","hip"),("hip","footL"),("hip","footR")]:
    add_limb(f"{a}-{b}", a, b)

# Slide-joint strings
strings={}
def add_string(key,L):
    sj=pymunk.SlideJoint(STATIC,j[key],A[key],(0,0),0.0,L)
    sj.collide_bodies=False; space.add(sj); strings[key]=sj
for k,L in {"head":0.35,"handL":0.50,"handR":0.50,"footL":0.70,"footR":0.70}.items():
    add_string(k,L)

# ------------------------------------------------------------------
# GUI sliders
# ------------------------------------------------------------------
SL_W = 260
x0,y = 10,10
ui={}
def slider(title,start,rng):
    global y
    ui[title]=pygame_gui.elements.UIHorizontalSlider(
        pygame.Rect((x0,y),(SL_W,20)),start,rng,manager)
    pygame_gui.elements.UILabel(
        pygame.Rect((x0+SL_W+4,y),(110,20)),title,manager)
    y+=24

# string length sliders
for k in A:        slider(f"len {k}",  strings[k].max, (0.0,2.0))
y+=10
# limb length sliders
for name,pj in limbs.items():
    base=pj.distance; slider(f"limb {name}", base, (base*0.3, base*2.0))
y+=10
# anchor X/Y sliders
for k in A:
    slider(f"X {k}",A[k].x,(-0.5,0.5))
    slider(f"Y {k}",A[k].y,(-0.5,0.5))

# ------------------------------------------------------------------
# Draw helpers
# ------------------------------------------------------------------
SEG=[("head","should"),("should","handL"),("should","handR"),
     ("should","hip"),("hip","footL"),("hip","footR")]
def draw():
    for a,b in SEG:
        pygame.draw.line(screen,(0,180,0),to_px(j[a].position),to_px(j[b].position),5)
    for k,s in strings.items():
        pygame.draw.line(screen,(200,0,0),to_px(A[k]),to_px(j[k].position),2)

# ------------------------------------------------------------------
# Main loop
# ------------------------------------------------------------------
running=True
while running:
    dt=clock.tick(FPS)/1000
    for e in pygame.event.get():
        if e.type==pygame.QUIT: running=False
        manager.process_events(e)
    manager.update(dt)

    # apply slider values ----------------------------------------------------
    for title,s in ui.items():
        v=s.get_current_value()
        if title.startswith("len "):
            strings[title[4:]].max=v
        elif title.startswith("limb "):
            limbs[title[5:]].distance=v
        elif title.startswith("X "):
            key=title[2:]; old=A[key]; new=pymunk.Vec2d(v,old.y)
            A[key]=new; strings[key].anchor_a=new
        elif title.startswith("Y "):
            key=title[2:]; old=A[key]; new=pymunk.Vec2d(old.x,v)
            A[key]=new; strings[key].anchor_a=new

    space.step(DT)

    screen.fill((240,240,240)); draw(); manager.draw_ui(screen)
    pygame.display.flip()

pygame.quit(); sys.exit()

