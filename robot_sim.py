import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Stick figure: Head, Shoulder, Hand_L, Hand_R, Hip, Foot_L, Foot_R

# Initial positions (y points downward)
points = np.array([
    [0, -0.1],   # Head
    [0, -0.3],   # Shoulder
    [-0.2, -0.5], # Hand_L
    [0.2, -0.5],  # Hand_R
    [0, -0.9],   # Hip
    [-0.1, -1.2], # Foot_L
    [0.1, -1.2],  # Foot_R
])

prev_points = points.copy()

# Limb constraints (pairs of indices, fixed length)
bones = [
    (0, 1),  # Head-Shoulder
    (1, 2),  # Shoulder-Hand_L
    (1, 3),  # Shoulder-Hand_R
    (1, 4),  # Shoulder-Hip
    (4, 5),  # Hip-Foot_L
    (4, 6),  # Hip-Foot_R
]

# String attachments: (point index, fixed point)
strings = [
    (2, [-0.3, 0.0]),  # Hand_L
    (3, [0.3, 0.0]),   # Hand_R
    (5, [-0.1, 0.0]),  # Foot_L
    (6, [0.1, 0.0]),   # Foot_R
    (0, [0.0, 0.0]),   # Head
]

bone_lengths = [np.linalg.norm(points[a] - points[b]) for a, b in bones]

dt = 0.03
g = 1.5  # gravity

def verlet(points, prev_points, dt):
    return points + (points - prev_points) + np.array([0, g * dt**2])

def apply_string_constraints(points, strings, string_lengths):
    for i, (pi, anchor) in enumerate(strings):
        vec = points[pi] - anchor
        dist = np.linalg.norm(vec)
        max_len = string_lengths[i]
        if dist > max_len:
            direction = vec / dist
            points[pi] = anchor + direction * max_len

def apply_bone_constraints(points, bones, bone_lengths):
    for i, (a, b) in enumerate(bones):
        pa, pb = points[a], points[b]
        delta = pb - pa
        dist = np.linalg.norm(delta)
        if dist == 0: continue
        diff = (dist - bone_lengths[i]) / 2
        correction = delta / dist * diff
        # Move both ends (unless string-attached)
        points[a] += correction
        points[b] -= correction

# Initial string lengths: measured from anchors to attachment points
string_lengths = [np.linalg.norm(points[pi] - anchor) for pi, anchor in strings]

fig, ax = plt.subplots()
ax.set_xlim(-0.5, 0.5)
ax.set_ylim(-1.5, 0.2)

def update(frame):
    global points, prev_points, string_lengths

    # Animate: wave hands
    string_lengths[0] = 0.5 + 0.05 * np.sin(frame / 10)  # Left hand
    string_lengths[1] = 0.5 + 0.05 * np.cos(frame / 13)  # Right hand

    temp = points.copy()
    points = verlet(points, prev_points, dt)
    prev_points = temp

    # Apply string constraints (limit how far each joint can fall)
    apply_string_constraints(points, strings, string_lengths)

    # Satisfy limb lengths (several passes)
    for _ in range(6):
        apply_bone_constraints(points, bones, bone_lengths)
        apply_string_constraints(points, strings, string_lengths)

    ax.clear()
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-1.5, 0.2)
    # Draw strings
    for i, (pi, anchor) in enumerate(strings):
        ax.plot([anchor[0], points[pi][0]], [anchor[1], points[pi][1]], 'r--')
        ax.plot(anchor[0], anchor[1], 'ko')
    # Draw bones
    for a, b in bones:
        ax.plot([points[a][0], points[b][0]], [points[a][1], points[b][1]], 'g-', lw=2)
    # Draw joints
    ax.plot(points[:,0], points[:,1], 'bo')

ani = FuncAnimation(fig, update, frames=300, interval=40)
plt.show()

