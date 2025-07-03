import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Stick figure: head, torso, arms, legs
# Positions: [shoulder], [hand_L], [hand_R], [foot_L], [foot_R]

def get_positions(string_lengths, fixed_points, body_lengths):
    # For simplicity, just plot fixed string lengths as lines for now.
    # Later: Solve for positions with inverse kinematics.
    points = []
    for i in range(5):
        # String goes straight down from fixed_point to the stick figure joint
        p = np.array(fixed_points[i]) + np.array([0, -string_lengths[i]])
        points.append(p)
    return points

# Control bar coordinates
fixed_points = [
    [0, 0],    # Shoulder (head string)
    [-0.3, 0], # Left hand
    [0.3, 0],  # Right hand
    [-0.2, 0], # Left foot
    [0.2, 0],  # Right foot
]
# Initial string lengths
string_lengths = [1.0, 1.3, 1.3, 1.7, 1.7]
body_lengths = {
    "torso": 0.5,
    "arm": 0.4,
    "leg": 0.5
}

fig, ax = plt.subplots()
ax.set_xlim(-1, 1)
ax.set_ylim(-2, 0.2)
lines = []

def update(frame):
    global string_lengths
    ax.clear()
    ax.set_xlim(-1, 1)
    ax.set_ylim(-2, 0.2)

    # Animate: sway left/right
    sway = 0.1 * np.sin(frame / 10)
    string_lengths[1] = 1.3 + sway
    string_lengths[2] = 1.3 - sway

    points = get_positions(string_lengths, fixed_points, body_lengths)
    # Draw control bar
    xs, ys = zip(*fixed_points)
    ax.plot(xs, ys, 'ko-', lw=2)
    # Draw strings
    for i, p in enumerate(points):
        ax.plot([fixed_points[i][0], p[0]], [fixed_points[i][1], p[1]], 'r--')
        ax.plot(p[0], p[1], 'bo')
    # Draw stick figure: torso, arms, legs (simplified)
    ax.plot([points[0][0], points[3][0]], [points[0][1], points[3][1]], 'g-')  # Left leg
    ax.plot([points[0][0], points[4][0]], [points[0][1], points[4][1]], 'g-')  # Right leg
    ax.plot([points[0][0], points[1][0]], [points[0][1], points[1][1]], 'b-')  # Left arm
    ax.plot([points[0][0], points[2][0]], [points[0][1], points[2][1]], 'b-')  # Right arm

ani = FuncAnimation(fig, update, frames=200, interval=50)
plt.show()

