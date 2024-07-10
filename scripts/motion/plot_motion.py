#!/usr/bin/env python3

"""Plots the motions in a file, might require some adjusting of the shown range and angle.

Original: One file with a motion sequences as position as list of list [[],[],...]

in_path: Path to the motion sequence file
"""

from matplotlib.animation import FuncAnimation
from matplotlib import pyplot as plt
import sys
import json


def plot_original_poses(poses):
    reduced_poses = []

    if not isinstance(poses[0], list):
        poses = [poses]

    for i in range(len(poses)):
        new_pose = [[], [], []]

        for ii in range(len(poses[i]) // 3):
            new_pose[0].append(poses[i][ii * 3])
            new_pose[1].append(poses[i][ii * 3 + 1])
            new_pose[2].append(poses[i][ii * 3 + 2])

        reduced_poses.append(new_pose)

    poses = reduced_poses

    x_values = poses[0][0]
    y_values = poses[0][1]
    z_values = poses[0][2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(0, -45)
    ax.set_xlim3d(300, 1800)
    ax.set_ylim3d(-750, 750)
    ax.set_zlim3d(0, 1500)
    title = ax.set_title("Runner")
    graph = ax.scatter(x_values, y_values, z_values)

    def update_graph(num):
        foo = [poses[num][0], poses[num][1], poses[num][2]]
        graph._offsets3d = foo
        title.set_text("pose={}".format(num))

    ani = FuncAnimation(fig, update_graph, len(poses), interval=30, blit=False, repeat=True)
    plt.show()
    # ani.save('/home/simon/DuD/behavioral_privacy/datasets/gifs/runner.gif', writer='imagemagick', fps=30)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        exit(1)

    walker_file = sys.argv[1]
    with open(walker_file, "r") as f:
        walker_data = json.load(f)
        plot_original_poses(walker_data)
