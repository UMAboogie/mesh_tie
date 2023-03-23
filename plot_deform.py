"""Plots a cloth trajectory rollout."""
import os

import pickle

import pathlib

from absl import app
from absl import flags

from matplotlib import animation
import matplotlib.pyplot as plt

import math
import datetime

import numpy as np
import mpl_toolkits.mplot3d as p3d

import torch

FLAGS = flags.FLAGS
# flags.DEFINE_string('rollout_path', 'E:\\meshgraphnets\\output\\flag_simple\\Tue-Jan-25-19-21-14-2022\\1\\rollout\\rollout.pkl', 'Path to rollout pickle file')
flags.DEFINE_string('rollout_dir', None, 'Path to rollout pickle file')
flags.DEFINE_string('gif_dir', None, 'dir name of animation')

def main(unused_argv):
    with open(FLAGS.rollout_dir + "/rollout.pkl", 'rb') as fp:
        rollout_data = pickle.load(fp)
    fig = plt.figure(figsize=(19.2, 10.8))
    ax_origin = fig.add_subplot(121, projection='3d')
    ax_pred = fig.add_subplot(122, projection='3d')

    skip = 10
    num_steps = rollout_data[0]['gt_pos'].shape[0]
    # print(num_steps)
    num_frames = len(rollout_data) * num_steps // skip


    # compute bounds
    bounds = []
    for trajectory in rollout_data:
        # print("bb_min shape", trajectory['gt_pos'].shape)
        bb_min = trajectory['gt_pos'].cpu().numpy().min(axis=(0, 1))
        bb_max = trajectory['gt_pos'].cpu().numpy().max(axis=(0, 1))
        bounds.append((bb_min, bb_max))

    def animate(num):
        traj = (num * skip) // num_steps
        step = (num * skip) % num_steps
        ax_origin.cla()
        ax_pred.cla()
        bound = bounds[traj]

        ax_origin.set_xlim([bound[0][0], bound[1][0]])
        ax_origin.set_ylim([bound[0][1], bound[1][1]])
        ax_origin.set_zlim([bound[0][2], bound[1][2]])

        ax_pred.set_xlim([bound[0][0], bound[1][0]])
        ax_pred.set_ylim([bound[0][1], bound[1][1]])
        ax_pred.set_zlim([bound[0][2], bound[1][2]])


        pos = rollout_data[traj]['pred_pos'][step].to('cpu')
        original_pos = rollout_data[traj]['gt_pos'][step].to('cpu')

        faces = rollout_data[traj]['faces'][step].to('cpu')

        ax_origin.plot_trisurf(original_pos[:, 0], original_pos[:, 1], faces, original_pos[:, 2], shade=True, alpha=0.3)
        ax_pred.plot_trisurf(pos[:, 0], pos[:, 1], faces, pos[:, 2], shade=True, alpha=0.3)

        ax_origin.set_title('ORIGIN Trajectory %d Step %d' % (traj, step))
        ax_pred.set_title('PRED Trajectory %d Step %d' % (traj, step))
        return fig,

    anima = animation.FuncAnimation(fig, animate, frames=num_frames, interval=100)
    # writervideo = animation.FFMpegWriter(fps=30)
    # anima.save(os.path.join(save_path, 'ani.mp4'), writer=writervideo)
    if FLAGS.gif_dir != None:
        os.makedirs(FLAGS.gif_dir, exist_ok=True)
        dt_now = datetime.datetime.now()
        file_name = dt_now.strftime('%y%m%d%H%M') + '_deform.gif'
        anima.save(os.path.join(FLAGS.gif_dir, file_name), writer="imagemagick")
    else:
        plt.show(block=True)


if __name__ == '__main__':
    app.run(main)
