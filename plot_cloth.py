# Lint as: python3
# pylint: disable=g-bad-file-header
# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
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

import torch


FLAGS = flags.FLAGS
# flags.DEFINE_string('rollout_path', 'E:\\meshgraphnets\\output\\flag_simple\\Tue-Jan-25-19-21-14-2022\\1\\rollout\\rollout.pkl', 'Path to rollout pickle file')
flags.DEFINE_string('rollout_dir', None, 'Path to rollout pickle file')
flags.DEFINE_string('gif_dir', None, 'dir name of animation')
flags.DEFINE_string('type', 'flag', 'simulation type')


def main(unused_argv):
    print("Ploting run")
    with open(FLAGS.rollout_dir + "/rollout.pkl", 'rb') as fp:
        rollout_data = pickle.load(fp)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
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
        ax.cla()
        bound = bounds[traj]

        ax.set_xlim([bound[0][0], bound[1][0]])
        ax.set_ylim([bound[0][1], bound[1][1]])
        ax.set_zlim([bound[0][2], bound[1][2]])

        pos = (rollout_data[traj]['pred_pos'])[step].to('cpu')
        original_pos = (rollout_data[traj]['gt_pos'])[step].to('cpu')
        # print(pos[10])
        faces = (rollout_data[traj]['faces'])[step].to('cpu')
        ax.plot_trisurf(pos[:, 0], pos[:, 1], faces, pos[:, 2], shade=True)
        ax.plot_trisurf(original_pos[:, 0], original_pos[:, 1], faces, original_pos[:, 2], shade=True, alpha=0.3)
        ax.set_title('Trajectory %d Step %d' % (traj, step))
        return fig,

    anima = animation.FuncAnimation(fig, animate, frames=num_frames, interval=100)
    # writervideo = animation.FFMpegWriter(fps=30)
    # anima.save(os.path.join(save_path, 'ani.mp4'), writer=writervideo)
    if FLAGS.gif_dir != None:
        os.makedirs(FLAGS.gif_dir, exist_ok=True)
        dt_now = datetime.datetime.now()
        file_name = dt_now.strftime('%y%m%d%H%M') + '_' + FLAGS.type + '.gif'
        anima.save(os.path.join(FLAGS.gif_dir, file_name), writer="imagemagick")
    else:
        plt.show(block=True)


if __name__ == '__main__':
    app.run(main)
