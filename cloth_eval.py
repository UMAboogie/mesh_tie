#Lint as: python3
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
"""Functions to build evaluation metrics for cloth data."""

# import tensorflow.compat.v1 as tf
import torch
from mesh_tie import common

device = torch.device('cuda')


def _rollout(model, initial_state, num_steps, target_world_pos):
    """Rolls out a model trajectory."""
    node_type = initial_state['node_type']
    mask = torch.eq(node_type[:, 0], torch.tensor([common.NodeType.NORMAL.value], device=device))
    mask = torch.stack((mask, mask, mask), dim=1)

    o_mask = torch.eq(node_type[:, 0], torch.tensor([common.NodeType.OBSTACLE.value], device=device))
    o_mask = torch.stack((o_mask, o_mask, o_mask), dim=1)

    def step_fn(prev_pos, cur_pos, trajectory, target_world_pos):
        # memory_prev = torch.cuda.memory_allocated(device) / (1024 * 1024)
        with torch.no_grad():
            prediction = model({**initial_state, 'prev_world_pos': prev_pos, 'world_pos': cur_pos}, is_training=False)

        next_pos = torch.where(mask, torch.squeeze(prediction), torch.squeeze(cur_pos))
        #print(torch.mean((cur_pos - next_pos)**2))
        next_pos = torch.where(o_mask, torch.squeeze(target_world_pos), next_pos)

        trajectory.append(cur_pos)
        return cur_pos, next_pos, trajectory

    prev_pos = torch.squeeze(initial_state['prev_world_pos'], 0)
    cur_pos = torch.squeeze(initial_state['world_pos'], 0)
    trajectory = []
    for step in range(num_steps):
        prev_pos, cur_pos, trajectory = step_fn(prev_pos, cur_pos, trajectory, target_world_pos[step])
    return torch.stack(trajectory)


def evaluate(model, trajectory, num_steps=None):
    """Performs model rollouts and create stats."""
    traj_squeeze = {k: torch.squeeze(v, 0) for k, v in trajectory.items()}
    initial_state = {k: torch.squeeze(v, 0)[0] for k, v in trajectory.items()}

    if num_steps is None:
        num_steps = traj_squeeze['cells'].shape[0]
    prediction = _rollout(model, initial_state, num_steps, traj_squeeze['target_world_pos'])

    error = torch.mean((prediction - traj_squeeze['world_pos'])**2, axis=-1).cpu()
    #print(error.shape)
    scalars = {'mse_%d_steps' % horizon: torch.mean(error[1:horizon+1])
                for horizon in [1, 10, 20, 50, 100, 200, len(error)-1]}

    traj_ops = {
        'faces': traj_squeeze['cells'],
        'mesh_pos': traj_squeeze['mesh_pos'],
        'gt_pos': traj_squeeze['world_pos'],
        'pred_pos': prediction
    }
    return scalars, traj_ops, error