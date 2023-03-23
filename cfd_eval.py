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
"""Functions to build evaluation metrics for CFD data."""
import torch

from mesh_tie import common

device = torch.device('cuda')


def _rollout(model, initial_state, num_steps):
    """Rolls out a model trajectory."""
    node_type = initial_state['node_type']
    mask = torch.logical_or(torch.eq(node_type[:, 0], torch.tensor([common.NodeType.NORMAL.value], device=device)),
                            torch.eq(node_type[:, 0], torch.tensor([common.NodeType.OUTFLOW.value], device=device)))
    mask = torch.stack((mask, mask), dim=1)

    def step_fn(velocity, trajectory):
        with torch.no_grad():
            prediction = model({**initial_state,
                                'velocity': velocity}, is_training=False)
            #print(torch.mean((prediction - velocity)**2))
        # don't update boundary nodes

        next_velocity = torch.where(mask, torch.squeeze(prediction), torch.squeeze(velocity))
        trajectory.append(velocity)
        #print(torch.mean((next_velocity - velocity)**2))
        return next_velocity, trajectory

    velocity = torch.squeeze(initial_state['velocity'], 0)
    trajectory = []
    for step in range(num_steps):
        velocity, trajectory = step_fn(velocity, trajectory)
    return torch.stack(trajectory)


def evaluate(model, trajectory): #traj: {1*598*N*2}
    """Performs model rollouts and create stats."""
    traj_squeeze = {k: torch.squeeze(v, 0) for k, v in trajectory.items()} #598*N*2
    #print(traj_squeeze['node_type'].shape)
    initial_state = {k: torch.squeeze(v, 0)[0] for k, v in trajectory.items()}
    num_steps = traj_squeeze['cells'].shape[0]
    prediction = _rollout(model, initial_state, num_steps)

    error = torch.mean((prediction - traj_squeeze['velocity'])**2, axis=-1).cpu()
    scalars = {'mse_%d_steps' % horizon: torch.mean(error[1:horizon+1])
                for horizon in [1, 10, 20, 50, 100, 200, len(error)-1]}
    traj_ops = {
        'faces': traj_squeeze['cells'],
        'mesh_pos': traj_squeeze['mesh_pos'],
        'gt_velocity': traj_squeeze['velocity'],
        'pred_velocity': prediction
    }
    return scalars, traj_ops, error
