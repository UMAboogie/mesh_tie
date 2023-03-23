"""Runs the learner/evaluator."""
import sys
import os
import pathlib
from pathlib import Path

import pickle
from absl import app
from absl import flags
import h5py

import torch

from mesh_tie import core_model_1
from mesh_tie import cloth_model_1
from mesh_tie import cloth_eval
from mesh_tie import cfd_model_1
from mesh_tie import cfd_eval
from mesh_tie import deform_model_1
from mesh_tie import deform_eval
from mesh_tie import air_model
from mesh_tie import air_eval

from mesh_tie import common
from mesh_tie import dataset
import logging

import numpy as np
import json

import time
import datetime
import functools

import matplotlib

matplotlib.use('AGG')
import matplotlib.pyplot as plt

device = torch.device('cuda')

# train and evaluation configuration
FLAGS = flags.FLAGS
flags.DEFINE_enum('model', 'cloth', ['cloth', 'deform', 'air', 'cfd'],
                  'Select model to run.')
flags.DEFINE_enum('mode', 'all', ['train', 'eval', 'all'],
                  'Train model, or run evaluation, or run both.')
flags.DEFINE_enum('rollout_split', 'valid', ['train', 'test', 'valid'],
                  'Dataset split to use for rollouts.')
flags.DEFINE_string('checkpoint_dir', None, 'Directory to save checkpoint')
flags.DEFINE_string('dataset_dir', None, 'Directory to load dataset from.')
flags.DEFINE_string('rollout_dir', None, 'Pickle file to save eval trajctories')
flags.DEFINE_boolean('load_chk', False, 'whether to load checkpoints')

flags.DEFINE_integer('max_epochs', 4, 'No. of max training epochs')
#flags.DEFINE_integer('num_training_steps', int(10e6), 'No. of training steps')
flags.DEFINE_integer('trajectories', 4, 'No. of training trajectories')
flags.DEFINE_integer('num_rollouts', 1, 'No. of rollout trajectories')

flags.DEFINE_integer('message_passing_steps', 5, 'No. of message passing steps')


PARAMETERS = {
    'cfd': dict(traj_num=[500,50,50], noise=0.02, gamma=1.0, field='velocity', history=False,
                size=2, batch=2, model=cfd_model_1, evaluator=cfd_eval, loss_type='cfd'),
    'cloth': dict(traj_num=[500,50,50], noise=0.003, gamma=0.1, field='world_pos', history=True,
                  size=3, batch=1, model=cloth_model_1, evaluator=cloth_eval, loss_type='cloth'),
    'deform': dict(traj_num=[600,50,50], noise=0.003, gamma=1.0, field='world_pos', history=False,
                  size=3, batch=2, model=deform_model_1, evaluator=deform_eval, loss_type='deform'),
    'air': dict(traj_num=[500,50,50], noise=10.0, gamma=1.0, field='velocity', history=False,
                  size=3, batch=2, model=air_model, evaluator=air_eval)
}



def pickle_save(path, data):
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def pickle_load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def split_and_preprocess(trajectory_data, field, scale, gamma, add_noise_bool=True, split_bool=True):
    def add_noise(frame):
        zero_size = torch.zeros(frame[field].size(), dtype=torch.float32).to(device)
        noise = torch.normal(zero_size, std=scale).to(device)
        other = torch.Tensor([common.NodeType.NORMAL.value]).to(device)
        mask = torch.eq(frame['node_type'], other.int())[:, 0]
        mask_sequence = []
        for i in range(noise.shape[1]):
            mask_sequence.append(mask)
        mask = torch.stack(mask_sequence, dim=1)
        noise = torch.where(mask, noise, torch.zeros_like(noise))
        frame[field] += noise
        frame['target_' + field] += (1.0 - gamma) * noise
        return frame
    
    def element_operation(trajectory):
        trajectory_steps = []
        for i in range(len(trajectory["node_type"])):
            trajectory_step = {}
            for key, value in trajectory.items():
                trajectory_step[key] = value[i]
            if add_noise_bool:
                trajectory_step = add_noise(trajectory_step)
            trajectory_steps.append(trajectory_step)
        return trajectory_steps

        
    traj_tensor = {}
    for key, feature in trajectory_data.items():
        feature_tensor = torch.from_numpy(feature).to(device)
        traj_tensor[key] = feature_tensor
    if split_bool:
        traj_tensor = element_operation(traj_tensor)
            
    return traj_tensor

def load_dataset(path, split):
    fp = h5py.File(os.path.join(path, split + ".h5"), 'r')
    return fp

def squeeze_data_frame(data_frame):
    for key, value in data_frame.items():
        data_frame[key] = torch.squeeze(value,0)
    return data_frame

def logger_setup(log_path):
    # set log configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    # console_output_handler = logging.StreamHandler(sys.stdout)
    # console_output_handler.setLevel(logging.INFO)
    file_log_handler = logging.FileHandler(filename=log_path, mode='w', encoding='utf-8')
    file_log_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt='%(asctime)s - %(message)s')
    # console_output_handler.setFormatter(formatter)
    file_log_handler.setFormatter(formatter)
    # root_logger.addHandler(console_output_handler)
    root_logger.addHandler(file_log_handler)
    return root_logger

def log_run_summary(root_logger):
    root_logger.info("")
    root_logger.info("=======================Run Summary=======================")
    root_logger.info("Simulation task is " + FLAGS.model + " simulation")
    root_logger.info("Mode is " + FLAGS.mode)
    if FLAGS.mode == 'eval' or FLAGS.mode == 'all':
        root_logger.info("Evaluation set is " + FLAGS.rollout_split)
    else:
        root_logger.info("No Evaluation")
    root_logger.info(
        "Train and/or evaluation configuration are " + str(FLAGS.max_epochs) + " epochs, " + str(
            FLAGS.trajectories) + " trajectories each epoch, number of rollouts is " + str(
            FLAGS.num_rollouts))
    root_logger.info("=========================================================")
    root_logger.info("")


def learner(model, params, config):
    root_logger = logging.getLogger()
    """Run a learner job."""
    #ds = load_dataset(FLAGS.dataset_dir, 'train')
    ds = dataset.trajDataset(FLAGS.dataset_dir, 'train', transform=functools.partial(split_and_preprocess, field=params['field'], scale=params['noise'], gamma=params['gamma'], 
                                                                                                                                add_noise_bool=True, split_bool=True))
    data_loader = torch.utils.data.DataLoader(ds, batch_size=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.1 + 1e-6)
    trained_epoch = 0
    count = 0
    pass_count = 500
    if FLAGS.load_chk:
        optimizer.load_state_dict(config['optimizer'])
        scheduler.load_state_dict(config['scheduler'])
        epoch = config['epoch']
        count = config['count']
        trained_epoch = config['epoch'] + 1
        root_logger.info("Loaded optimizer, scheduler and model epoch checkpoint\n")
        pass_count = 500 - count if count < 500 else 0

    # model training
    is_training = True

    epoch_run_times = []
    all_trajectory_train_losses = []
    epoch_training_losses = []


    for epoch in range(FLAGS.max_epochs)[trained_epoch:]:
                    
        current_time = time.time()
        # every time when model.train is called, model will train itself with the whole dataset
        root_logger.info("Epoch " + str(epoch + 1) + "/" + str(FLAGS.max_epochs))
        epoch_training_loss = 0.0
        
        for trajectory_index, trajectory in enumerate(data_loader):
            root_logger.info(
                "    trajectory index " + str(trajectory_index + 1) + "/" + str(params['traj_num'][0]))
            #trajectory = ds[str(trajectory_index)]
            #trajectory = split_and_preprocess(trajectory, params['field'], params['noise'], params['gamma'], add_noise_bool=True)
            trajectory_loss = 0.0
            for data_frame_index, data_frame in enumerate(trajectory):
                count += 1
                data_frame = squeeze_data_frame(data_frame)
                loss = model.loss(data_frame)
                if count % 1000 == 0:
                    root_logger.info("    1000 step loss " + str(loss))
                if pass_count > 0:
                    pass_count -= 1
                else:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    trajectory_loss += loss.detach().cpu()
            all_trajectory_train_losses.append(trajectory_loss)
            epoch_training_loss += trajectory_loss
            root_logger.info("        trajectory_loss")
            root_logger.info("        " + str(trajectory_loss))

            chk_traj = {
                'model': model.state_dict(),
                'optimizer':optimizer.state_dict(),
                'scheduler':scheduler.state_dict(),
                'count':count,
                'epoch':None
            }
            torch.save(chk_traj,
                       os.path.join(FLAGS.checkpoint_dir,
                                    "trajectory_checkpoint" + ".pth"))
            model.save_model(FLAGS.checkpoint_dir)
        epoch_training_losses.append(epoch_training_loss)
        root_logger.info("Current mean of epoch training losses")
        root_logger.info(torch.mean(torch.stack(epoch_training_losses)))
        chk_epoch = {
                'model': model.state_dict(),
                'optimizer':optimizer.state_dict(),
                'scheduler':scheduler.state_dict(),
                'count':count,
                'epoch':epoch
            }
        torch.save(chk_epoch, os.path.join(FLAGS.checkpoint_dir,
                                    "epoch_checkpoint" + ".pth"))
        model.save_model(FLAGS.checkpoint_dir, epoch=epoch)
        if epoch == 13:
            scheduler.step()
            root_logger.info("Call scheduler in epoch " + str(epoch))
        epoch_run_times.append(time.time() - current_time)
    chk_final = {
                'model': model.state_dict(),
                'optimizer':optimizer.state_dict(),
                'scheduler':scheduler.state_dict(),
                'count':count,
                'epoch':epoch
            }
    torch.save(chk_final, os.path.join(FLAGS.checkpoint_dir,
                                    "final_checkpoint" + ".pth"))
    model.save_model(FLAGS.checkpoint_dir)
    loss_record = {}
    loss_record['train_total_loss'] = torch.sum(torch.stack(epoch_training_losses))
    loss_record['train_mean_epoch_loss'] = torch.mean(torch.stack(epoch_training_losses)).item()
    loss_record['train_max_epoch_loss'] = torch.max(torch.stack(epoch_training_losses)).item()
    loss_record['train_min_epoch_loss'] = torch.min(torch.stack(epoch_training_losses)).item()
    loss_record['train_epoch_losses'] = epoch_training_losses
    loss_record['all_trajectory_train_losses'] = all_trajectory_train_losses
    return loss_record





def evaluator(model, params, config):
    root_logger = logging.getLogger()
    """Run a model rollout trajectory."""
    #ds = load_dataset(FLAGS.dataset_dir, FLAGS.rollout_split)
    ds = dataset.trajDataset(FLAGS.dataset_dir, FLAGS.rollout_split, transform=functools.partial(split_and_preprocess, field=None, scale=0, gamma=1, 
                                                                                                                    add_noise_bool=False, split_bool=False))
    data_loader = torch.utils.data.DataLoader(ds, batch_size=1)
    trajectories = []
    scalars = []
    loss_record = {}

    #for index in range(FLAGS.num_rollouts):
    for index, trajectory in enumerate(data_loader):
        if index >= FLAGS.num_rollouts:
            break 
        root_logger.info("Evaluating trajectory " + str(index + 1))
        
        scalar_data, prediction_trajectory, error = params['evaluator'].evaluate(model , trajectory)
        error = torch.mean(error, dim=-1)
        if index == 0:
            error_sum = error
        else:
            error_sum += error
        scalars.append(scalar_data)
        trajectories.append(prediction_trajectory)
    error_mean = error_sum / FLAGS.num_rollouts
    error_np = error_mean.detach().numpy().copy()
    np.save('tmp/error',error_np)

    os.makedirs(FLAGS.rollout_dir, exist_ok=True)
    pickle_save(os.path.join(FLAGS.rollout_dir, "rollout.pkl"), trajectories)
    for key in scalars[0]:
        root_logger.info(str(key) + ":" + str(np.mean([x[key] for x in scalars])))
        loss_record[str(key)] = str(np.mean([x[key] for x in scalars]))
    return loss_record

def main(argv):
    del argv
    params = PARAMETERS[FLAGS.model]
    config = {}

    # record start time
    run_step_start_time = time.time()
    run_step_start_datetime = datetime.datetime.fromtimestamp(run_step_start_time).strftime('%c')

    log_dir = os.path.join(FLAGS.checkpoint_dir, 'log')
    if FLAGS.mode == 'train':
        root_logger = logger_setup(os.path.join(log_dir, 'log_train.log'))
    elif FLAGS.mode == 'eval':
        root_logger = logger_setup(os.path.join(log_dir, 'log_eval.log'))
    else:
        root_logger = logger_setup(os.path.join(log_dir, 'log.log'))

    root_logger.info("Program started at time " + str(run_step_start_datetime))

    root_logger.info("Start training......")

    learned_model = core_model_1.EncodeProcessDecode(
      output_size=params['size'],
      latent_size=128,
      num_layers=2,
      message_passing_steps=FLAGS.message_passing_steps,
      num_heads=2)
    model = params['model'].Model(params, learned_model)
    config = None

    if FLAGS.load_chk:
        config = torch.load(os.path.join(FLAGS.checkpoint_dir, "final_checkpoint" + ".pth"))
        model.load_state_dict(config['model'])
        model.load_model(FLAGS.checkpoint_dir, config['epoch'])
        root_logger.info(
            "Loaded checkpoint file in " + str(
                os.path.join(FLAGS.checkpoint_dir, "epoch_checkpoint" + ".pth")) + " and starting retraining...")

    model.to(device)
    train_loss_record = None
    if FLAGS.mode == 'train' or FLAGS.mode == 'all':
        # record train time
        train_start = time.time()
        train_loss_record = learner(model, params, config)
        train_end = time.time()
        train_elapsed_time_in_second = train_end - train_start


        # load train loss if exist and combine the previous and current train loss
        if FLAGS.load_chk:
            saved_train_loss_record = pickle_load(os.path.join(FLAGS.checkpoint_dir, 'log', 'train_loss.pkl'))
            train_loss_record['train_epoch_losses'] = saved_train_loss_record['train_epoch_losses'] + \
                                                      train_loss_record['train_epoch_losses']
            train_loss_record['train_total_loss'] = torch.sum(torch.stack(train_loss_record['train_epoch_losses']))
            train_loss_record['train_mean_epoch_loss'] = torch.mean(
                torch.stack(train_loss_record['train_epoch_losses'])).item()
            train_loss_record['train_max_epoch_loss'] = torch.max(
                torch.stack(train_loss_record['train_epoch_losses'])).item()
            train_loss_record['train_min_epoch_loss'] = torch.min(
                torch.stack(train_loss_record['train_epoch_losses'])).item()
            train_loss_record['all_trajectory_train_losses'] = saved_train_loss_record['all_trajectory_train_losses'] + \
                                                               train_loss_record['all_trajectory_train_losses']
            # load train elapsed time if exist and combine the previous and current train loss
            saved_train_elapsed_time_in_second = pickle_load(
                os.path.join(FLAGS.checkpoint_dir, 'log', 'train_elapsed_time_in_second.pkl'))
            train_elapsed_time_in_second += saved_train_elapsed_time_in_second
        train_elapsed_time_in_second_pkl_file = os.path.join(log_dir, 'train_elapsed_time_in_second.pkl')
        Path(train_elapsed_time_in_second_pkl_file).touch()
        pickle_save(train_elapsed_time_in_second_pkl_file, train_elapsed_time_in_second)
        train_elapsed_time = str(datetime.timedelta(seconds=train_elapsed_time_in_second))

        # save train loss
        train_loss_pkl_file = os.path.join(log_dir, 'train_loss.pkl')
        Path(train_loss_pkl_file).touch()
        pickle_save(train_loss_pkl_file, train_loss_record)

        root_logger.info("Finished training......")


    if FLAGS.mode == 'eval' or FLAGS.mode == 'all':
        root_logger.info("Start evaluating......")
        model.evaluate()
        model.to(device)
        eval_loss_record = evaluator(model, params, config)
        root_logger.info("Finished evaluating......")
    run_step_end_time = time.time()
    run_step_end_datetime = datetime.datetime.fromtimestamp(run_step_end_time).strftime('%c')
    root_logger.info("Program ended at time " + run_step_end_datetime)
    elapsed_time_in_second = run_step_end_time - run_step_start_time
    elapsed_time = str(datetime.timedelta(seconds=elapsed_time_in_second))

    # run summary
    log_run_summary(root_logger)
    root_logger.info("Run total elapsed time " + elapsed_time + "\n")

    if FLAGS.mode == 'train' or FLAGS.mode == 'all':
        root_logger.info("--------------------train loss record--------------------")
        for item in train_loss_record.items():
            root_logger.info(item)






    

if __name__ == '__main__':
  app.run(main)

