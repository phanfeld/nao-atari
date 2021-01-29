import gym
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import gym
import sys
sys.path.append("../agent/")
from AtariAgent import DQN

from collections import deque
import random

import pickle

import argparse

import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 


#---Helper functions
def preprocess(frame):
    """
    Reduces the input frame to grayscale and ensures storing the data in an appropriate data type.
    Args:
        frame: a (210, 160, 3) frame of the Atari environment
    Returns:
        a (210, 160) frame, pixel values between 0 and 255, stored as uint8.
        
    """               
    return frame.mean(2) / 255.

def frame_to_tensor(frame):
    """
    Turn the frame into a PyTorch tensor for forwarding it through a network. This is done shortly
    before the forward pass to reduce memory usage.
    Args:
        frame: a (210, 160) frame, pixel values between 0 and 255
        device: a PyTorch device
    Returns:
        a (210,160) tensor on the specified device
    """
    tensor = torch.tensor(frame, dtype=torch.float32, requires_grad=False)
    return tensor


def deque_to_tensor(stack):
    '''
    Input: Deque of 4 (210,160) tensors, pixel values between 0 and 1
    Output: Tensor of shape (1,4,210,160) for forward pass through the network
    '''
    stack = list(stack)
    stack = torch.stack(stack).to(device)
    return stack.unsqueeze(0)


def ics(model, transition, target_action, n_actions, epsilon=7e-3, lr=3e-2, _print=False, _max=100):
    assert target_action in range(n_actions), "The target action must be in {}".format(range(n_actions))
    target_action = torch.LongTensor([target_action]).to(device)
    
    delta = torch.zeros((4,210,160), requires_grad=True, device=device)
    opt = torch.optim.Adam([delta], lr=lr)

    pred_action = model.predict_action(transition).item()
    
    logits = model.predict_q(transition + delta)
    
    i = 0
    while torch.argmax(logits) != target_action:
        loss = F.cross_entropy(logits, target_action)
        past_delta = delta
        opt.zero_grad()
        loss.backward()
        opt.step()
        logits = model.predict_q(transition + delta)
        i += 1
        if i > _max:
            break
        if _print:
            if i % 100 == 0:
                 print("Epoch: {}, Loss: {}, Predicted action: {}".format(i, loss, torch.argmax(logits)))

    if _print:
        print("Action {} changed to action {} with perturbation".format(pred_action, \
                                                                     model.predict_action(transition+delta).item()))
        print(i)
    return delta

def test(env_name, mode, runs, intensity, seed=1, gif=False):
    #--Create environment and load trained agent
    env = gym.make('{}Deterministic-v4'.format(env_name))
    env.seed(seed)
    n_actions = env.action_space.n


    if env_name == 'Enduro':
        model = DQN(n_actions, hidden=128)
    else:
        model = DQN(n_actions, hidden=256)

    max_frames = 20000
    model = model.to(device)
    model.load_state_dict(torch.load("../agent/{}.pth".format(env_name)))
    model.eval()
    eps_reward = 0
    frame_number = 0
    done = False

    state = env.reset()
    states = deque(maxlen=4)
    rewards = []

    last_lives = 0

    i = 1

    
    perturbation = None

    if mode in ['ucs', 'cduap']:
        perturbation = pickle.load( open( '../perturbations/{}/{}'.format(env_name, mode), 'rb'))
        perturbation = torch.FloatTensor(perturbation).to(device)

    while i <= runs:
        states.append(frame_to_tensor(preprocess(state)))
        if len(states) < 4:
            action = env.action_space.sample()
        else:
            stack = deque_to_tensor(states)
            if mode == 'None':
                action = model.predict_action(stack)
            else:
                if mode == 'ics':
                    if env_name == 'Breakout':
                        perturbation = ics(model, stack, target_action=0, n_actions=n_actions, lr=5e-4)
                    else:
                        perturbation = ics(model, stack, target_action=4, n_actions=n_actions, lr=5e-4)
                action = model.predict_action(stack + (perturbation * intensity))

        
        frame_number += 1
        next_state, reward, done, info = env.step(action)


        eps_reward += reward
        if env_name == 'Breakout':
            if frame_number >= max_frames:
                next_state = env.reset()
                with open('./{1}/test_{0}_{1}_{2}.log'.format(mode, env_name, intensity), 'a') as reward_file:
                    print("[{}]\tTotal episode reward: ".format(i), eps_reward, file=reward_file)
                rewards.append(eps_reward)
                eps_reward = 0
                frame_number = 0
                done = False
                i += 1
        if done == True:
            #if gif:
            #    generate_gif(frame_number, frames_for_gif, eval_rewards[0], '{0}/gifs/'.format(path))
            next_state = env.reset()
            with open('./{1}/test_{0}_{1}_{2}.log'.format(mode, env_name, intensity), 'a') as reward_file:
                print("[{}]\tTotal episode reward: ".format(i), eps_reward, file=reward_file)
            print("[{}]\tTotal episode reward: ".format(i), eps_reward)
            rewards.append(eps_reward)
            eps_reward = 0
            done = False
            i += 1
        state = next_state

    print("Done!")
    avg_reward_altered = np.average(rewards)
    with open('./{1}/test_{0}_{1}_{2}.log'.format(mode, env_name, intensity), 'a') as reward_file:
        print("Average reward over {} runs: {}+-{}".format(len(rewards), avg_reward_altered, np.std(rewards)), file=reward_file)
    print("Average reward over {} runs: {}+-{}".format(len(rewards), avg_reward_altered, np.std(rewards)))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default='Enduro', type=str, help="define the environment to test the perturbations on")
    parser.add_argument("--int", default=[1, 0.75, 0.5, 0.25, 0.05], nargs='+', type=float, help="define intensities of perturbation")
    parser.add_argument("--mode", default=['ics','ucs','cduap'], nargs='+', help="choose modes: 'ics', 'ucs', 'cduap' or None")
    parser.add_argument("--seed", default=1, nargs='+', type=int, help="set seed for the game environment")
    parser.add_argument("--runs", default=100, type=int, help="set number of test episodes")


    args = parser.parse_args()

    if args.env not in ['Enduro', 'RoadRunner', 'Breakout']:
        print("Choose one of these environments: 'Enduro', 'RoadRunner' or 'Breakout'")
    else:

        env_name = args.env


        runs =args.runs
        seed = args.seed
        modes = args.mode
        intensities = args.int
        print(intensities)

        for mode in modes:
            for intensity in intensities:
                print("Begin test on {}, attack mode '{}', intensity {}".format(env_name, mode, intensity))
                test(env_name, mode, runs, intensity, seed)