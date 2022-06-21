import numpy as np
import torch
import gym
from torch import nn
import matplotlib.pyplot as plt
import park

from plato.utils.reinforcement_learning.policies import base
from plato.trainers import basic
from plato.config import Config
from plato.trainers import basic
from torch.nn.utils.clip_grad import clip_grad_norm_

import os
import logging

import csv

import park
import a2c
# Memory
# Stores results from the networks, instead of calculating the operations again from states, etc.

class StateNormalizer(object):
    def __init__(self, obs_space):
        self.shift = obs_space.low
        self.range = obs_space.high - obs_space.low

    def normalize(self, obs):
        return (obs - self.shift) / self.range

class Memory():
    def __init__(self):
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []

    def add(self, log_prob, value, reward, done):
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)
    
    def clear(self):
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.dones.clear()  
    
    def _zip(self):
        return zip(self.log_probs,
                self.values,
                self.rewards,
                self.dones)
    
    def __iter__(self):
        for data in self._zip():
            return data
    
    def reversed(self):
        for data in list(self._zip())[::-1]:
            yield data
    
    def __len__(self):
        return len(self.rewards)

class Trainer(basic.Trainer):
    def __init__(self, model=None):
        super().__init__()
        #pass

        self.env = park.make(Config().algorithm.env_park_name) #make it park env soon

        #env = park.make(Config().algorithm.env_park_name)

        seed = self.client_id

        self.env.seed(seed)
        self.env.reset()
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.env_name = model.get_env_name()
        self.algorithm_name = model.get_rl_algo()

        self.model = model
        self.actor = model.actor
        self.critic = model.critic
        self.adam_actor = torch.optim.Adam(self.actor.parameters(), lr=Config().algorithm.learning_rate)
        self.adam_critic = torch.optim.Adam(self.critic.parameters(), lr=Config().algorithm.learning_rate)

        self.memory = Memory()
        self.obs_normalizer = StateNormalizer(self.env.observation_space)

        self.episode_reward = []
        self.server_reward = []
        self.avg_reward = []
        #self.avg_reward = 0
        self.episode_num = 0
        self.total_reward = 0
        self.done = True
        self.trace_idx = 0
        self.steps = 0

        self.actor_state_dict = None
        self.critic_state_dict = None

        self.timesteps_since_eval = 0

        if not os.path.exists(Config().results.results_dir):
            os.makedirs(Config().results.results_dir)


    def t(self, x): 
        return torch.from_numpy(x).float()


    def train_model(self, config, trainset, sampler, cut_layer):
        """Main Training"""
        #We will put what exectues in the "main function of a2c_abr_sim.py here"

        round_episodes = 0
        while round_episodes < Config().algorithm.max_round_episodes:
            
            #Evaluates policy at a frequency set in config file
            if self.timesteps_since_eval >= Config().algorithm.eval_freq:
                self.avg_reward.append(self.evaluate_policy())
                path = Config().results.results_dir +"/"+Config().results.file_name+"_"+str(self.client_id)+"_avg_reward"
                np.savez("%s" %(path), a=self.avg_reward)
                np.savetxt("%s.csv" %(path), self.avg_reward, delimiter=",")
                self.timesteps_since_eval = 0

            self.done = False
            self.total_reward = 0
            self.trace_idx = 0
            #TODO different clients with differnet traces
            if (self.episode_num % 700 == 0):
                self.trace_idx = int(self.episode_num / 700)
                print( "change trace to: ", self.trace_idx )
            state = self.env.reset(trace_idx=self.trace_idx)
            state = self.obs_normalizer.normalize(state)
            self.steps = 0

            while not self.done:
                probs = self.actor(self.t(state))
                dist = torch.distributions.Categorical(probs=probs)
                action = dist.sample()

                next_state, reward, self.done, info = self.env.step(action.detach().data.numpy())
                next_state = self.obs_normalizer.normalize(next_state)

                self.total_reward += reward
                self.steps += 1
                self.memory.add(dist.log_prob(action), self.critic(self.t(state)), reward, self.done)

                state = next_state
                
                if self.done or (self.steps % Config().algorithm.batch_size == 0):
                    last_q_val = self.critic(self.t(next_state)).detach().data.numpy()
                    self.train_helper(self.memory, last_q_val)
                    self.memory.clear()

            self.episode_num += 1
            self.timesteps_since_eval += 1
            self.episode_reward.append(self.total_reward)
            np.savez("%s" %(Config().results.results_dir +"/"+Config().results.file_name+"_"+str(self.client_id)), a=self.episode_reward)
            np.savetxt("%s.csv" %(Config().results.results_dir +"/"+Config().results.file_name+"_"+str(self.client_id)), self.episode_reward, delimiter=",")
            round_episodes += 1
            print("Episode number: %d, Reward: %d" % (self.episode_num, self.total_reward))

        path = Config().results.results_dir +"/"+Config().results.file_name+"_"+str(self.client_id)+"_avg_reward"
        self.avg_reward.append(self.evaluate_policy())
        np.savez("%s" %(path), a=self.avg_reward)
        np.savetxt("%s.csv" %(path), self.avg_reward, delimiter=',')


        

    def train_helper(self, memory, q_val):
        #We will put the train loop here
        values = torch.stack(memory.values)
        q_vals = np.zeros((len(memory), 1))

        #target values calculated backward
        #important to handle correctly done states
        #for those cases we want our target to be equal to the reward only
        for i, (_, _, reward, done) in enumerate(memory.reversed()):
            q_val = reward + Config().algorithm.gamma*q_val*(1.0-done)
            q_vals[len(memory)-1 - i] = q_val #store values from end to the start
        
        #advantage function!!
        advantage = torch.Tensor(q_vals) - values

        critic_loss = advantage.pow(2).mean() #loss_value_v
        self.adam_critic.zero_grad()
        critic_loss.backward()
        
        entropy_loss = (torch.stack(memory.log_probs) * torch.exp(torch.stack(memory.log_probs))).mean()
        actor_loss = (-torch.stack(memory.log_probs)*advantage.detach()).mean() + entropy_loss * Config().algorithm.entropy_ratio
        if Config().algorithm.grad_clip_val > 0:
            clip_grad_norm_(self.actor.parameters(), Config().algorithm.grad_clip_val)

        self.adam_actor.zero_grad()
        actor_loss.backward()

        self.adam_critic.step()
        self.adam_actor.step()



                
    def load_model(self, filename=None, location=None):
        """Loading pre-trained model weights from a file."""
        #We will load actor and critic models here

        #TODO LOADING CAUSES A DISCONNECTION
        model_path = Config(
        ).params['model_path'] if location is None else location
        actor_model_name = 'actor_model'
        critic_model_name = 'critic_model'
        env_algorithm = self.env_name+ self.algorithm_name

        if filename is None:
            actor_filename = filename + '_actor'
            actor_model_path = f'{model_path}/{actor_filename}'
            critic_filename = filename + '_critic'
            critic_model_path = f'{model_path}/{critic_filename}'
        else:
            actor_model_path = f'{model_path}/{env_algorithm+actor_model_name}.pth'
            critic_model_path = f'{model_path}/{env_algorithm+critic_model_name}.pth'
    
        if self.client_id == 0:
            logging.info("[Server #%d] Loading models from %s, and %s.", os.getpid(),
                         actor_model_path, critic_model_path)
        else:
            logging.info("[Client #%d] Loading a model from %s, and %s.",
                         self.client_id, actor_model_path, critic_model_path)

        if self.client_id != 0:
           
            file_name = "%s_%s.npz" % ("training_status", str(self.client_id)) 
            file_path = os.path.join(model_path, file_name)
            data = np.load(file_path)
            self.episode_num = int((data['a'])[0])
    
            self.actor.load_state_dict(torch.load(actor_model_path), strict=True)
            self.critic.load_state_dict(torch.load(critic_model_path), strict=True)

            #load episode_reward so it doesn't overwrite
            arr = np.load("%s.npz" %(Config().results.results_dir +"/"+Config().results.file_name+"_"+str(self.client_id)))
            self.episode_reward = list(arr['a'])

            #load avg_reward so it doesn't overwrite
            path = Config().results.results_dir +"/"+Config().results.file_name+"_"+str(self.client_id)+"_avg_reward"
            arr = np.load("%s.npz" %(path))
            self.avg_reward = list(arr['a'])

            #unsure if we need these
            self.adam_actor = torch.optim.Adam(self.actor.parameters(), lr=Config().algorithm.learning_rate)
            self.adam_critic = torch.optim.Adam(self.critic.parameters(), lr=Config().algorithm.learning_rate)



    def save_model(self, filename=None, location=None):
        """Saving the model to a file."""
        #We will save actor and critic models here
        """Saving the model to a file."""
        model_path = Config(
        ).params['model_path'] if location is None else location
        actor_model_name = 'actor_model'
        critic_model_name = 'critic_model'
        env_algorithm = self.env_name+ self.algorithm_name

        try:
            if not os.path.exists(model_path):
                os.makedirs(model_path)
        except FileExistsError:
            pass

        if filename is None:
           # model_path = f'{model_path}/{filename}'
           # model_filename = filename + _'model'
           # model path = Config().params stuff
           actor_filename = filename + '_actor'
           critic_filename = filename + '_critic'
           actor_model_path = f'{model_path}/{actor_filename}'
           critic_model_path = f'{model_path}/{critic_filename}'
        else:
            actor_model_path = f'{model_path}/{env_algorithm+actor_model_name}.pth'
            critic_model_path = f'{model_path}/{env_algorithm+critic_model_name}.pth'

        if self.model_state_dict is None:
            torch.save(self.actor.state_dict(), actor_model_path)
            torch.save(self.critic.state_dict(),critic_model_path)
        else:
            torch.save(self.actor_state_dict, actor_model_path)
            torch.save(self.critic_state_dict, critic_model_path)

        if self.client_id != 0:
            file_name = "%s_%s.npz" % ("training_status", str(self.client_id)) 
            file_path = os.path.join(model_path, file_name)
            np.savez(file_path, a=np.array([self.episode_num]))       

        if self.client_id == 0:
            logging.info("[Server #%d] Saving models to %s, and %s.", os.getpid(),
                         actor_model_path, critic_model_path)
        else:
            logging.info("[Client #%d] Saving a model to %s, and %s.",
                         self.client_id, actor_model_path, critic_model_path)


    async def server_test(self, testset, sampler=None, **kwargs):
        #We will return the average reward here
        avg_reward = self.evaluate_policy()
        self.server_reward.append(avg_reward)
        file_name = "A2C_RL_SERVER"
        np.savetxt("%s.csv" %(Config().results.results_dir +"/"+file_name), self.server_reward, delimiter=",")
        return avg_reward

        
    def evaluate_policy(self, eval_episodes = 10):
        
        for trace_idx in range(3):
            avg_reward = 0
            for _ in range(eval_episodes):
                episode_reward = 0
                done = False
                state = self.env.reset(trace_idx=trace_idx)
                state = self.obs_normalizer.normalize(state)
                while not done:
                    probs = self.actor(self.t(state))
                    dist = torch.distributions.Categorical(probs=probs)
                    action = dist.sample()
                
                    next_state, reward, done, info = self.env.step(action.detach().data.numpy())
                    next_state = self.obs_normalizer.normalize(next_state)
                    state = next_state
                    episode_reward += reward

                avg_reward += episode_reward
            avg_reward /= eval_episodes
            print("------------------")
            print("Average Reward over trace %s is %s" % (str(trace_idx), str(avg_reward)))
            print("------------------")
            return avg_reward
        #with open("evaluations.csv", 'a') as f:
            #writer = csv.writer(f)
            #writer.writerow(avg_rewards)
