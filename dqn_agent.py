import numpy as np
import random
from collections import namedtuple, deque

from model import Qnetwork

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

REPLAY_START_SIZE = 100
UPDATE_EVERY = 4
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent:
    def __init__(self,state_shape,action_size,gamma=0.99,lr=5e-4,
                     buffer_size=int(1e6),batch_size=64,tau=1e-3):
        # defining local and target networks
        self.qnet_local = Qnetwork(state_shape,action_size).to(device)
        self.qnet_target = Qnetwork(state_shape,action_size).to(device)
        
        # set local and target parameters equal to each other
        self.soft_update(tau=1.0)
        
        # experience replay buffer
        self.memory = ReplayBuffer(buffer_size,batch_size)
        
        # defining variables
        self.state_shape = state_shape
        self.action_size = action_size
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.lr = lr
        self.tau = tau
        
        self.t_step = 0
        
        # optimizer
        self.optimizer = optim.RMSprop(self.qnet_local.parameters(), lr=self.lr, alpha=0.95, eps=0.01, centered=True)
    
    def step(self,state,action,reward,next_state,done,t_update):
        """ saves the step info in the memory buffer and perform a learning iteration
        Input : 
            state,action,reward,state,done : non-batched numpy arrays
        
        Output : 
            none
        """
        # add sample to the memory buffer
        self.memory.add(state,action,reward,next_state,done)
        
        # use replay buffer to learn if it has enough samples
        if t_update == 0 and len(self.memory) > REPLAY_START_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences)
        
    def learn(self,experiences):
        """ perform a learning iteration by using sampled experience batch
        Input : 
            experience : tuple from the memory buffer
            states, actions, rewards, next_states, dones = experiences
            eg : states.shape = [N,state_shape]
        Output : 
            none
        """
        #states, actions, rewards, next_states, dones,wj,choose = experiences
        states, actions, rewards, next_states, dones = experiences

        # set optimizer gradient to zero
        self.optimizer.zero_grad()
        
        # predicted action value
        q_pred = self.qnet_local.forward(states).gather(1,actions)
        
        # target action value
        ## use double DQNs, refer https://arxiv.org/abs/1509.06461
        next_action_local = self.qnet_local.forward(next_states).max(1)[1]
        q_target = rewards + self.gamma*(1-dones)*self.qnet_target.forward(next_states)[range(self.batch_size),next_action_local].unsqueeze(1)
        
        # compute td error
        td_error = q_target-q_pred
        # update td error in Replay buffer
        #self.memory.update_td_error(choose,td_error.detach().cpu().numpy().squeeze())

        # defining loss
        #loss = ((wj*td_error)**2).mean()
        loss = ((td_error)**2).mean()
        
        # running backprop and optimizer step
        loss.backward()
        self.optimizer.step()
        
        # run soft update
        self.soft_update(self.tau)
        
    def act(self,state,eps=0.):
        """ return the local model's predicted action for the given state
        Input : 
            state : [state_shape]
        
        Output : 
            action : action with dim = action_size
        """

        if len(self.memory) < REPLAY_START_SIZE:
            return np.random.randint(self.action_size)

        else:    
            # converts lazy array to torch tensor
            state = torch.tensor(state).float().unsqueeze(dim=0).to(device) 
            # change state to (N,C,H,W) format
            state = state.permute(0,3,1,2)
            
            self.qnet_local.eval() # put net in test mode
            with torch.no_grad():
                max_action = np.argmax(self.qnet_local(state)[0].cpu().data.numpy())
            self.qnet_local.train() # put net back in train mode
            
            rand_num = np.random.rand() # sample a random number uniformly between 0 and 1
            
            # implementing epsilon greedy policy
            if rand_num < eps:
                return np.random.randint(self.action_size)
            else: 
                return max_action
        
    def soft_update(self,tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        """
        for target_param, local_param in zip(self.qnet_target.parameters(), self.qnet_local.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    def __init__(self,buffer_size,batch_size=32):
        self.buffer = deque(maxlen=buffer_size)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done","td_error"])
        self.batch_size = batch_size
        self.epsilon = 1e-6
        self.alpha = 2
        self.beta = .3

    def add(self,state,action,reward,next_state,done):
        max_td_error = max([e.td_error for e in self.buffer if e is not None]+[0])
        e = self.experience(state,action,reward,next_state,done,max_td_error)
        self.buffer.append(e)
    
    def update_td_error(self,choose,td_errors):
        abs_td_errors = np.abs(td_errors)
        for j,td_error in zip(choose,abs_td_errors):
            self.buffer[j] = self.buffer[j]._replace(td_error=td_error)

    def sample(self,random=True):
        if random:
            choose = np.random.choice(range(len(self.buffer)),self.batch_size,replace=False)
            experiences = [self.buffer[i] for i in choose]
        else:
            # prioritised experience replay
            pi = np.array([e.td_error for e in self.buffer if e is not None]) + self.epsilon
            Pi = pi**self.alpha
            Pi = Pi/np.sum(Pi)
            wi = (len(self.buffer)*Pi)**(-self.beta)
            wi_ = wi/np.max(wi)
            choose = np.random.choice(range(len(self.buffer)),self.batch_size,replace=False,p=Pi)
            experiences = [self.buffer[j] for j in choose]
            wj = [wi_[j] for j in choose]
            wj = torch.from_numpy(np.vstack(wj)).float().to(device)
        
        states = torch.from_numpy(np.vstack([np.array(e.state)[np.newaxis] for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([np.array(e.next_state)[np.newaxis] for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        
        # change states and next_states to (N,C,H,W) format
        states = states.permute(0,3,1,2)
        next_states = next_states.permute(0,3,1,2)

        return (states,actions,rewards,next_states,dones)#,wj,choose)
    
    def __len__(self):
        return len(self.buffer)