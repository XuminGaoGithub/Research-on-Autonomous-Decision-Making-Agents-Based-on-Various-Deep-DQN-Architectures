import torch
import random, numpy as np
from pathlib import Path
from torch.autograd import Variable
from neural import MarioNet,MarioNet_RNN,MarioNet_Dueling
from transformer import vit_base_patch16_224_in21k as transformer_model
from collections import deque


class Mario:
    def __init__(self, state_dim, action_dim, save_dir, agent_type='', checkpoint=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.agent_type = agent_type
        self.memory = deque(maxlen = 18000) # 100000
        self.batch_size = 64

        self.exploration_rate = 1
        self.exploration_rate_decay = 0.99999975
        self.exploration_rate_min = 0.1 #0
        self.gamma = 0.9

        self.curr_step = 0
        self.burnin = 1e4 # 1e5 # min. experiences before training
        self.learn_every = 3   # no. of experiences between updates to Q_online
        self.sync_every = 1000 #4, 4 eposide update target_net  # no. of experiences between Q_target & Q_online sync

        self.save_every = 1000#1000 # 500 #500 eposide save once   # no. of experiences between saving Mario Net
        self.save_dir = save_dir

        self.use_cuda = torch.cuda.is_available()
        
        if self.use_cuda:
            print("train on cuda") 
        else:
            print("train on cpu")
            
        # Mario's DNN to predict the most optimal action - we implement this in the Learn section
        if self.agent_type == 'Nature_DQN': #Nature_DQN using CNN
            self.policy_net = MarioNet(self.state_dim, self.action_dim).float()
            self.target_net = MarioNet(self.state_dim, self.action_dim).float()
        elif self.agent_type == 'Nature_DQN_RNN': #Nature_DQN using RNN
            self.policy_net = MarioNet_RNN(self.state_dim, self.action_dim).float()
            self.target_net = MarioNet_RNN(self.state_dim, self.action_dim).float()
        elif self.agent_type == 'Nature_DQN_Transformer': #Nature_DQN using Transformer
            self.policy_net = transformer_model(self.state_dim[-1],self.action_dim).float()
            self.target_net = transformer_model(self.state_dim[-1], self.action_dim).float()
        elif self.agent_type == 'Double_DQN': #Double_DQN using CNN
            self.policy_net = MarioNet(self.state_dim, self.action_dim).float()
            self.target_net = MarioNet(self.state_dim, self.action_dim).float()
        elif self.agent_type == 'Dueling_DQN': #Dueling_DQN using CNN
            self.policy_net = MarioNet_Dueling(self.state_dim, self.action_dim).float()
            self.target_net = MarioNet_Dueling(self.state_dim, self.action_dim).float()

        if self.use_cuda:
            self.policy_net = self.policy_net.to(device='cuda')
            self.target_net = self.target_net.to(device='cuda')
        if checkpoint:
            self.load(checkpoint)

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=0.00001) # 0.000025
        self.loss_fn = torch.nn.SmoothL1Loss()


    def act(self, state):
        """
        Given a state, choose an epsilon-greedy action and update value of step.

        Inputs:
        state(LazyFrame): A single observation of the current state, dimension is (state_dim)
        Outputs:
        action_idx (int): An integer representing which action Mario will perform
        """
        # EXPLORE
        if np.random.rand() < self.exploration_rate:
        #if np.random.rand() < 0:

            action_idx = np.random.randint(self.action_dim)

        # EXPLOIT
        else:

            state = torch.FloatTensor(state).cuda() if self.use_cuda else torch.FloatTensor(state)
            state = state.unsqueeze(0)

            action_values = self.policy_net(state)
            action_idx = torch.argmax(action_values, axis=1).item()

        # decrease exploration_rate
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        # increment step
        self.curr_step += 1
        return action_idx

    def cache(self, state, next_state, action, reward, done):
        """
        Store the experience to self.memory (replay buffer)
        Inputs:
        state (LazyFrame),
        next_state (LazyFrame),
        action (int),
        reward (float),
        done(bool))
        """

        state = torch.FloatTensor(state).cuda() if self.use_cuda else torch.FloatTensor(state)
        next_state = torch.FloatTensor(next_state).cuda() if self.use_cuda else torch.FloatTensor(next_state)
        action = torch.LongTensor([action]).cuda() if self.use_cuda else torch.LongTensor([action])
        reward = torch.DoubleTensor([reward]).cuda() if self.use_cuda else torch.DoubleTensor([reward])
        done = torch.BoolTensor([done]).cuda() if self.use_cuda else torch.BoolTensor([done])
        self.memory.append( (state, next_state, action, reward, done,) )


    def recall(self):
        """
        Retrieve a batch of experiences from memory
        """
        batch = random.sample(self.memory, self.batch_size)
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

    def td_estimate(self, state, action):

        #print('state:', state)
        #print('state.shape:', state.shape)
        #print('[np.arange(0, self.batch_size), action]:',[np.arange(0, self.batch_size), action])
        #print(' self.policy_net(state)',  self.policy_net(state))
        #print(' self.policy_net(state)[0]', self.policy_net(state[0]))
        #print('state,',state)

        # do computing for each item from one batch,then merge the result
        if self.agent_type == 'Nature_DQN_RNN' or self.agent_type == 'Nature_DQN_Transformer':
            policy_out = []
            for i in range(state.shape[0]):
                #print('state[i]:',state[i])
                RNN_out = self.policy_net(state[i])
                #print('RNN_out:',RNN_out)
                RNN_out = RNN_out.numpy().tolist()
                #print('list_RNN_out[0]:', RNN_out[0])
                policy_out.append(RNN_out[0])
            #print('policy_out',policy_out)
            current_Q = torch.Tensor(policy_out)
            current_Q = Variable(current_Q, requires_grad=True)
            current_Q = current_Q.cuda()
            current_Q = current_Q[np.arange(0, self.batch_size), action]  # Q_online(s,a)
            #print('current_Q', current_Q)

        else:
            #print('self.policy_net(state):',self.policy_net(state))
            current_Q = self.policy_net(state)[np.arange(0, self.batch_size), action] # Q_online(s,a)
            #print('current_Q',current_Q)

        return current_Q

    @torch.no_grad()
    def td_target(self, reward, next_state, done):

        # refer to https://github.com/sachinruk/Mario/blob/master/dqn_agent.py
        if self.agent_type=='Nature_DQN' or self.agent_type == 'Nature_DQN_RNN' or self.agent_type == 'Nature_DQN_Transformer': #Nature_DQN
            #print('next_state.shape,', next_state.shape)

            # do computing for each item from one batch,then merge the result
            if self.agent_type == 'Nature_DQN_RNN' or self.agent_type == 'Nature_DQN_Transformer':
                Q_out = []
                for i in range(next_state.shape[0]):
                    out = self.target_net(next_state[i]).max(-1, keepdim=True)[0]
                    #print('out:',out)
                    out = out.numpy().tolist()
                    #print('list_out[0]:', out[0])
                    Q_out.append(out[0])
                #print('Q_out',Q_out)
                next_Q = torch.Tensor(Q_out)
                next_Q = next_Q.cuda()
                #print('next_Q', next_Q)
                return (reward + (1 - done.float()) * self.gamma * next_Q).float()

            else:
                next_Q = self.target_net(next_state).max(-1, keepdim=True)[0]
                #print('next_Q',next_Q)
                return (reward + (1 - done.float()) * self.gamma * next_Q).float()

        else:

            next_state_Q = self.policy_net(next_state)
            best_action = torch.argmax(next_state_Q, axis=1)
            next_Q = self.target_net(next_state)[np.arange(0, self.batch_size), best_action]
            return (reward + (1 - done.float()) * self.gamma * next_Q).float()

    def update_Q_online(self, td_estimate, td_target) :
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()


    def sync_Q_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())


    def learn(self):
        # if self.curr_step % self.sync_every == 0:
        #     self.sync_Q_target()

        # if self.curr_step % self.save_every == 0:
        #     self.save()

        if self.curr_step < self.burnin:
            return None, None

        if self.curr_step % self.learn_every != 0:
            return None, None

        # Sample from memory
        state, next_state, action, reward, done = self.recall()

        # Get TD Estimate
        td_est = self.td_estimate(state, action)

        # Get TD Target
        td_tgt = self.td_target(reward, next_state, done)

        # Backpropagate loss through Q_online
        #print('td_est:',td_est)
        #print('td_tgt', td_tgt)
        #print('td_est.shape:', td_est.shape)
        #print('td_tgt.shape', td_tgt.shape)
        loss = self.update_Q_online(td_est, td_tgt)
            
        return (td_est.mean().item(), loss)


    def save(self,e,agent):
        if agent == 'Nature_DQN':
            save_path = self.save_dir /  f"Nature_DQN_mario_net_{int(e // self.save_every)}.chkpt"
        elif agent == 'Nature_DQN_RNN':
            save_path = self.save_dir /  f"Nature_DQN_RNN_mario_net_{int(e // self.save_every)}.chkpt"
        elif agent == 'Nature_DQN_Transformer':
            save_path = self.save_dir / f"Nature_DQN_Transformer_mario_net_{int(e // self.save_every)}.chkpt"
        elif agent == 'Double_DQN':
            save_path = self.save_dir /  f"Double_DQN_mario_net_{int(e // self.save_every)}.chkpt"
        elif agent == 'Dueling_DQN':
            save_path = self.save_dir /  f"Dueling_DQN_mario_net_{int(e // self.save_every)}.chkpt"

        #save_path = self.save_dir /  f"mario_net_{int(e // self.save_every)}.chkpt"
        torch.save(
            dict(
                model=self.policy_net.state_dict(),
                exploration_rate=self.exploration_rate
            ),
            save_path
        )
        print(f"MarioNet saved to {save_path} at step {self.curr_step}")


    def load(self, load_path):
        if not load_path.exists():
            raise ValueError(f"{load_path} does not exist")

        ckp = torch.load(load_path, map_location=('cuda' if self.use_cuda else 'cpu'))
        exploration_rate = ckp.get('exploration_rate')
        state_dict = ckp.get('model')

        print(f"Loading model at {load_path} with exploration rate {exploration_rate}")
        self.policy_net.load_state_dict(state_dict)
        self.sync_Q_target()
        self.exploration_rate = exploration_rate
