#!/usr/bin/env python
# coding: utf-8

# In[17]:


import os
import glob
import time
from datetime import datetime

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import torch.nn.functional as F
import random
import numpy as np


class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.UE_ID = []

    

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.UE_ID[:]



class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init, test, device, time_step):
        super(ActorCritic, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space
        self.test=test
        self.time_step = time_step
        self.device = device
        
        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)
            

        # actor
        if has_continuous_action_space :
            self.actor = nn.Sequential(
                            nn.Linear(state_dim, 32),
                            nn.ReLU(),
                            nn.Linear(32, 32),
                            nn.ReLU(),
                            nn.Linear(32, 32),
                            nn.ReLU(),
                            nn.Linear(32, 32),
                            nn.ReLU(),
                            nn.Linear(32, 32),
                            nn.ReLU(),
                            nn.Linear(32, 32),
                            nn.ReLU(),
                            nn.Linear(32, action_dim),
                            nn.Tanh()
                        )
        else:
            self.actor = nn.Sequential(
                            nn.Linear(state_dim, 30),
                            nn.ReLU(),
                            nn.Linear(30, 20),
                            nn.ReLU(),
                            nn.Linear(20, 20),
                            nn.ReLU(),
                            nn.Linear(20, action_dim),
                            nn.Softmax(dim=-1)
                        )

        
        # critic
        self.critic = nn.Sequential(
                        nn.Linear(state_dim, 1000),
                        nn.ReLU(),
                        nn.Linear(1000, 500),
                        nn.ReLU(),
                        nn.Linear(500, 250),
                        nn.ReLU(),
                        nn.Linear(250, 100),
                        nn.ReLU(),
                        nn.Linear(100, 1)
                        #nn.Sigmoid()
            
                    )
        
        self.encoder1 = torch.nn.Sequential(torch.nn.Linear(10, 10),
                                    torch.nn.ReLU())

        
        #self.encoder1 = torch.nn.Sequential(torch.nn.Linear(594, 400),
        #                    torch.nn.ReLU(),
        #                    torch.nn.Linear(400, 200),
        #                    torch.nn.ReLU(),
        #                    torch.nn.Linear(200, 100),
        #                    torch.nn.ReLU()) 


    def set_action_std(self, new_action_std):

        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")


    def forward(self):
        raise NotImplementedError
        
        
    def encoding2(self, x_t):    
        
        X = self.critic(x_t)
        return X
    
    def encoding(self, adj,x_t,num_of_UE, num_of_micro, t_graph):
        
        #Cell_X= [[0,0,0,0,0,0,1],[0,0,0,0,0,1,0],[0,0,0,0,1,0,0],[0,0,0,1,0,0,0],[0,0,1,0,0,0,0],\
        #         [0,1,0,0,0,0,0],[1,0,0,0,0,0,0]]
        #th_Cell_X=torch.FloatTensor(Cell_X).to(self.device)
        
        X1= (adj @ x_t)/num_of_UE
        #print('X1')
        #print(X1.size())
        #X2= torch.cat((X1[:num_of_micro],th_Cell_X), 1)
        X2= X1[:num_of_micro]
        #print('GIN')
        #print(X2.size())
        #print('g')
        #print(t_graph.size())
        X3= torch.cat((X2,t_graph), 1)
        #print(X2)
        X4= torch.reshape(X3, (-1,))
        
        #X3 = torch.reshape(X, (-1,))
        #print(X4.size())
        X5 = self.encoder1(X4)
        
        #print('graphencode')
        return X5
    
    def act(self, state, connect, timestep):

        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            #print('state: '+str(state))
            action_probs = self.actor(state)
            
            dist = Categorical(action_probs)
            
        #print('dist ' +str(dist))
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        
        
        '''
        if (self.test):
            action = torch.argmax(action_probs)
            action_logprob = torch.log(torch.max(action_probs))
        '''
        #print(action_probs)
        #print(action_probs[7])
        #print(torch.log(action_probs[7]))
        '''
        if connect == 1 and timestep%2 ==0:
            action = torch.argmax(state[:7]) 
            action_logprob = dist.log_prob(action)
            
        
        if connect == 2 and timestep%2 ==0: 
            i = torch.argmax(state[:7])
            T = torch.cat([state[0:i], state[i:]])
            T[i] = 0
            action = torch.argmax(T[:7])
            #action = torch.tensor(7).to(self.device)
            action_logprob = dist.log_prob(action)
        
        
        if connect == 1 and timestep%1 ==0 and not (0 in state[7:14]):
            y = torch.div(state[:7],state[7:14]) 
            action = torch.argmax(y[:7]) 
            action_logprob = dist.log_prob(action)
            
            
        if connect == 2 and timestep%1 ==0 and not (0 in state[7:14]): 
            y = torch.div(state[:7],state[7:14]) 
            i = torch.argmax(y[:7])             
           
            T = torch.cat([state[0:i], state[i:]])
            T[i] = 0
            y1 =torch.div(T[:7],T[7:14]) 
            action = torch.argmax(y1[:7])
            action_logprob = dist.log_prob(action)
        '''
            
        '''
        R = timestep 
        
        if connect == 2 and random.randint(0, R) == 7:
                  
            action = torch.tensor(7).to(self.device)
            action_logprob = dist.log_prob(action)
       '''  
        #if timestep%2==0:
        #    print(state)
        #print('ac: ' +str(action))
       

        return action.detach(), action_logprob.detach()
    

    def evaluate(self, state):
        '''
        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(device)
            dist = MultivariateNormal(action_mean, cov_mat)
            
            # for single action continuous environments
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)

        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)
        '''
        #action_logprobs = dist.log_prob(action)
        #dist_entropy = dist.entropy()
        state_values = self.critic(state)
        #state_values_all = self.critic(state)
        #print('all:' +str(state_values_all))
        #state_values = state_values_all[action]
        
        #return action_logprobs, state_values, dist_entropy
        return state_values

class PPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std, device ,test, time_step, re_mean, re_std, num_ue ):

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_std = action_std_init

        self.time_step = time_step
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.buffer = RolloutBuffer()
        self.test = test
        self.re_mean = re_mean
        self.re_std =re_std

        self.policy = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std, test,device, self.time_step).to(device)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                    ])

        self.policy_old = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std,test, device,time_step).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
        self.CRE = nn.CrossEntropyLoss()
        self.device = device
        self.num_ue = num_ue
        

    def set_action_std(self, new_action_std):
        
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)
        
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        print("--------------------------------------------------------------------------------------------")

        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if (self.action_std <= min_action_std):
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)

        else:
            print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")

        print("--------------------------------------------------------------------------------------------")
    
    def getStatesSize(self):
        return len(self.buffer.states)


    #def select_action(self, state, timestep, th, graph_edge, node_features,new_graph_edge, num_of_UE, num_of_micro):
    def select_action(self, state, timestep, num_of_UE, num_of_st, c_st, E):
            
        #print(graph_edge)
        #print(node_features)
        #print(state)

        if self.has_continuous_action_space:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(self.device)
                action, action_logprob = self.policy_old.act(state)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)

            return action.detach().cpu().numpy().flatten()

        else:
            with torch.no_grad():
                
                #state_0 = []
                #encode = []
                #t_graph_edge = list(map(list, zip(*graph_edge)))
                #t_new_graph_edge = list(map(list, zip(*new_graph_edge)))
                
                #tx_graph=torch.FloatTensor(t_new_graph_edge).to(self.device)
                #print(tx_graph.size())


                
                            
                #adj = torch.FloatTensor(adj).to(self.device)
                
                #print('adj')
                #print(adj.size())
                #x_graph=torch.FloatTensor(graph_edge).to(self.device)
                #x_node_features=torch.FloatTensor(node_features).to(self.device)
                
                #x_t= torch.cat((x_graph,x_node_features), 1)

                #v_encode= torch.reshape(x_t, (-1,))
                #print(v_encode.size())
                
                #print('x_node_feature')
                #print(x_node_feature)

                #encode = torch.FloatTensor(encode).to(self.device)
                #dim = 168
                #print(encode)
                #v_encode = self.policy_old.encoding(adj,x_t,num_of_UE, num_of_micro, tx_graph)       
                #print(v_encode.size())
                
                #for i in range(num_of_micro+3):
                    
                #    state_0.append(state[i]) 
                
                #for i in range(num_of_micro+1):
                #    state_0.append(0)
                    
               
                #print('s0')
                #print(state_0)
                #new = torch.cat((state_0,v_encode),0)
                
                
                state_0 = torch.FloatTensor(state).to(self.device)
                eg = random.random()
                
                if eg <E:
                    r= random.randint(0,len(c_st)-1)
                    eg= c_st[r]
                    #eg = random.randint(0,num_of_st-1)
                    ac =[0]*num_of_st
                    ac[eg]=1
                    ac_t = torch.FloatTensor(ac).to(self.device)
                    new1 = torch.cat((state_0,ac_t),0)
                    
                    self.buffer.states.append(new1)
                    fin_ac = eg
                    
                else:
                    max_ac_v =  -999999
                    max_ac = -999999
                    #print('- - - - ')
                    #print(c_st)
                    for i in range(len(c_st)):
                    #for i in range(num_of_st):
                        ac =[0]*num_of_st
                        
                        ac[c_st[i]]=1
                        #ac[i]=1
                        ac_t = torch.FloatTensor(ac).to(self.device)
                        new1 = torch.cat((state_0,ac_t),0)
                        q_value = self.policy_old.encoding2(new1)
                        
                        if E ==0:
                            print('st: ' +str(c_st[i]))
                            print(q_value)
                        
                        if q_value> max_ac_v:
                            #max_ac=i
                            max_ac=c_st[i]
                            max_ac_v = q_value
                        else:
                            pass
                        
                    ac =[0]*num_of_st    
                    ac[max_ac]=1
                    ac_t = torch.FloatTensor(ac).to(self.device)
                    new1 = torch.cat((state_0,ac_t),0)    
                    self.buffer.states.append(new1)
                    fin_ac = max_ac
                    #print('____')
                    
                return fin_ac
                
                #action, action_logprob = self.policy_old.act(new ,1, timestep)
                
                #self.buffer.states.append(new)
                #self.buffer.actions.append(action)
                #self.buffer.logprobs.append(action_logprob)
               
            
                '''
                for i in range(num_connect):
                    action, action_logprob = self.policy_old.act(new ,1, timestep)
                    self.buffer.states.append(new)
                    self.buffer.actions.append(action)
                    self.buffer.logprobs.append(action_logprob)
                    print('action')
                    print(action)
                '''    
                #thr = torch.log(torch.tensor(1/8)).to(self.device)
                        
            #print('ac0: '+str(action))
            
            #self.buffer.states.append(new)
            #self.buffer.actions.append(action)
            #self.buffer.logprobs.append(action_logprob)
            
            #print('done 1')
            
            ###seq2sdeq###   
            #print('done 2')
            #print(action.item(), action_1.item())
            
            


    def update(self):

        # Monte Carlo estimate of returns
        #print(self.buffer.UE_ID)
        #print(self.buffer.is_terminals)
        '''
        UE_table =[]
        for i in range(self.num_ue):
            UE_table.append(0)
        rewards = []
        discounted_reward = 0
        count_ac = 1
        num = 0
        for i in range(len(self.buffer.rewards)):
        #for i in range(10):
            
            UE_table =[]
            for u in range(self.num_ue):
                UE_table.append(0)

            cul_reward = 0
            for j in range(self.num_ue):
                UE_reward = 0
                #print('UE: ' +str(j))
                for k in range(i,len(self.buffer.rewards)):
                    
                #for k in self.buffer.UE_ID[i:]:
                    if k== 0 or (k !=0 and self.buffer.is_terminals[k-1] != True):
                        
                        if j == self.buffer.UE_ID[k]:
                       
                        
                            #print('k: ' +str(k))
                            
                            discounted_reward = self.buffer.rewards[k]*(self.gamma**UE_table[j])
                            
                            UE_reward += discounted_reward
                            #print(discounted_reward)
                            if (k+1)%2==0:
                                UE_table[j]+=1
                            
                                
                    elif self.buffer.is_terminals[k-1] == True:
                        #print('end: ' +str(k))
                        break
                                        
                    else:
                        pass

                cul_reward +=  UE_reward
            rewards.append(cul_reward)
            
        #print(rewards)
        
        '''
        rewards = []
        discounted_reward = 0
        #count_a0 = 1
        #count_a1 = 0
        #for i in self.buffer.rewards:
            #if type(i) == str:
                #print('y')
                #print(i)
        #print(self.buffer.rewards)
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
                #count_a0 = 1
                #count_a1 = 0
                
            #discounted_reward = (reward + (self.gamma * discounted_reward)) /count_a0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
            #count_a0 = count_a0+self.gamma**count_a1
            #count_a1+=1
            #print(reward)
            
            #discounted_reward
        #print('dis_count_reward '+str(rewards))
        
        
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        #rewards = (rewards) / (rewards.std() + 1e-7)
        #rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        
        self.re_mean = rewards.mean()
        self.re_std = rewards.std()
        #forward_actions = self.forward_action.detach().to(self.device)
        #forward_states  = self.forward_state.detach().to(self.device)
        
        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(self.device)
        #old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(self.device)
        #old_logprobs =torch.squeeze(torch.stack(self.buffer.logprobs,dim=0)).detach().to(self.device)

        
        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            #logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            state_values = self.policy.evaluate(old_states)
            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            #找Importance Weight
            #ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
           
            #advantages = rewards 
        
            #advantages = rewards - state_values.detach()   
            #surr1 = ratios * advantages
            #surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            
            #print('device_dis_count_reward '+str(advantages))
            # final loss of clipped objective PPO
            
            #loss = -torch.min(surr1, surr2) - 0.01*dist_entropy
            
            
            
            #loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
            loss = self.MseLoss(state_values, rewards) 

                
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        #print(self.MseLoss(state_values, rewards))
        
        #print(state_values[::])
        #print(rewards[::])
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()
    
    
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
   

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        
        
       


# In[20]:



'''
print("============================================================================================")


################################### Training ###################################


####### initialize environment hyperparameters ######

#env_name = "CartPole-v1"
#env_name = "MountainCar-v0"
#has_continuous_action_space = False
#action_std = None           # set same std for action distribution which was used while saving



env_name="MountainCarContinuous-v0"
has_continuous_action_space=True
max_ep_len = 300                  # max timesteps in one episode

max_training_timesteps = int(1000)   # break training loop if timeteps > max_training_timesteps

print_freq = max_ep_len * 4     # print avg reward in the interval (in num timesteps)
log_freq = max_ep_len * 2       # log avg reward in the interval (in num timesteps)
save_model_freq = int(2e4)      # save model frequency (in num timesteps)

action_std = 0.6                    # starting std for action distribution (Multivariate Normal)
action_std_decay_rate = 0.05        # linearly decay action_std (action_std = action_std - action_std_decay_rate)
min_action_std = 0.1                # minimum action_std (stop decay after action_std <= min_action_std)
action_std_decay_freq = int(2.5e5)  # action_std decay frequency (in num timesteps)


#####################################################


## Note : print/log frequencies should be > than max_ep_len


################ PPO hyperparameters ################


update_timestep = max_ep_len * 4      # update policy every n timesteps
K_epochs = 10               # update policy for K epochs
eps_clip = 0.2              # clip parameter for PPO
gamma = 0.995                # discount factor

lr_actor = 0.001       # learning rate for actor network
lr_critic = 0.002       # learning rate for critic network

random_seed = 0         # set random seed if required (0 = no random seed)

#####################################################



print("training environment name : " + env_name)

env = gym.make(env_name)

# state space dimension
state_dim = env.observation_space.shape[0]

# action space dimension
if has_continuous_action_space:
    action_dim = env.action_space.shape[0]
else:
    action_dim = env.action_space.n



###################### logging ######################

#### log files for multiple runs are NOT overwritten

log_dir = "PPO_logs"
if not os.path.exists(log_dir):
      os.makedirs(log_dir)

log_dir = log_dir + '/' + env_name + '/'
if not os.path.exists(log_dir):
      os.makedirs(log_dir)


#### get number of log files in log directory
run_num = 0
current_num_files = next(os.walk(log_dir))[2]
run_num = len(current_num_files)


#### create new log file for each run 
log_f_name = log_dir + '/PPO_' + env_name + "_log_" + str(run_num) + ".csv"

print("current logging run number for " + env_name + " : ", run_num)
print("logging at : " + log_f_name)

#####################################################


################### checkpointing ###################

run_num_pretrained = 2      #### change this to prevent overwriting weights in same env_name folder

directory = "PPO_preTrained"
if not os.path.exists(directory):
      os.makedirs(directory)

directory = directory + '/' + env_name + '/'
if not os.path.exists(directory):
      os.makedirs(directory)


checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
print("save checkpoint path : " + checkpoint_path)

#####################################################


############# print all hyperparameters #############

print("--------------------------------------------------------------------------------------------")

print("max training timesteps : ", max_training_timesteps)
print("max timesteps per episode : ", max_ep_len)

print("model saving frequency : " + str(save_model_freq) + " timesteps")
print("log frequency : " + str(log_freq) + " timesteps")
print("printing average reward over episodes in last : " + str(print_freq) + " timesteps")

print("--------------------------------------------------------------------------------------------")

print("state space dimension : ", state_dim)
print("action space dimension : ", action_dim)

print("--------------------------------------------------------------------------------------------")

if has_continuous_action_space:
    print("Initializing a continuous action space policy")
    print("--------------------------------------------------------------------------------------------")
    print("starting std of action distribution : ", action_std)
    print("decay rate of std of action distribution : ", action_std_decay_rate)
    print("minimum std of action distribution : ", min_action_std)
    print("decay frequency of std of action distribution : " + str(action_std_decay_freq) + " timesteps")

else:
    print("Initializing a discrete action space policy")

print("--------------------------------------------------------------------------------------------")

print("PPO update frequency : " + str(update_timestep) + " timesteps") 
print("PPO K epochs : ", K_epochs)
print("PPO epsilon clip : ", eps_clip)
print("discount factor (gamma) : ", gamma)

print("--------------------------------------------------------------------------------------------")

print("optimizer learning rate actor : ", lr_actor)
print("optimizer learning rate critic : ", lr_critic)

if random_seed:
    print("--------------------------------------------------------------------------------------------")
    print("setting random seed to ", random_seed)
    torch.manual_seed(random_seed)
    env.seed(random_seed)
    np.random.seed(random_seed)

#####################################################

print("============================================================================================")

################# training procedure ################

# initialize a PPO agent
ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)

#load checkpoint
if(os.path.exists(checkpoint_path)):
    ppo_agent.load(checkpoint_path)

# track total training time
start_time = datetime.now().replace(microsecond=0)
print("Started training at (GMT) : ", start_time)

print("============================================================================================")


# logging file
log_f = open(log_f_name,"w+")
log_f.write('episode,timestep,reward\n')


# printing and logging variables
print_running_reward = 0
print_running_episodes = 0

log_running_reward = 0
log_running_episodes = 0

time_step = 0
i_episode = 0


# training loop
while time_step <= max_training_timesteps:

    print('state: '+str(len(ppo_agent.buffer.states)))
    print('action: '+str(len(ppo_agent.buffer.actions))) 
    print('ac_log: '+str(len(ppo_agent.buffer.logprobs))) 
    print('reward: '+str(len(ppo_agent.buffer.rewards)))
    print('terminals: '+str(len(ppo_agent.buffer.is_terminals)))
        
    state = env.reset()
    current_ep_reward = 0

    for t in range(1, max_ep_len+1):
        
        # select action with policy
        action = ppo_agent.select_action(state)
        state, reward, done, _ = env.step(action)
        
        # saving reward and is_terminals
        ppo_agent.buffer.rewards.append(reward)
        ppo_agent.buffer.is_terminals.append(done)
        
        time_step +=1
        current_ep_reward += reward

        # update PPO agent
        if time_step % update_timestep == 0:
            ppo_agent.update()

        # if continuous action space; then decay action std of ouput action distribution
        if has_continuous_action_space and time_step % action_std_decay_freq == 0:
            ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)

        # log in logging file
        if time_step % log_freq == 0:

            # log average reward till last episode
            log_avg_reward = log_running_reward / log_running_episodes
            log_avg_reward = round(log_avg_reward, 4)

            log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
            log_f.flush()

            log_running_reward = 0
            log_running_episodes = 0

        # printing average reward
        if time_step % print_freq == 0:

            # print average reward till last episode
            print_avg_reward = print_running_reward / print_running_episodes
            print_avg_reward = round(print_avg_reward, 2)
            
            print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step, print_avg_reward))
            if(print_avg_reward!=-200.0):
                ppo_agent.save(directory+'good_weight.pth')
            print_running_reward = 0
            print_running_episodes = 0

            

            
        # save model weights
        if time_step % save_model_freq == 0:
            print("--------------------------------------------------------------------------------------------")
            print("saving model at : " + checkpoint_path)
            ppo_agent.save(checkpoint_path)
            print("model saved")
            print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
            print("--------------------------------------------------------------------------------------------")
            
        # break; if the episode is over
        if done:
            break

    print_running_reward += current_ep_reward
    print_running_episodes += 1

    log_running_reward += current_ep_reward
    log_running_episodes += 1

    i_episode += 1


log_f.close()
env.close()




# print total training time
print("============================================================================================")
end_time = datetime.now().replace(microsecond=0)
print("Started training at (GMT) : ", start_time)
print("Finished training at (GMT) : ", end_time)
print("Total training time  : ", end_time - start_time)
print("============================================================================================")



'''

