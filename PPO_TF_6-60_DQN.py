#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import PPO_A2C 
#import Dual_Connect_2 as Dual_Connect
import DQN_40_B as PPO_A2C_SEQ
import STHD_1 as Dual_Connect

import pandas as pd
import multiprocessing as mp
import datetime
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import os
import torch

from typing import Type


# In[2]:


class PPO_TF_HO_Algo(Dual_Connect.Handover):
    
    #def init_connect_cell(self, RL_Agent:Type[PPO_A2C_SEQ.PPO], RL_input, delay_buffer,\
    #                      graph_edge, node_features, new_graph_edge, num_of_UE, num_of_micro):
    def init_connect_cell(self, RL_Agent:Type[PPO_A2C_SEQ.PPO], RL_input, num_of_UE, num_of_st, angle_time_table,                         servering_st, c_st, E):
        
        done = False
        # input()
        action = RL_Agent.select_action(RL_input, self.time, num_of_UE, num_of_st, c_st ,E)
        
        

        self.s_cell['cell_type'][0] = 'ST'
        self.s_cell['cell_ID'][0] = action
        self.angle=angle_time_table[self.time][action]
        self.ini = True
        RL_Agent.buffer.is_terminals.append(done)
        
        servering_st[action]+=1

        
        
        #RL_Agent.buffer.is_terminals.append(done)
        #RL_Agent.buffer.UE_ID.append(self.ID)
        #RL_Agent.buffer.UE_ID.append(self.ID)
        #delay_buffer.append('S')
        #delay_buffer.append('S')    
        #self.A2_start = 1
        #self.A2_index = len(delay_buffer)-1
        #print(Rank) 
        
        '''
        for i in range(len(Rank)):
            
            self.s_cell['cell_type'][0]='micro'
            self.s_cell['cell_ID'][0]=action_1
            if action_2 != num_of_micro:

                self.s_cell['cell_type'][1]='micro'
                self.s_cell['cell_ID'][1]=action_2
            else:
                self.s_cell['cell_type'][1]=[]
                self.s_cell['cell_ID'][1]=[]
            #print(self.s_cell['cell_ID'][i] )
            self.s_cell['SINR'][i] = \
            float(self.df[str(self.s_cell['cell_type'][i]) + '_' + str(self.s_cell['cell_ID'][i]) + '_SINR'][self.time])
            self.s_cell['RSRP'][i] = \
            float(self.df[str(self.s_cell['cell_type'][i]) + '_' + str(self.s_cell['cell_ID'][i]) + '_L3_RSRP'][self.time])
            self.connection[i] = 1
            
            ##print('cell_ID '+str(self.s_cell['cell_ID'][i]))
            ##print('ID '+ str(self.ID))
            graph_edge[int(self.ID)][int(self.s_cell['cell_ID'][i])]=1
            se = float(self.df[str(self.s_cell['cell_type'][i]) +\
                '_' + str(self.s_cell['cell_ID'][i])+'_SINR'][self.time])
                    
            node_features[int(self.ID)][int(self.s_cell['cell_ID'][i])]=(Dual_Connect.mcs_rule(se))/5.555
        #print('----------------------')
        #print('UE ini :'+str(self.ID))
        #print('UE s :'+str(self.s_cell['cell_ID']))
        #print('new_g ' +str(graph_edge[self.ID]))
        #print('new_f ' +str( node_features[self.ID]))
        #print('need_reward'+str(self.A2_index))
        self.position_x = self.df['position_x'][self.time]
        self.position_y = self.df['position_y'][self.time]
        self.ini = True
        
        #print('______________')
        #print(self.ID)
        #print('ini index')
        #print(self.A2_index)
        #print(self.s_cell['cell_ID'])
        #print('______________')
        
        
        #print(str(self.ID)+ 'ini')
        '''
    #def TF_handover(self, RL_Agent:Type[PPO_A2C_SEQ.PPO], RL_input, delay_buffer,\
    #                      graph_edge, node_features, new_graph_edge, num_of_UE, num_of_micro):
    
    
    def TF_handover(self, RL_Agent:Type[PPO_A2C_SEQ.PPO], RL_input, num_of_UE, num_of_st, servering_st, c_st, E):
        
        done = False
        
        
        
        #action = RL_Agent.select_action(RL_input, self.time_step, th, graph_edge, node_features, new_graph_edge,\
        #                                                        num_of_UE, num_of_micro)
          
        action = RL_Agent.select_action(RL_input, self.time, num_of_UE, num_of_st, c_st, E)
        
        #print('action')
        #print(action)
        
        if action !=  self.s_cell['cell_ID'][0] :
            
            self.HO_count += 1
            
            servering_st[self.s_cell['cell_ID'][0]]-=1
            
            self.s_cell['cell_ID'][0] = action

            servering_st[action]+=1
        
        
     
        
        
        
        HO_state = 'M'
        
        #self.t_cell['cell_type'][0] = 'ST'
        #self.t_cell['cell_ID'][0] = action

        #HO_state = 'H_R'      

        #if(self.time == 600-1):
        #    done = True
            
        RL_Agent.buffer.is_terminals.append(done)
            
            
            
        return HO_state
            

        #return HO_state, A2_event
        
    
    #def handover(self,RL_Agent, RL_input, delay_buffer, graph_edge, node_features,\
    #             new_graph_edge, num_of_UE, num_of_micro):

    def handover(self,RL_Agent, RL_input, num_of_UE, num_of_st, servering_st,c_st,E):
        
        #A2_Event=False
        HO_state = self.HO_state

        #self.All_cell_connect_test()     

        #if self.RLF == 1:

        #    self.RLF_reset()
        #    self.RLF_count +=1

        #else:

        if HO_state == 'M':
                    
            #HO_state,A2_Event = self.TF_handover(RL_Agent, RL_input, delay_buffer,\
            #                    graph_edge, node_features, new_graph_edge, num_of_UE, num_of_micro)               
            HO_state = self.TF_handover(RL_Agent, RL_input, num_of_UE, num_of_st, servering_st, c_st, E)                

        #elif HO_state == 'H_R':
        #    HO_state = self.HO_success_process()

        #elif HO_state == 'H_R' and self.s_cell['cell_ID'][0] != [] or self.s_cell['cell_ID'][1] != []:
        #    HO_state = self.HO_success_process(graph_edge,node_features)


        else:
            print('error')
            
        self.HO_state = HO_state

     


        


# In[3]:


def UE_data_to_csv(case ,em_name, algo_name ,num_ue):
    
    path = case + '_'+ em_name + '_' + algo_name +'_'+ num_ue
    folder = os.path.exists(path)    
    if not folder:       
        os.makedirs(path)
        print('Done Path')
    else:        
        print( path + ' is exist')


# In[4]:


'''
Timeslot=200
Carrier_bandwidth=200
Num_of_UE=20
Demand=50
Num_of_macro=0
Num_of_micro=7
Time=900
'''
################################## set device ##################################

print("============================================================================================")


# set device to cpu or cuda
ppo_device = torch.device('cpu')

if(torch.cuda.is_available()): 
    ppo_device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(ppo_device)))
else:
    print("Device set to : cpu")
    
print("============================================================================================")

'''
cell_num=7
K_epochs = 15               # update policy for K epochs
eps_clip = 0.2              # clip parameter for PPO
gamma = 0.99                 # discount factor
lr_model = 0.001       # learning rate for actor network
lr_critic = 0.002       # learning rate for critic network
max_len=3               # max output length
embbeding_dim=24        # embbeding dimension
batch_size = 2048
t_agent = Transformer_PPO.PPO( cell_num, lr_model, lr_critic, gamma, K_epochs, eps_clip, 
                              max_len,ppo_device
                              , embbeding_dim)
'''
'''
Num_of_UE=100
Demand=20
'''


Num_of_st=298
Num_of_UE=10
Demand=50
#119+18 = 137
#168+18
#70+18
#200+11+3+11
#100+21+3+21+1
#[cover, load, v_time, action space]
state_dim = Num_of_st*3+Num_of_st
action_dim = 22
lr_actor = 0.001
lr_critic = 0.001
#gamma = 0.9
gamma = 0
K_epochs = 100
has_continuous_action_space=False
eps_clip = 0.2 
action_std = 0.6  
device = ppo_device
test = False
time_step = 0
re_mean =0
re_std =0

t_agent = PPO_A2C_SEQ.PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, 
                K_epochs, eps_clip, has_continuous_action_space, action_std, \
                           device, test, time_step, re_mean, re_std, num_ue =Num_of_UE  )


# In[5]:


################### checkpointing ###################
run_num_pretrained=1

directory = "PPO_preTrained"
if not os.path.exists(directory):
      os.makedirs(directory)
em = 'Eichstatt'
algo = 'PPO_TF'
case = 'Result'
env_name=em+'_'+algo+'_'+case
directory = directory + '/' + env_name + '/'
if not os.path.exists(directory):
      os.makedirs(directory)


checkpoint_path = directory + "PPO_{}_{}.pth".format(env_name, run_num_pretrained)
print("save checkpoint path : " + checkpoint_path)
#reward_table_checkpoint_path=directory + "reward_table{}_{}.pth".format(env_name, run_num_pretrained)
#print("save reward_table checkpoint path : " + reward_table_checkpoint_path)

#critic_checkpoint_path=directory + "critic_{}_{}.pth".format(env_name, run_num_pretrained)
#print("save critic checkpoint path : " + critic_checkpoint_path)


# In[9]:


Timeslot=1000
Carrier_bandwidth=200
Time=3600
#Time=1000
power = 35
Num_of_UE=30
s= 2
#s=1, Max_serving_time
#s=2, Max_node_capacity
#s=3, Random_ST
#s=4, graph
num_of_channel =5
e_greedy =0.3
#e_greedy = 1  #always random
# training part
time_step=0
max_training_timesteps=5
update_timestep=1
save_model_freq=10

start_time = datetime.datetime.now().replace(microsecond=0)


qv =[]

print("Started training at (GMT) : ", start_time)

print("============================================================================================")
#load checkpoint
if(os.path.exists(checkpoint_path)):
    t_agent.load(checkpoint_path)
    print("model loaded")
    input()
    #t_agent.load(checkpoint_path,reward_table_checkpoint_path,critic_checkpoint_path)
#
Rand = 1

# t_agent = None
while time_step <= max_training_timesteps:
    #Rand = random.randint(1,10) 
    # if time_step == 50:
    #     e_greedy-=0.1
    # if time_step == 100:
    #     e_greedy-=0.1
    if time_step == 150:
        e_greedy = 0
    print('random'+str(Rand))
    Dual_Connect.sim(algo=PPO_TF_HO_Algo,algo_name='PPO_TF_HO_Algo',timeslot=Timeslot,                    carrier_bandwidth=Carrier_bandwidth,
                    num_of_UE=Num_of_UE,demand=Demand,num_of_st=Num_of_st,\
                    time=Time, RL_Agent=t_agent, time_step = time_step, power = power,\
                    time_th=max_training_timesteps/2, R =Rand, strategy=s, Per=num_of_channel,E= e_greedy)
    

    
    
    time_step +=1
    if t_agent!=None:
        t_agent.time_step = time_step
    if time_step>0 and time_step% save_model_freq == 0 and t_agent!=None:
        print("--------------------------------------------------------------------------------------------")
        print("saving model at : " + checkpoint_path)
        t_agent.save(checkpoint_path)
    '''
    #qv.append(fq)
    
    print('time_step:' +str(t_agent.time_step ))
    
    
    if time_step>0 and time_step % update_timestep == 0:
        print('Updating...')
        #t_agent.update()
    if (time_step>0 and time_step% save_model_freq) == 0:
        print("--------------------------------------------------------------------------------------------")
        print("saving model at : " + checkpoint_path)
        t_agent.save(checkpoint_path)
        
        #print("saving model at : " + 
        #      checkpoint_path, reward_table_checkpoint_path, critic_checkpoint_path)
        #t_agent.save(checkpoint_path, reward_table_checkpoint_path, critic_checkpoint_path)
        print("model saved")
        print("Elapsed Time  : ", datetime.datetime.now().replace(microsecond=0) - start_time)
        print("--------------------------------------------------------------------------------------------")
    '''
    
        


# In[ ]:


for i in range(0,6):
    print(t_agent.buffer.rewards[i])

# for i in t_agent.buffer.rewards:
#     print(i)


# In[ ]:


'''Eichstatt_900_sec_100_slot_1_UE_4
Timeslot=200
Carrier_bandwidth=200

state

Demand=20
Num_of_macro=0
Num_of_micro=7
Time=900


Num_of_UE=10
Demand=200

Num_of_UE=100
Demand=20
'''
'''
state_dim = 186
#state_dim = 17
action_dim = 8
lr_actor = 0.003  
lr_critic = 0.001
gamma = 0.95
K_epochs = 15 
has_continuous_action_space=False
eps_clip = 0.2 
action_std = 0.6  
device = ppo_device
test = False
#Rand = random.randint(1,10)


if(os.path.exists(checkpoint_path)):
    
    t_agent.load(checkpoint_path)
    #t_agent.load(checkpoint_path,reward_table_checkpoint_path,critic_checkpoint_path)

for i in range(1):
    #power+=5
    
    print(Rand)
    print(Num_of_UE)
    lst_time,lst_QoS_1 = Dual_Connect.sim(algo=PPO_TF_HO_Algo,algo_name='PPO_TF_HO_Algo',timeslot=Timeslot,\
                    carrier_bandwidth=Carrier_bandwidth,
                    num_of_UE=Num_of_UE,demand=Demand,num_of_macro=Num_of_macro,\
                    num_of_micro=Num_of_micro,time=Time,RL_Agent=t_agent,\
                    time_step = time_step, power = power, time_th=max_training_timesteps/2,R =Rand)

    
    t_agent.buffer.clear()
    lst_time,lst_QoS_2 = Dual_Connect.sim(algo=Dual_Connect.Handover,algo_name='max_sinr',timeslot=Timeslot,\
                        carrier_bandwidth=Carrier_bandwidth,
                        num_of_UE=Num_of_UE,demand=Demand,num_of_macro=Num_of_macro,\
                        num_of_micro=Num_of_micro,time=Time, time_step = time_step,\
                                        time_th=max_training_timesteps/2,power = power, R=Rand )
'''

