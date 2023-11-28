#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import multiprocessing as mp
import datetime
import math
import random
import numpy as np
import os
import csv
#import Dual_Connect_2 as Dual_Connect
#import REBL


class Handover:

    def __init__(self, ID=0, algo_name=None, time=0, hom=3, threshold=-50, data_frame=None, ini_connect=1,  
                 position_x='N', position_y='N', HO_state='M', stage=None, Rank=None, connection=None,  
                 action=None,    s_cell=None, t_cell=None, num_connect=1, RLF=0, T310=None, T310_counter=None, 
                 size_of_timeslot=40, RLF_count=0, HO_count=0, ini=False, slot_thr=None, Data_rate=None, 
                 resource_block=None, num_of_st=298, demand=40, QoS='N', A2_th=-86,
                 Trigger_count=0, A2_reward_1=0, A2_reward_2=0,  A2_time=0, A2_start=0, A2_index='N', 
                 A2_index_old = 'N', A2_event=False, time_step = 0, time_th =0, angle = 1.2, st=2 ):

        assert algo_name != None, 'Error : mission hom_algo'

        self.ID = ID
        self.time = time
        self.hom = hom
        self.threshold = threshold
        self.algo_name = algo_name
        #self.position_x = position_x
        #self.position_y = position_y
        self.HO_state = HO_state
        self.num_connect = num_connect
        self.RLF = RLF
        self.ini_connect = ini_connect
        #self.T310 = T310
        #self.T310_counter = T310_counter
        self.RLF_count = RLF_count
        self.HO_count = HO_count
        self.df = data_frame
        self.ini = ini
        self.demand = demand*((self.ID%3)+1) 
        self.QoS = QoS
        #self.num_of_micro = num_of_micro
        #self.num_of_macro = num_of_macro
        self.num_of_st = num_of_st
        self.size_of_timeslot = size_of_timeslot
        self.A2_th = A2_th
        self.Trigger_count = Trigger_count
        self.A2_reward_1= A2_reward_1
        self.A2_reward_2= A2_reward_2
        self.A2_time= A2_time
        self.A2_start= A2_start
        self.A2_index= A2_index
        self.A2_index_old = A2_index_old
        self.A2_event =A2_event
        self.time_step = time_step
        self.time_th = time_th 
        self.min_angle = angle
        self.angle = -1
        self.Max_st =st

        if Rank == None:
            self.Rank = []

        if T310 == None:
            T310 = []
            for i in range(self.num_connect):
                T310.append(0)
            self.T310 = T310

        if slot_thr == None:
            slot_thr = []
            for i in range(self.num_connect):
                slot_thr.append([])
            self.slot_thr = slot_thr
            
        if action == None:
            action = []
            for i in range(self.num_connect):
                action.append([])
            self.action = action

        if Data_rate == None:
            Data_rate = []
            for i in range(num_of_st):
                Data_rate.append(0)
            self.Data_rate = Data_rate

        if T310_counter == None:
            T310_counter = []
            for i in range(self.num_connect):
                T310_counter.append(0)
            self.T310_counter = T310_counter

        if connection == None:
            connection = []
            for i in range(self.num_connect):
                connection.append([])
            self.connection = connection

        if stage == None:
            stage = []
            for i in range(self.num_connect):
                stage.append(1)

            self.stage = stage

        if resource_block == None:
            resource_block = []
            for i in range(self.num_connect):
                resource_block.append([])
                for j in range(int(size_of_timeslot)):
                    resource_block[i].append([])

            self.resource_block = resource_block

        if t_cell == None:
            self.t_cell = {'cell_type': [],
                           'cell_ID': [], 'RSRP': [], 'SINR': []}
            for i in range(self.num_connect):
                self.t_cell['cell_type'].append([])
                self.t_cell['cell_ID'].append([])
                self.t_cell['RSRP'].append([])
                self.t_cell['SINR'].append([])
      

        if s_cell == None:
            self.s_cell = {'cell_type': [],
                           'cell_ID': [], 'RSRP': [], 'SINR': []}
            for i in range(self.num_connect):
                self.s_cell['cell_type'].append([])
                self.s_cell['cell_ID'].append([])
                self.s_cell['RSRP'].append([])
                self.s_cell['SINR'].append([])

    def init_connect_cell(self, ST_serving_table, servering_st, angle_time_table, con_time_table, nc):

        #HO_state = self.HO_measure()
        #HO_state = self.HO_Request_process()
        #HO_state = self.HO_transfer_process()

        #HO_state = self.HO_success_process()
        
        if self.Max_st ==1:
            

            ID = Max_serving_time(ST_serving_table,self.time)
            
    

            
        elif self.Max_st ==2:
            
            if self.s_cell['cell_ID'][0]!= []:
                st = self.s_cell['cell_ID'][0]
            else:
                st = -1
                    
            #ID = Max_node_capacity(servering_st, self.time, self.num_of_st, con_time_table, angle_time_table, st)
            ID = Max_node_capacity(ST_serving_table,self.time, servering_st ,nc)
            
        elif self.Max_st ==3:
                    
            ID = Random_ST(ST_serving_table,self.time )
            
        
        elif self.Max_st ==4:
                    
            ID = graph(ST_serving_table,self.time, servering_st ,nc)
            

            
            
        
        

        '''
        for i in ['macro', 'micro']:

            if i == 'macro':
                num_of_cell = self.num_of_macro
            else:
                num_of_cell = self.num_of_micro

            SINR_list = {}
            SINR_rank = {}
            for j in range(num_of_cell):

                SINR_list[str(
                    i) + '_' + str(j)] = float(self.df[str(i) + '_' + str(j) + '_SINR'][self.time])

            SINR_measure = dict(
                sorted(SINR_list.items(), key=lambda item: item[1]))

        for key, value in SINR_measure.items():

            # print('key'+str(key))
            # print('value'+str(value))

            self.Rank.append(key)
        self.Rank.reverse()
        self.Rank = self.Rank[:self.num_connect]
        # print(self.Rank)
        '''
        
        #for i in range(self.ini_connect):
        #print(ID)
        self.s_cell['cell_type'][0] = 'ST'
        self.s_cell['cell_ID'][0] = ID
        self.angle=angle_time_table[self.time][ID]
        
        #self.s_cell['SINR'][i] = float(self.df[str(
        #    self.s_cell['cell_type'][i]) + '_' + str(self.s_cell['cell_ID'][i]) + '_SINR'][self.time])
        #self.s_cell['RSRP'][i] = float(self.df[str(
        #    self.s_cell['cell_type'][i]) + '_' + str(self.s_cell['cell_ID'][i]) + '_L3_RSRP'][self.time])
        self.connection[0] = 1

        #while len(self.s_cell['cell_ID']) < self.num_connect:
        #    self.s_cell['cell_type'].append([])
        #    self.s_cell['cell_ID'].append([])

        #self.position_x = self.df['position_x'][self.time]
        #self.position_y = self.df['position_y'][self.time]
        self.ini = True
        # print(self.s_cell['cell_type'][i])
        # print(self.s_cell['cell_ID'][i])

    def RLF_reset(self):
        # print('reset')

        self.init_connect_cell()
        self.RLF = 0
        self.T310 = 0
        self.T310_counter = 0

        stage = []
        for i in range(self.num_connect):
            stage.append(1)
        self.stage = stage

    def Data_rate_sec_cal(self):
        if self.time % 1 == 0 and self.position_x != 'N':
            
            for i in range(self.num_of_micro):
                self.Data_rate[i] = 0
            
            for i in range(self.num_connect):
                if self.slot_thr[i] != []:
                    self.Data_rate[int(self.s_cell['cell_ID'][i])] = self.slot_thr[i]
                       
        #print(self.Data_rate)

    def QoS_sec_cal(self):

        if self.time % 1 == 0 and self.position_x != 'N':
            total = 0
            for i in range(self.num_of_micro):
                total += self.Data_rate[i]

            if total/self.demand >= 1:
                self.QoS = 1
            else:
                self.QoS = total/self.demand
        else:
            pass
        
    def cal_A2_cumul_reward(self):
        
        
        if self.A2_reward_1 !=0 and self.A2_reward_2!=0 and self.A2_time !=0:
        
            cumul_reward_1 = ((self.A2_reward_1/ self.A2_time) )-0.7
            #cumul_reward_1 = self.A2_reward_1

            #cumul_reward = self.A2_reward
            cumul_reward_2 = ((self.A2_reward_2 /self.A2_time) )-0.7
            #cumul_reward_2 = self.A2_reward_2 
        else:
            cumul_reward_1 ='N'
            cumul_reward_2 ='N'
            buffer_ind_old ='N'
            buffer_ind_new ='N'
            
        
        if self.A2_index_old == 'N'and self.time == (600-1):
            buffer_ind_old = int(self.A2_index)
            buffer_ind_new = 'N'
            #print('N ' + str(self.A2_index))
            #print(self.time)
            
        elif self.A2_index_old != 'N'and self.time == (600-1):
            buffer_ind_old = int(self.A2_index)
            buffer_ind_new = 'N'
            
        elif self.A2_index_old =='N'and self.time != (600-1):
            buffer_ind_old = int(self.A2_index)
            buffer_ind_new = 'N'
            print('Error do start handover')
            
        else:
            buffer_ind_old = int(self.A2_index_old)
            buffer_ind_new = int(self.A2_index)
        
        return cumul_reward_1, cumul_reward_2, buffer_ind_old, buffer_ind_new
    
    
    def A2_cumul(self,avg_QoS):
        
        #self.A2_reward+= (avg_QoS - REBL.reward_list(int(self.time-1)))
        #print('ID'+str(self.ID))
        a=0
        b= 0
        if self.s_cell['cell_ID'][0] != []:
            a = self.Data_rate[int(self.s_cell['cell_ID'][0])]
            
        #print(self.s_cell['cell_ID'][1])
        
        if self.s_cell['cell_ID'][1] != []:    
            b = self.Data_rate[int(self.s_cell['cell_ID'][1])]
        #print(str(a)+str(b))
        
        #self.A2_reward_1+= (avg_QoS)*(a/(a+b))
        #self.A2_reward_2+= (avg_QoS)*(b/(a+b))

        self.A2_reward_1+= (avg_QoS)
        self.A2_reward_2+= (avg_QoS)
       
        #self.A2_reward+= (avg_QoS)
        self.A2_time +=1
        #self.A2_time+= 0.2
        
        
    def A2_ini(self):
                
        self.A2_reward_1 = 0
        self.A2_reward_2 = 0                          
        self.A2_time = 0
        #self.A2_index_old = 0
        self.A2_start = 1
        
    
    def RLF_test_MT(self, i):
        # print(self.s_cell_type,self.s_ID,self.time)

        # print(str(self.s_cell['cell_type']))
        # print(str(self.s_cell['cell_ID']))
        #print(type(self.df[str(self.s_cell['cell_type'][i]) + '_' + str(self.s_cell['cell_ID'][i]) + '_SINR' ][self.time - 1 ]))

        if float(self.df[str(self.s_cell['cell_type'][i]) + '_' + str(self.s_cell['cell_ID'][i]) + '_SINR'][self.time]) <= -8:

            self.T310[i] += 1

            self.T310_counter[i] += 1

        elif self.T310_counter[i] != 0:

            self.T310_counter[i] += 1

        else:
            pass

        if self.T310_counter[i] == 25 and float(self.df[str(self.s_cell['cell_type'][i]) + '_' + str(self.s_cell['cell_ID'][i]) + '_SINR'][self.time]) <= -6:

            self.connection[i] = 1

        elif self.T310[i] == 25:

            self.connection[i] = 1

        elif self.T310_counter[i] >= 25 and self.T310[i] < 25:

            self.T310[i] = 0

            self.T310_counter[i] = 0

        else:

            pass

        return None

    def RLF_test_prepare(self, i):

        if self.connection[i] == 0:

            #print('connection[i]=',0 )

            pass

        elif self.s_cell['cell_ID'][i] != []:

            #print('self.s_cell != []:')

            if self.T310 == 25 or float(self.df[str(self.s_cell['cell_type'][i]) + '_' + str(self.s_cell['cell_ID'][i]) + '_SINR'][self.time]) <= -8:

                # self.RLF = 1

                # HO_state = ''

                self.connection[i] == 0

                #self.stage[i] = 0

            elif self.stage[i] == 1:

                self.stage[i] = 2

                #HO_state = 'H_R'

            elif self.stage[i] == 2:

                self.stage[i] = 3

        elif self.s_cell['cell_ID'][i] == []:

            #print('self.s_cell[i] == []')

            if self.stage[i] == 1:

                self.stage[i] = 2

                #HO_state = 'H_R'

            elif self.stage[i] == 2:

                self.stage[i] = 3
        else:
            print('ERROR')

    def RLF_test_tranfer(self, i):

        if self.connection[i] == 0:

            pass

        elif self.T310 == 25 and float(self.df[str(self.t_cell['cell_type'][i]) + '_' + str(self.t_cell['cell_ID'][i]) + '_SINR'][self.time]) <= -8:

            #self.RLF = 1

            self.connection[i] == 0

            #HO_state = 'H_T'

            #self.stage[i] = 1

        elif self.stage[i] == 1:

            self.stage[i] = 2

            #HO_state = 'H_T'

        elif self.stage[i] == 2:
            self.stage[i] = 3

    def release(self, i):

        if self.stage[i] == 1:

            self.stage[i] = 2

            #HO_state = 'H_T'

        elif self.stage[i] == 2:
            self.stage[i] = 3

    def handover_target(self):

        serve_cell = []
        target = []

        for i in range(self.num_connect):

            s = str(self.s_cell['cell_type'][i]) +                 '_'+str(self.s_cell['cell_ID'][i])
            serve_cell.append(s)

        while len(self.Rank) < self.num_connect:
            self.Rank.append('NT')
           

        for i in serve_cell:

            if (i in self.Rank) == True:
                # print('y')
                target.append('XXXXX_X')
                self.Rank.remove(i)

            elif (i in self.Rank) == False:

                for j in self.Rank:

                    if (j in serve_cell) == True:
                        pass

                    elif (j in serve_cell) == False:

                        if j != 'NT':
                            target.append(j)
                            self.Rank.remove(j)
                            break

                        elif j == 'NT' and i == '[]_[]':
                            target.append('NNNNN_N')
                            self.Rank.remove(j)
                            break

                        elif j == 'NT' and i != '[]_[]':
                            target.append('RRRRR_R')
                            self.Rank.remove(j)
                            break

        for i in range(self.num_connect):
            self.t_cell['cell_type'][i] = target[i][:5]
            self.t_cell['cell_ID'][i] = target[i][6]

        for i in range(self.num_connect):
            if self.t_cell['cell_ID'][i] != 'X' and self.t_cell['cell_ID'][i] != 'N':
                HO_state = 'H_R'
                # print('H_R')
                break
            else:
                HO_state = 'M'

        return HO_state

    def HO_measure(self, ST_serving_table, servering_st, con_table, angle_table, nc):
        
        
        if self.Max_st ==1:
            

            ID = Max_serving_time(ST_serving_table,self.time)
            
        elif self.Max_st ==2:
            
            if self.s_cell['cell_ID'][0]!= []:
                st = self.s_cell['cell_ID'][0]
            else:
                st = -1
                    
                    
            ID = Max_node_capacity(ST_serving_table,self.time, servering_st ,nc )
            
        elif self.Max_st ==3:
                    
            ID = Random_ST(ST_serving_table,self.time )
                      
            
        elif self.Max_st ==4:
                    
            ID = graph(ST_serving_table,self.time, servering_st ,nc)
            

        self.t_cell['cell_type'][0] = 'ST'
        self.t_cell['cell_ID'][0] = ID

        HO_state = 'H_R'
        
        '''
        if 
        A2_enent = False

        #if self.time % 1 == 0 or self.ini == False:

            for i in range(self.num_connect):
                if self.s_cell['RSRP'][i] != 'N' and self.s_cell['RSRP'][i] != []:
                    #print(self.s_cell ['RSRP'][i])
                    if self.s_cell['RSRP'][i] > self.min_angle or self.ini == False:
                        A2_enent = True
                        self.Trigger_count += 1
                        #print(str(self.ID) +'_'+ str(self.Trigger_count))

                        break
                    else:
                        pass

            if A2_enent == True:

                # print('A2_enent')
                count = 0
                for i in range(self.num_connect):
                    if self.Data_rate[i] != 'N':
                        count += 1

                if count > 0:
                    serve_cell = []
                    self.t_cell = {'cell_type': [],
                                   'cell_ID': [], 'RSRP': [], 'SINR': []}

                    for i in range(self.num_connect):
                        self.t_cell['cell_type'].append([])
                        self.t_cell['cell_ID'].append([])
                        self.t_cell['RSRP'].append([])
                        self.t_cell['SINR'].append([])

                    for i in range(len(self.s_cell['cell_ID'])):

                        s = str(self.s_cell['cell_type'][i]) + \
                            '_'+str(self.s_cell['cell_ID'][i])
                        serve_cell.append(s)

                    for i in ['macro', 'micro']:

                        if i == 'macro':
                            num_of_cell = self.num_of_macro
                        else:
                            num_of_cell = self.num_of_micro

                        SINR_list = {}
                        SINR_rank = {}
                        for j in range(num_of_cell):

                            SINR_list[str(
                                i) + '_' + str(j)] = float(self.df[str(i) + '_' + str(j) + '_SINR'][self.time])

                        SINR_measure = dict(
                            sorted(SINR_list.items(), reverse=False, key=lambda item: item[1]))
                        # print(SINR_measure)

                    for key, value in SINR_measure.items():
                        self.Rank.append(key)

                    self.Rank.reverse()

                    total_thr = 0

                    for i in range(self.num_connect):
                        if self.Data_rate[i] != 'N':

                            total_thr += self.Data_rate[i]

                    if total_thr/self.demand < 1:

                        count = 0

                        for i in range(self.num_connect):

                            if self.s_cell['cell_ID'][i] != []:
                                count += 1

                        #count = random.randint(1, 2)
                        self.Rank = self.Rank[:(count+1)]
                        #self.Rank = self.Rank[:(count+1)]

                        if len(self.Rank) > self.num_connect:

                            self.Rank = self.Rank[:self.num_connect]

                        HO_state = self.handover_target()

                    elif total_thr/self.demand >= 1:
                        HO_state = 'M'
                        
  

                else:
                    HO_state = 'M'
            else:

                HO_state = 'M'
        else:

            HO_state = 'M'
        '''
        
        return HO_state

    def E_TTT_test(self):

        if self.s_cell['s_RSRP'] + self.hom <= self.df[str(self.t_cell['cell_type'][0]) + '_' + str(self.t_cell['cell_ID'][0])+'_L3_RSRP'][self.time]:

            HO_state = 'H_R'

            if self.T310 == 5:

                self.RLF = 1

                HO_state = 'M'

            else:
                pass

        else:

            HO_state = 'M'  # does not pass

        return HO_state

    def HO_Request_process(self):

        for i in range(self.num_connect):
            if self.t_cell['cell_ID'][i] != 'X' and self.t_cell['cell_ID'][i] != 'N':

                self.RLF_test_prepare(i)

        for i in range(self.num_connect):

            if self.stage[i] == 3:

                HO_state = 'H_T'

                for j in range(self.num_connect):
                    self.stage[j] = 1
                break

            elif self.stage[i] == 1:

                HO_state = 'H_R'

            elif self.stage[i] == 2:

                HO_state = 'H_R'

        return HO_state

    def HO_transfer_process(self):

        for i in range(self.num_connect):

            if self.t_cell['cell_ID'][i] != 'X' and self.t_cell['cell_ID'][i] != 'N' and self.t_cell['cell_ID'][i] != 'R':

                self.RLF_test_tranfer(i)

            elif self.t_cell['cell_ID'][i] == 'R':

                self.release(i)

        for i in range(self.num_connect):

            if self.stage[i] == 3:

                HO_state = 'H_S'

                for j in range(self.num_connect):
                    self.stage[j] = 1
                break

            elif self.stage[i] == 1:

                HO_state = 'H_T'

            elif self.stage[i] == 2:

                HO_state = 'H_T'

        return HO_state

    def HO_success_process(self):

        HO_state = 'M'
        
        
        self.s_cell['cell_type'][0] = self.t_cell['cell_type'][0]
        self.s_cell['cell_ID'][0] = self.t_cell['cell_ID'][0]
        self.connection[0] = 1
        self.t_cell['cell_ID'][0]=[]
        self.HO_count += 1
        
        #print(self.time)
        #print('Yes')
        #print('upadte')
        #print('server')
        #print(self.s_cell['cell_ID'][0])
        #print('-----------------------')
        
        '''
        #print('h susse')
        #print(self.s_cell['cell_ID'])
        #print(self.t_cell['cell_ID'])

        for i in range(self.num_connect):

            if self.t_cell['cell_ID'][i] == 'R' or self.t_cell['cell_ID'][i] == 'N':
                
                #print('old_g '+str(graph_edge))
                #print(self.s_cell['cell_ID'][i])
                if self.s_cell['cell_ID'][i] != []:
                    graph_edge[int(self.ID)][int(self.s_cell['cell_ID'][i])]=0
                    #print('set zero'+str(self.s_cell['cell_ID'][i]))
                    #graph_edge[int(self.s_cell['cell_ID'][i])][int(self.ID)]=0
                    #print('new_g ' +str(graph_edge))
                self.s_cell['cell_type'][i] = []
                self.s_cell['cell_ID'][i] = []
                self.connection[i] = 0
                

            elif self.t_cell['cell_ID'][i] != 'X':
                
                #print('old_g '+str(graph_edge))
                
                #print(self.s_cell['cell_ID'][i])
                if self.s_cell['cell_ID'][i] != []:
                    #graph_edge[int(self.s_cell['cell_ID'][i])][int(self.ID)]=1
                    graph_edge[int(self.ID)][int(self.s_cell['cell_ID'][i])]=0
                    graph_edge[int(self.ID)][int(self.t_cell['cell_ID'][i])]=1
                    #print('release' +str(int(self.s_cell['cell_ID'][i])))
                    #print('id '+str(self.ID))
                    #print('new_g ' +str(graph_edge))
                self.s_cell['cell_type'][i] = self.t_cell['cell_type'][i]
                self.s_cell['cell_ID'][i] = self.t_cell['cell_ID'][i]
                self.connection[i] = 1
                
                #print('add ' + str(self.s_cell['cell_ID'][i]))
                #print(self.s_cell['cell_ID'])
                #graph_edge[int(self.s_cell['cell_ID'][i])][int(self.ID)]=1
                #print('add ' + str(int(self.s_cell['cell_ID'][i])))
                #print('id '+str(self.ID))
                #print('new_g ' +str(graph_edge))

                self.HO_count += 1

            elif self.t_cell['cell_ID'][i] == 'X': #X NO CHANGE CELL 
                pass
        '''
        #print('handover_UE')
        #print(self.ID)
        #print(graph_edge)
        
        return HO_state

    def All_cell_connect_test(self):
        #HO_state = self.HO_state

        for i in range(self.num_connect):

            if self.s_cell['cell_ID'][i] != []:

                self.RLF_test_MT(i)

            elif self.s_cell['cell_ID'][i] == []:

                pass

        for i in range(self.num_connect):
            if self.s_cell['cell_ID'][i] != []:

                if self.connection[i] == 0:
                    self.RLF = 1

                elif self.connection[i] == 1:
                    self.RLF = 0
                    break
                else:
                    pass

            elif self.s_cell['cell_ID'][i] == []:

                pass

    def Update_UE_information(self,angle_table):
        #print(self.time)
        #print(self.angle)
        
        self.angle = angle_table[self.time][self.s_cell['cell_ID'][0]]
        
        '''
        for i in range(self.num_connect):

            if self.s_cell['cell_ID'][i] != []:
                self.s_cell['RSRP'][i] = float(self.df[str(
                    self.s_cell['cell_type'][i]) + '_' + str(self.s_cell['cell_ID'][i])+'_L3_RSRP'][self.time])
                self.s_cell['SINR'][i] = float(self.df[str(
                    self.s_cell['cell_type'][i]) + '_' + str(self.s_cell['cell_ID'][i])+'_SINR'][self.time])
            elif self.s_cell['cell_ID'][i] == []:
                self.s_cell['RSRP'][i] = []
                self.s_cell['SINR'][i] = []

        #self.position_x = self.df['position_x'][self.time]
        #self.position_y = self.df['position_y'][self.time]
        '''
        
    def handover(self, v_time_table, serving_st, con_table, angle_table, nc ):

        # print('A3_HO_start')
        HO_state = self.HO_state

        #self.All_cell_connect_test()

        #if self.RLF == 1:

        #    self.RLF_reset()
        #    self.RLF_count += 1
            #print('stuck at rlf')

        #else:

            # print('start_HO_test')

        if HO_state == 'M':
            HO_state = self.HO_measure( v_time_table, serving_st, con_table, angle_table, nc)
            #HO_state = self.HO_Request_process()
            #HO_state = self.HO_transfer_process()
        elif HO_state == 'H_R' :#and self.s_cell['cell_ID'][0] != [] and self.s_cell['cell_ID'][1] != []:
            HO_state = self.HO_success_process()

        #elif HO_state == 'H_R':
        #    HO_state = 'M'

            # print('ho')

        else:
            print('error')
            '''
            if HO_state == 'M':                        
                HO_state = self.HO_measure()                 
           
            elif HO_state == 'H_R':
                HO_state = self.HO_Request_process()
                
            elif HO_state == 'H_T':
                HO_state = self.HO_transfer_process()
                
            elif HO_state == 'H_S':
                HO_state = self.HO_success_process()
            '''

        #self.Update_UE_information(angle_table)
        self.HO_state = HO_state


# In[4]:


def calc_num_of_ue_in_cell(obj_UE, num_of_UE, num_of_st, node_capacity):
    num_of_ue_in_cell = []
    block_rate=0
    for i in range(num_of_st):
        num_of_ue_in_cell.append(0)

    for i in range(num_of_UE):
        for j in obj_UE[i].s_cell['cell_ID']:
            print("int or list",j)
            # for j in multi_process_list[i].s_cell['cell_ID']:
            if j == []:
                print("not serving")
                # pass
            else:
                print("serving cell for ",i , "is", int(j))
                num_of_ue_in_cell[int(j)] += 1
                
            
    #print(num_of_ue_in_cell)
    
    num_of_cell_block = []
    
    block_count = 0

    max_block_rate = 0

    block_rates = []
    

    for i in range(num_of_st):
        
        if num_of_ue_in_cell[i]!=0:
            block_count+=1
    
    for i in range(num_of_st):
        if num_of_ue_in_cell[i]>node_capacity:
            
            num_of_cell_block.append(1)
            
            #block_count +=1
            block_rate+=(num_of_ue_in_cell[i]-node_capacity)/node_capacity
            block_rates.append((num_of_ue_in_cell[i]-node_capacity)/node_capacity)
            #print(num_of_ue_in_cell[i])
            #print(node_capacity)
            
            #print(block_rate)
        else:
            
            
            num_of_cell_block.append(0)
            pass
    #print('-=-=-=-')
    #print(block_rate)
    # print("block_rate",block_rate)
    # print("block_count",block_count)
    if block_count!= 0:
        block_rate =block_rate/block_count
    else:
        block_rate =0
    if len(block_rates)!= 0:
        max_block_rate = max(block_rates)
    else:
        max_block_rate =0
        #block_rate =block_rate/num_of_st
    
    return num_of_ue_in_cell, block_rate, num_of_cell_block, max_block_rate

def Rewards_cal(obj_UE, num_of_UE, num_of_cell_block,block_rate,L, connect_table, v_table, time, RL_Agent):

    st_ID= obj_UE.s_cell['cell_ID'][0]
    
    # print(num_of_cell_block)
    # print('-----')
    # print(st_ID)
    
    if num_of_cell_block[st_ID]==1:
        
        #reward = -10
        BR= block_rate
        reward = -((BR)/v_table[time][st_ID])*1000
        # print("bR",BR)
        # print("v_table[time][st_ID]",v_table[time][st_ID])
        # print("overload and the reword = ",reward)
        
    elif num_of_cell_block[st_ID]==0:   
        
        if connect_table[time][st_ID] ==0:
            
            reward = -20
        
        elif connect_table[time][st_ID]==1: 
            
            reward = v_table[time][st_ID]
        else:
            print('v or cover_ERROR')
    else:
        print('connect Not 0 or 1')
        
    RL_Agent.buffer.rewards.append(reward)


# In[5]:


def Allow_Bandwidth(carrier_bandwidth, num_of_ue_in_cell, num_of_micro):
    Avg_BW = []
    for i in range(num_of_micro):
        if num_of_ue_in_cell[i] != 0:
            Avg_BW.append(carrier_bandwidth * (1/num_of_ue_in_cell[i]))
        else:
            Avg_BW.append(carrier_bandwidth)

    return Avg_BW


# In[6]:


def mcs_rule(sinr):
    if float(sinr) < -8:
        SE = 0
    elif float(sinr) < -6.658 and float(sinr) > -8:
        SE = 0.15237
    elif float(sinr) < -4.098 and float(sinr) > -6.658:
        SE = 0.2344
    elif float(sinr) < -1.798 and float(sinr) > -4.098:
        SE = 0.377
    elif float(sinr) < 0.399 and float(sinr) > -1.798:
        SE = 0.6016
    elif float(sinr) < 2.424 and float(sinr) > 0.399:
        SE = 0.877
    elif float(sinr) < 4.489 and float(sinr) > 2.424:
        SE = 1.1758
    elif float(sinr) < 6.367 and float(sinr) > 4.489:
        SE = 1.4766
    elif float(sinr) < 8.456 and float(sinr) > 6.367:
        SE = 1.9141
    elif float(sinr) < 10.266 and float(sinr) > 8.456:
        SE = 2.4063
    elif float(sinr) < 12.218 and float(sinr) > 10.266:
        SE = 2.7305
    elif float(sinr) < 14.122 and float(sinr) > 12.218:
        SE = 3.3223
    elif float(sinr) < 15.849 and float(sinr) > 14.122:
        SE = 3.9023
    elif float(sinr) < 17.789 and float(sinr) > 15.849:
        SE = 4.5234
    elif float(sinr) < 19.809 and float(sinr) > 17.789:
        SE = 5.1152
    elif float(sinr) > 19.809:
        SE = 5.5547
    else:
        print("ERROR")

    return SE


def MCS_TH(Bandwidth, SE):
    return Bandwidth*SE
# In[7]:


def convert_dB_to_ratio(dB):
    return 10**(dB/10)


# In[8]:


def throughput_calc(sinr, bandwidth):
    if sinr > -8:
        #SINR_in_watt = convert_dB_to_ratio(sinr)
        se = mcs_rule(sinr)
        thr = MCS_TH(Bandwidth=bandwidth, SE=se)
    else:
        thr = 0
    return thr


# In[9]:


def record_HO(i, record_UE_data, obj_UE):
    m = obj_UE
    record_UE_data[i]['UE_ID'].append(str(i))
    record_UE_data[i]['s_ID'].append(str(m[i].s_cell['cell_ID']))
    #record_UE_data[i]['time'].append(str(m[i].time))
    #record_UE_data[i]['position_x'].append(str(m[i].position_x))
    #record_UE_data[i]['position_y'].append(str(m[i].position_y))
    record_UE_data[i]['s_cell_type'].append(str(m[i].s_cell['cell_type']))
    #record_UE_data[i]['s_RSRP'].append(str(m[i].s_cell['RSRP']))
    #record_UE_data[i]['s_SINR'].append(str(m[i].s_cell['SINR']))
    record_UE_data[i]['HO_state'].append(str(m[i].HO_state))
    #record_UE_data[i]['RLF'].append(str(m[i].T310))
    #record_UE_data[i]['RLF_count'].append(str(m[i].RLF_count))
    record_UE_data[i]['HO_count'].append(str(m[i].HO_count))
    record_UE_data[i]['angle'].append(str(m[i].angle))

def UE_data_to_csv(case, em_name, algo_name, num_ue):

    path = case + '_' + em_name + '_' + algo_name + '_' + num_ue
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print('Done Path')
    else:
        print(path + ' is exist')
        
        
#def Max_node_capacity(servering_st, time,num_ST, con_time_table, angle_table, serving_ue_st_id):
def Max_node_capacity(ST_serving_table,time, servering_st ,nc):   
    
    '''
    #df_3[0].argmax()
    min_st = 10000000
    target_ID = 'N'
    
    ###
    if serving_ue_st_id != -1:
        servering_st[serving_ue_st_id]-=1
    
    #print(time)
    for i in range(num_ST):
        #print(con_time_table[time][i])
        #if con_time_table[time][i] !=0 and v_servering_st[i] < min_st:# and angle_table[time][i] <1.15:
        if con_time_table[time][i] !=0 and servering_st[i] < min_st and angle_table[time][i] <1.15:
            #print()
            target_ID= i
            min_st = servering_st[i] 
            
    servering_st[target_ID]+=1
    #target_ID = servering_st.index(min_st)
    #print(target_ID)
    
    '''
    
    
    table_1 =[]
    
    for i in range(len(ST_serving_table[time])):
        
        if ST_serving_table[time][i] >0: #and servering_st[i]<nc:
                  
            table_1.append([i, servering_st[i]])
            
    #print(servering_st[i])
            
    target = -99
    target_value = 999
    
    if len(table_1)!=0:
        for i in range(len(table_1)):

            if table_1[i][1]<target_value:
                target = table_1[i][0]
                target_value = table_1[i][1]

        #target_ID = target

        if target != -99:

            target_ID = target

        else:
            target_ID = ST_serving_table[time].argmax()  

        
        
    else:
        
        table_2= []


        for i in range(len(ST_serving_table[time])):

            if ST_serving_table[time][i] >0:

                table_2.append([i,-1*servering_st[i]])

        target = -99
        target_value = -999


        for i in range(len(table_2)):

            if table_2[i][1]>target_value:
                target = table_2[i][0]
                target_value = table_2[i][1]
                
                #print(target_value)
                #print(target)

        #target_ID = target

        if target != -99:

            target_ID = target

        else:
            target_ID = ST_serving_table[time].argmax()  

            
            
    
    #print(target_ID)

    servering_st[target_ID]+=1
        
    
    return target_ID



def graph(ST_serving_table,time, servering_st ,nc):
    
    target = -99
    target_value = -999
    
    for i in range(len(ST_serving_table[time])):
        
        #print('---')
        #print('ST: '+str(i))
        #print('time '+str(ST_serving_table[time][i]))
        #print('nc '+str(servering_st[i]))
        #w = 0.7*(ST_serving_table[time][i])/700 + 0.3*(nc - servering_st[i])/nc
        
        if servering_st[i] < nc:
            #w = 0.5*(nc - servering_st[i])
            w = 0.6*(ST_serving_table[time][i])/800 + 0.4*(nc - servering_st[i])/nc
            #w = 0.7*(ST_serving_table[time][i]) + 0.3*(nc - servering_st[i])/nc
        else:
            
            #print('no')
            w = 0.6*ST_serving_table[time][i]/800 
            #w = 0.3*(nc - servering_st[i])/nc
            #w = 0.3*(nc - servering_st[i])/nc
        
        
        if w>target_value:
            
            target =i
            target_value =w
            
        else:
            pass
        
    target_ID = target
    '''       
    if target != -99:
        
        target_ID = target
        
    else:
        
        target_ID = ST_serving_table[time].argmax()  
        
    #print(target_ID)
    '''
    servering_st[target_ID]+=1
    
        
    return target_ID
        

        
def Max_serving_time(ST_serving_table,time):
    
    #df_3[0].argmax()
    target_ID = ST_serving_table[time].argmax()
    return target_ID
        
    
def Random_ST(ST_serving_table, time, ):
    
    #df_3[0].argmax()
    target_ID = random.randint(0, 297)
    return target_ID
        

def sim(algo, algo_name, timeslot, carrier_bandwidth, num_of_UE, demand, num_of_st, time, time_step, time_th, power,R,strategy, Per, E, RL_Agent=None):
    
    if(RL_Agent != None):
        RL_Agent = RL_Agent

    algo = algo
    demand = demand
    num_of_UE = num_of_UE
    num_of_st = num_of_st
    #num_of_macro = num_of_macro
    #num_of_micro = num_of_micro
    time = time
    em = 'Eichstatt'
    power = power
    case = 'Result'
    RLF = []
    HO_N = []
    totall_block = 0
    totall_max_block = 0
    re = []
    #algo = 'A3_HO'
    obj_UE = []
    record_UE_data = []
    # timeslot = 200 #ms
    size_of_timeslot = int(1000/timeslot)
    sim_time = time*size_of_timeslot
    # carrier_bandwidth = 200 # M
    obj_UE = []
    record_UE_data = []
    lst_QoS = []
    lst_time = []
    last_QOS = 0
    delay_buffer = []
    time_step = time_step
    rand = R
    pertage =Per
    node_capacity = pertage
    #node_capacity = int(num_of_UE*pertage)
    
    if node_capacity<1:
        node_capacity=1
        
    MAX_stage = strategy
    #print('nc')
    #print(node_capacity)
    #print(rand)
    print('sim time step: ' +str(time_step))
    
    #RL_Agent.time_step = time_step
    
    angle_time_table = pd.read_csv('ST_298_Telesat_6360_time/ST_angle').T
    cul_time_table =pd.read_csv('ST_298_Telesat_6360_time/ST_cul_time').T
    con_time_table =pd.read_csv('ST_298_Telesat_6360_time/ST_connect').T
    
    for i in range(num_of_UE):
        
        dict_UE = {'UE_ID': [], 'HO_state': [], 's_cell_type': [], 's_ID': [], 'HO_count': [],'angle': []}

        #dict_UE = {'UE_ID': [], 'time': [], 'position_x': [], 'position_y': [],
        #           'HO_state': [], 's_cell_type': [], 's_ID': [], 's_RSRP': [], 's_SINR': [], 'RLF': [], 'RLF_count': [],
        #           'HO_count': [], 'throughput': [], 'Data_rate': [], 'QoS': []}

        record_UE_data.append(dict_UE)
        #df = pd.read_csv(f'{em}_{R}_{time}_sec_40_slot_1/{em}_{R}_{time}_sec_0_slot_1_UE_{i}', index_col=0)
        
        #df = pd.read_csv(f'{em}_{rand}_{time}_sec_{power}_40_slot_1_MH/{em}_{rand}_{time}_sec_{power}_40_slot_1_MH_UE_{i}', index_col=0)

        Algo_1 = algo(algo_name=str(algo), ID=i, hom=3,
                      num_of_st=num_of_st, demand=demand, time_step= time_step,\
                      time_th=time_th, st=strategy)
        
        #Algo_1 = algo(algo_name=str(algo), ID=i, hom=3, data_frame=df,
        #              num_of_micro=num_of_micro, demand=demand, time_step= time_step,\
        #              time_th=time_th )

        # Algo_1 = Dual_Connect.Handover(algo_name= str(algo), ID = i, hom = 3, data_frame = df, \
        # #                               num_of_micro = num_of_micro,demand = 160)
        Algo_1 = Handover(algo_name=str(algo), ID=i, hom=3,
                      num_of_st=num_of_st, demand=demand, time_step= time_step,\
                      time_th=time_th, st=strategy)
        obj_UE.append(Algo_1)

    #print(datetime.datetime.now())
    #graph_edge =[]
    #for i in range(num_of_UE):
    #    graph_edge.append([])
    #    for j in range(num_of_micro):
    #        graph_edge[i].append(0)

        
    servering_st = []
    for i in range(num_of_st):
        servering_st.append(0)
        
    
    handoverTimes = 0
    print("sim start")
    print("strategy",MAX_stage)

    for i in range(sim_time):
        
        #print(servering_st)
        
        if RL_Agent != None:
            if len(RL_Agent.buffer.states) % 100 == 0 and len(RL_Agent.buffer.states) !=0 and E !=0:
                print('Updating...{}'.format(RL_Agent.getStatesSize()))
                RL_Agent.update()

        # if i %100 == 0 and i !=0 and E !=0:
        #     print('Updating...{}'.format(i))
        #     RL_Agent.update()
           
            
        v_time= cul_time_table[i]
        cover = con_time_table[i]
        c_st = []
        
        for c in range(298):
            if cover[c]==1:
                c_st.append(c)
        #print('Cover: '+ str(c_st))
                                            
        for j in range(num_of_UE):
            #print(str(j)+' UE_A2: '+str(obj_UE[j].time) +' time: '+str(obj_UE[j].A2_index))
            
            if obj_UE[j].ini == False:
                
                if RL_Agent == None:
                    obj_UE[j].init_connect_cell(cul_time_table, servering_st, angle_time_table, con_time_table, node_capacity)
                    #obj_UE[j].init_connect_cell(cul_time_table, servering_st, angle_time_table, con_time_table)
                    #print('INI SUC')

                    '''(self,ST_serving_table, servering_st, angle_time_table, con_time_table, nc):
                    if obj_UE[j].df['micro_0_SINR'][obj_UE[j].time] != 'N' and RL_Agent == None and\
                    obj_UE[j].ID < i:
                    #if obj_UE[j].df['micro_0_SINR'][obj_UE[j].time] != 'N':
                        obj_UE[j].init_connect_cell()
                        # print(obj_UE[j].s_cell['SINR'])
                    '''
                    
                elif RL_Agent != None: #tt
                    
                    PPO_input = []
                    for k in range(num_of_st):
                        
                        PPO_input.append(cover[k])
                    
                    #print(PPO_input)
                    
                    for k in range(num_of_st):
                        
                        PPO_input.append(servering_st[k]/node_capacity)
                        
                    #print(PPO_input[298:])
                    
                    for k in range(num_of_st):
                    
                        PPO_input.append(v_time[k])
                        
                    #print(PPO_input[596:])
                    
                    obj_UE[j].init_connect_cell(RL_Agent, PPO_input, num_of_UE, num_of_st, angle_time_table, servering_st, c_st, E)
                    servering_st, block_rate, block_table, max_block_rate = calc_num_of_ue_in_cell(obj_UE, num_of_UE, num_of_st, node_capacity)
                    Rewards_cal(obj_UE[j], num_of_UE, block_table,block_rate, node_capacity,  con_time_table, cul_time_table, i, RL_Agent)
                    
                 
                    if(RL_Agent != None):
                        done = False
  
                '''
                elif obj_UE[j].df['micro_0_SINR'][obj_UE[j].time] != 'N' and RL_Agent != None and\
                obj_UE[j].ID < i:

                    ###
                    PPO_input = []
                    for k, serve_cell in enumerate(servering_cell):
                        SINR = float(
                            obj_UE[j].df['micro_'+str(k)+'_SINR'][obj_UE[j].time])
                        # RSRP=process_RSRP_val(RSRP)

                        N_SE = mcs_rule(SINR)/5.555
                        #N_serve_cell= serve_cell/num_of_UE
                        #N_SE = mcs_rule(SINR)



                        PPO_input.append(float(N_SE))


                    #for k, serve_cell in enumerate(servering_cell):
                        #SINR = float(
                            #obj_UE[j].df['micro_'+str(k)+'_SINR'][obj_UE[j].time])
                        # RSRP=process_RSRP_val(RSRP)

                        #N_SE = mcs_rule(SINR)/5.555
                        #N_serve_cell= serve_cell/num_of_UE

                        #N_serve_cell= serve_cell
                        ###PPO_input.append(float(N_serve_cell))
                        
                        #print(N_serve_cell)
                        #PPO_input.append(float(SINR))
                        #PPO_input.append(float(serve_cell))
                        #  obj_UE[j].A3_HO()
                    #print('state: '+str(PPO_input))
                    
                    demand_type =[0,0,0]
                    demand_type[(obj_UE[j].ID%3)] = 1
                    for d in range(3):
                        PPO_input.append(demand_type[d])

                    #PPO_input.append(float(i/sim_time))
                    obj_UE[j].init_connect_cell(RL_Agent, PPO_input, delay_buffer ,graph_edge, node_features)
                    if(RL_Agent != None):
                        done = False

                else:
                    
                    pass
                
                
                '''
            


            elif obj_UE[j].ini == True:  #and obj_UE[j].df['micro_0_SINR'][obj_UE[j].time] != 'N':
                
                
                
                #if(i == sim_time-1):
                #    done = True
                    
                    
                if(RL_Agent == None):
                    
                    #print(obj_UE[j].s_cell['cell_ID'][0])
                    #print(angle_time_table[i][obj_UE[j].s_cell['cell_ID'][0]])

                    #print('------------------')
                    #print( obj_UE[j].min_angle)
                    
                    if angle_time_table[i][obj_UE[j].s_cell['cell_ID'][0]] > obj_UE[j].min_angle or angle_time_table[i][obj_UE[j].s_cell['cell_ID'][0]]==-1: 
                    
                        obj_UE[j].handover(cul_time_table, servering_st,con_time_table, angle_time_table, node_capacity)
                        #print('time')
                        #print(i)
                        
                        #print('handover')
                        
                    else:
                        pass
                else:
                    #if angle_time_table[i] > obj_UE[j].min_angle 
                    
                    PPO_input = []
                    for k in range(num_of_st):
                        
                        PPO_input.append(cover[k])
                        
                    for k in range(num_of_st):
                        
                        #PPO_input.append(servering_st[k])
                        PPO_input.append(servering_st[k]/node_capacity)
                        
                    #print(PPO_input[298:])
                        
                    for k in range(num_of_st):
                    
                        PPO_input.append(v_time[k])

                    #print('state: '+str(PPO_input))
                    ## handover event trigger
                    if angle_time_table[i][obj_UE[j].s_cell['cell_ID'][0]] > obj_UE[j].min_angle or  angle_time_table[i][obj_UE[j].s_cell['cell_ID'][0]]==-1: 
                        
                        handoverTimes += 1
                        # print(handoverTimes ," handover triggr at time ",i)
                        obj_UE[j].A2_event = obj_UE[j].handover(RL_Agent, PPO_input,num_of_UE, num_of_st, servering_st, c_st, E)
                        servering_st, block_rate, block_table, max_block_rate = calc_num_of_ue_in_cell(obj_UE, num_of_UE, num_of_st, node_capacity)
                        Rewards_cal(obj_UE[j], num_of_UE, block_table,block_rate, node_capacity,  con_time_table, cul_time_table, i, RL_Agent)
                    else:
                       pass
                    # obj_UE[j].A2_event = obj_UE[j].handover(RL_Agent, PPO_input,num_of_UE, num_of_st, servering_st, c_st, E)

                    #A2_Event = obj_UE[j].handover(RL_Agent, PPO_input)
                    
            record_HO(j, record_UE_data, obj_UE)
            
            #print('j')
            #print(j)
            obj_UE[j].Update_UE_information(angle_time_table)
            obj_UE[j].time += 1
            #print('UE_time')
            #print(obj_UE[j].time)
            #print(obj_UE[j].angle)
        #servering_st : num of ue in cell
        #block table : if the st Overload?
        servering_st, block_rate, block_table, max_block_rate = calc_num_of_ue_in_cell(obj_UE, num_of_UE, num_of_st, node_capacity)
        
        #print(block_rate)
        totall_block+= block_rate
        totall_max_block += max_block_rate
        
        # if RL_Agent != None:
        #     for j in range(num_of_UE):
        #         Rewards_cal(obj_UE[j], num_of_UE, block_table,servering_st, node_capacity,  con_time_table, cul_time_table, i, RL_Agent)
        # else:
        #     pass
        
        
        #A_BW = Allow_Bandwidth(carrier_bandwidth, servering_cell, num_of_micro)
        '''
        if i % 1 == 0 and i != 0:
            # lst_time.append(int(i/5))
            avg_sec_QoS = 0
            num_run_ue = 0
        
        
        for k in range(num_of_UE):
            if obj_UE[k].df['micro_0_SINR'][obj_UE[k].time] != 'N':

                #obj_UE[k].HO_state != 'H_T'

                for l in range(obj_UE[k].num_connect):
                    if obj_UE[k].s_cell['SINR'][l] != [] and obj_UE[k].HO_state != 'H_T':
                        SINR = obj_UE[k].s_cell['SINR'][l]
                        # print(obj_UE[k].s_cell['cell_ID'][l])
                        bandwidth = A_BW[int(obj_UE[k].s_cell['cell_ID'][l])]
                        obj_UE[k].slot_thr[l] =\
                            throughput_calc(SINR, bandwidth) / size_of_timeslot
                        #total_throughput += throughput_calc(SINR, bandwidth)
                #obj_UE[k].slot_thr = total_throughput / size_of_timeslot

                    elif obj_UE[k].s_cell['SINR'][l] == []:
                        obj_UE[k].slot_thr[l] = []

                    elif obj_UE[k].HO_state == 'H_T' and obj_UE[k].t_cell['cell_ID'][l] == 'X':
                        SINR = obj_UE[k].s_cell['SINR'][l]
                        # print(obj_UE[k].s_cell['cell_ID'][l])
                        bandwidth = A_BW[int(obj_UE[k].s_cell['cell_ID'][l])]
                        obj_UE[k].slot_thr[l] = throughput_calc(
                            SINR, bandwidth) / size_of_timeslot

                    elif obj_UE[k].HO_state == 'H_T' and obj_UE[k].t_cell['cell_ID'][l] != 'X':

                        obj_UE[k].slot_thr[l] = 0
                        
                obj_UE[k].Data_rate_sec_cal()
                obj_UE[k].QoS_sec_cal()

            elif obj_UE[k].df['micro_0_SINR'][obj_UE[k].time] == 'N':
                
                obj_UE[k].QoS ='N'
                
                for nm in range(num_of_micro):
                    obj_UE[k].Data_rate[nm]=0
                
                #pass

            #print('demand:' +str(obj_UE[k].demand))
            #print('QoS: '+ str(obj_UE[k].QoS))
            
            if i % 1 == 0 and i != 0:

                if obj_UE[k].QoS != 'N':
                    
                    avg_sec_QoS += obj_UE[k].QoS
                    num_run_ue += 1

                    ###print('QoS start')

                    if obj_UE[k].A2_start !=0 and len(lst_QoS)>0:
                    
                        obj_UE[k].A2_cumul(lst_QoS[-1])
                        
                        ###print('Time: ' + str(i)+' UE_'+str(k) + 'reward: '+\
                        ###      str(obj_UE[k].A2_reward))
                        
                    elif obj_UE[k].A2_start !=0 and len(lst_QoS) == 0:
                        
                        obj_UE[k].A2_cumul(obj_UE[k].QoS)
                    

            record_UE_data[k]['throughput'].append(str(obj_UE[k].slot_thr))
            record_UE_data[k]['Data_rate'].append(str(obj_UE[k].Data_rate))
            record_UE_data[k]['QoS'].append(str(obj_UE[k].QoS))
            

            if(obj_UE[k].A2_event):
                #print('yes A2 start')
                if(RL_Agent != None):
                    done = False
                      
                    
                    if(i == sim_time-1):
                        done = True

                    #if obj_UE[k].A2_start ==1:
                    
                    #    obj_UE[k].A2_start =2
                    
                    #elif obj_UE[k].A2_start ==2:
                    th = 0.81
                    if obj_UE[k].A2_start ==2 and (i != sim_time-1):
                    
                        reward_1, reward_2, buf_ind_o, buf_ind_n = obj_UE[k].cal_A2_cumul_reward()
                        
                        #print('buf_ID '+ str(buf_index))
                        #print('reward '+ str(reward))
                        #print('server '+ str(obj_UE[k].s_cell['cell_ID']))

                        ###print('UE ' +str(k)+' handover done')

                        
                        #if obj_UE[k].s_cell['cell_ID'][0] != [] and obj_UE[k].s_cell['cell_ID'][1] !=[]:



                                #delay_buffer[buf_ind_o] = ((critic_eva_rewa).tolist())[0]
                        delay_buffer[buf_ind_o] = reward_1
                            #delay_buffer[buf_index] = pre_reward
                        delay_buffer[buf_ind_o+1] = reward_2 
                                #delay_buffer[buf_ind_o+1] = reward - delay_buffer[buf_ind_o]
                                
                                #print('buffer_id_r_'+str(buf_ind_o)+': '+str( delay_buffer[buf_ind_o]))
                                #print('delay_id_r_'+str(buf_ind_o+1)+':'+str( delay_buffer[buf_ind_o+1]))
                            
                            #else:
                            #    print('error reward at '+str(i)+' time '+str(k) +' UE')
                            #print('action_1_and_2_reward_done')
                            #print('buf_index_1:' +str(buf_index) + ' reward ' +str(pre_reward))
                            #print('buf_index_2:' +str(buf_index+1) + ' reward ' +str(reward))
                        #else:
                            
                        #    print('index error')
   
                            #delay_buffer[buf_index] = reward
                            #print('buf_index_1:' +str(buf_index) +  ' reward ' +str(reward))
                 
                            ###print(delay_buffer) 
                        obj_UE[k].A2_ini()
                        
                    ###print('Time: ' + str(i)+' UE_'+str(k) + ' start A2_reward')
                    
                    
                    if obj_UE[k].t_cell['cell_ID'][0] != [] and obj_UE[k].t_cell['cell_ID'][1] !=[]:
                        
                        print(obj_UE[k].t_cell['cell_ID'][0],obj_UE[k].t_cell['cell_ID'][1])
                        delay_buffer.append(0)
                        RL_Agent.buffer.is_terminals.append(done)
                        obj_UE[k].A2_index = len(delay_buffer)-1
                        
                        
                        delay_buffer.append(0)
                        RL_Agent.buffer.is_terminals.append(done)
                        print('action_1_2 terminal_done')
                        print('########################')
                        #obj_UE[k].A2_index = len(delay_buffer)-1
                       
                    else:
                        print(obj_UE[k].t_cell['cell_ID'][0],obj_UE[k].t_cell['cell_ID'][1])
                        delay_buffer.append(0)
                        RL_Agent.buffer.is_terminals.append(done)
                        obj_UE[k].A2_index = len(delay_buffer)-1
                        print('action_1 terminal_done')
                        print('########################')
                    
                    
                    ###print(delay_buffer) 
                    #print(len(delay_buffer))                  
                    ###print(obj_UE[k].A2_index )
                    
                    
                    #RL_Agent.buffer.rewards.append(float(reward))
                    #RL_Agent.buffer.is_terminals.append(done)
                    
            if obj_UE[k].A2_start ==1 and (i == sim_time-1):
                #print('UE_'+str(k)+'_final reward in if 1')
                if obj_UE[k].A2_time == 0:
                    #print('UE_'+str(k)+'_do not final reward in if 2')
                    ### time up but UE do action so clean buffer
                    ### this part is cleaned at ppo_tf.py
                    pass
                    
                else:
                    print('UE')
                    print(k)  
                    if RL_Agent != None:
                        reward_1, reward_2, buf_ind_o, buf_ind_n = obj_UE[k].cal_A2_cumul_reward()

                        if obj_UE[k].A2_index != 'N' : #and time_step <=10:
                            buf_ind_o = int(obj_UE[k].A2_index)
                        else:
                            pass

                        delay_buffer[buf_ind_o] = reward_1
                            #delay_buffer[buf_index] = pre_reward
                        delay_buffer[buf_ind_o+1] = reward_2 

                        #print('UE_'+str(k)+'_do final reward in if 2')

                        #print('buf_ID '+ str(buf_index))
                        #print('reward '+ str(reward))
                        ###print('UE ' +str(k)+' handover timeout done')

                        #else:


                        #print('FFFFF index error')
                        #delay_buffer[int(buf_index)] = reward
                        #print('buf_index_1:' +str(buf_index) +  ' reward ' +str(reward))


                        obj_UE[k].A2_ini()
            
            
            obj_UE[k].time += 1
            
                 
        if i <10 and i >4:
            print(servering_cell)
            print('run ue '+str(num_run_ue))
            
        
            

        if i % 1 == 0 and i > 0:
            
            if avg_sec_QoS > 0:
                lst_time.append(int(i/1))
                lst_QoS.append(avg_sec_QoS/num_run_ue)
                #last_QOS = avg_sec_QoS/num_run_ue
              
        ###
        
        for k in range(num_of_UE):
            if(A2_Event):
                if(RL_Agent != None):
                    done = False
                    if(i == sim_time-1):
                        done = True
                    if obj_UE[k].QoS != 'N':
                        reward = float(last_QOS)
                    else:
                        reward = 0
                    #if(reward < 0.7):
                    #    reward = reward-1
                    RL_Agent.buffer.rewards.append(float(reward))
                    RL_Agent.buffer.is_terminals.append(done)
        
        ###
        '''

        ###if i % int(sim_time/10) == 0 and i != 0:
        ###    print(f'sim_time_percentage: {int(i/sim_time*100)}%')



    
    # if RL_Agent != None:
    #     for j in range(num_of_UE):
    #         Rewards_cal(obj_UE[j], num_of_UE, block_table,servering_st, node_capacity,  con_time_table, cul_time_table, i, RL_Agent)
    print("total HO times" , handoverTimes)
    print(f'sim_time_percentage: {int(sim_time/sim_time*100)}%')
    #print('ser_UE: ' + str(servering_st))
    
    
    #for i in range(len(lst_time)):
    #    last_QOS+= lst_QoS[i]
        
    #print('last_QOS:', last_QOS/len(lst_QoS))
    #UE_data_to_csv(case, em, algo_name, str(num_of_UE))
    Total_HO_count =0
    
    for i in range(num_of_UE):
        #Total_RLF_count += multi_process_list[i].RLF_count
        Total_HO_count += obj_UE[i].HO_count
        
        #Total_resource += multi_process_list[i].reservation_time
        # Algo_1.init_all_value()

        df = pd.DataFrame(data=record_UE_data[i])

        record_name = case+'_'+em + '_'+algo_name+'_'+str(num_of_UE)
        #df.to_csv(f'{record_name}/UE_{i}.csv',index=False)

    # RLF.append(Total_RLF_count/num_of_UE/15)
    
    BBB =totall_block/sim_time
    HHH =Total_HO_count/num_of_UE
    MMM = totall_max_block/sim_time
    print("Total_HO_count/num_of_UE", Total_HO_count/num_of_UE)
    print("totall_block/sim_time", totall_block/sim_time)
    with open('./data/result.csv','a',newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|',quoting=csv.QUOTE_MINIMAL)
        writer.writerow([BBB]+[HHH]+[MMM])
        # re.append(Total_resource/num_of_UE/15)

    print(datetime.datetime.now())
    
    if (RL_Agent != None):
        for i in range(len(delay_buffer)):

            #RL_Agent.buffer.rewards.append(delay_buffer[i])
            
            
            if delay_buffer[i] == 'N':
                print('ID: '+str(i))
                
                remove_ind = (RL_Agent.buffer.rewards.index('N'))
                
                RL_Agent.buffer.actions.pop(remove_ind)
                RL_Agent.buffer.states.pop(remove_ind)
                RL_Agent.buffer.logprobs.pop(remove_ind)
                RL_Agent.buffer.rewards.pop(remove_ind)
                RL_Agent.buffer.is_terminals.pop(remove_ind)
                RL_Agent.buffer.UE_ID.pop(remove_ind)

        '''    
        for i in range(len(RL_Agent.buffer.actions)):
            print(RL_Agent.buffer.states[i])
            print(RL_Agent.buffer.actions[i])
            print(RL_Agent.buffer.rewards[i])
        '''
        print('RL buffer size')
        #print(len(RL_Agent.buffer.actions))
        print(len(RL_Agent.buffer.states))
        #print(len(RL_Agent.buffer.logprobs))
        print(len(RL_Agent.buffer.rewards))
        print(len(RL_Agent.buffer.is_terminals))
        #print(len(RL_Agent.buffer.UE_ID))
        print('##########################')
        if len(RL_Agent.buffer.is_terminals) > 0:
            RL_Agent.buffer.is_terminals.pop()
        RL_Agent.buffer.is_terminals.append(True)
        #print('reward')
        #print(RL_Agent.buffer.rewards)

        #print('delay_buffer: ' +str(delay_buffer))

        #print(delay_buffer)
        
        
        
        #if RL_Agent.test == True:
        #    print(RL_Agent.buffer.states[::])
        #    print(RL_Agent.buffer.logprobs[::])
        #    print(RL_Agent.buffer.rewards[::])
        
    return BBB, HHH





