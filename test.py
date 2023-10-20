import torch
# ppo_device = torch.device('cpu')

# if(torch.cuda.is_available()): 
#     ppo_device = torch.device('cuda:0') 
#     torch.cuda.empty_cache()
#     print("Device set to : " + str(torch.cuda.get_device_name(ppo_device)))
# else:
#     print("Device set to : cpu")
t = [[1,2,3],[4,5,6]]
t1 = [[7,8,9],[10,11,12]]
g = torch.FloatTensor(t)
g1 = torch.FloatTensor(t1)
h = torch.cat((g,g1),2)
#.to(ppo_device)
print(h)