import torch
ppo_device = torch.device('cpu')

if(torch.cuda.is_available()): 
    ppo_device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(ppo_device)))
else:
    print("Device set to : cpu")
t = [1,2,3]
g = torch.FloatTensor(t)
#.to(ppo_device)
print(g)