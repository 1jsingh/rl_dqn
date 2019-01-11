import torch
import torch.nn as nn
import torch.nn.functional as F

class Qnetwork(nn.Module):
    def __init__(self,state_shape,action_size):
        # initialising the super class properties
        super(Qnetwork,self).__init__()
        
        H,W,C = state_shape

        # defining layers
        # Dueling networks, refer https://arxiv.org/abs/1511.06581
        
        # common network layers

        # input.shape = (N,4,84,84)
        self.conv1 = nn.Conv2d(C,32,kernel_size=8,stride=4) # (N,32,41,41)  
        self.conv2 = nn.Conv2d(32,64,kernel_size=4,stride=2) #(N,64,20,20)
        self.conv3 = nn.Conv2d(64,64,kernel_size=3,stride=1) # (N,128,9,9)
        
        # # Value network layers
        # self.fc4_v = nn.Linear(64*7*7,128)
        # self.out_v = nn.Linear(128,1)

        # Advantage estimate layers
        self.fc4_a = nn.Linear(64*7*7,512)
        self.out_a = nn.Linear(512,action_size) 
        
    def forward(self,states):

        # common network
        x = F.relu(self.conv1(states))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # flatten the tensor before feeding it to the fc layers
        x = x.view(x.shape[0],-1)

        # # value network
        # v = F.relu(self.fc4_v(x))
        # v = self.out_v(v)

        # advantage network
        a = F.relu(self.fc4_a(x))
        a = self.out_a(a)

        # # refine advantage
        # a_ = a - a.mean(dim=1,keepdim=True)

        # # combine v and a_
        # q = v + a_ 
        return a

# class Qnetwork(nn.Module):
#     def __init__(self,state_shape,action_size):
#         # initialising the super class properties
#         super(Qnetwork,self).__init__()
        
#         H,W,C = state_shape

#         # defining layers
#         # Dueling networks, refer https://arxiv.org/abs/1511.06581
        
#         # common network layers

#         # input.shape = (N,4,84,84)
#         self.conv1 = nn.Conv2d(C,32,kernel_size=8,stride=4) # (N,32,41,41)  
#         self.conv2 = nn.Conv2d(32,64,kernel_size=4,stride=2) #(N,64,20,20)
#         self.conv3 = nn.Conv2d(64,64,kernel_size=3,stride=1) # (N,128,9,9)
        
#         # Value network layers
#         self.fc4_v = nn.Linear(64*7*7,128)
#         self.out_v = nn.Linear(128,1)

#         # Advantage estimate layers
#         self.fc4_a = nn.Linear(64*7*7,128)
#         self.out_a = nn.Linear(128,action_size) 
        
#     def forward(self,states):

#         # common network
#         x = F.relu(self.conv1(states))
#         x = F.relu(self.conv2(x))
#         x = F.relu(self.conv3(x))
        
#         # flatten the tensor before feeding it to the fc layers
#         x = x.view(x.shape[0],-1)

#         # value network
#         v = F.relu(self.fc4_v(x))
#         v = self.out_v(v)

#         # advantage network
#         a = F.relu(self.fc4_a(x))
#         a = self.out_a(a)

#         # refine advantage
#         a_ = a - a.mean(dim=1,keepdim=True)

#         # combine v and a_
#         q = v + a_ 
#         return q