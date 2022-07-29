from time import sleep
import torch.nn.functional as F
import torch.nn as nn
import torch
import sys
import torch.nn.init as init
sys.path.append('..')

class NNetArchitecture(nn.Module):
    def __init__(self, game, args):
        super(NNetArchitecture, self).__init__()
        # game params
        self.feat_cnt = args.feat_cnt
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args
        
        """
            TODO: Add anything you need
        """

        self.cv1=nn.Sequential(
            nn.ZeroPad2d(padding=(2,2,2,2)),
            nn.Conv2d(self.feat_cnt,16,5),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.cv2=nn.Sequential(
            nn.ZeroPad2d(padding=(2,2,2,2)),
            nn.Conv2d(16,32,5),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.cv3=nn.Sequential(
            nn.ZeroPad2d(padding=(2,2,2,2)),
            nn.Conv2d(32,64,5),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.cv4=nn.Sequential(
            nn.ZeroPad2d(padding=(2,2,2,2)),
            nn.Conv2d(64,128,5),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.cv5=nn.Sequential(
            nn.ZeroPad2d(padding=(2,2,2,2)),
            nn.Conv2d(128,256,5),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.cv6=nn.Sequential(
            nn.ZeroPad2d(padding=(2,2,2,2)),
            nn.Conv2d(256,512,5),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.fc1=nn.Linear(512*9*9,self.action_size)
        #self.bn2=nn.BatchNorm1d(self.action_size)
        for m in self.modules():
            if isinstance(m,nn.Linear):
                init.xavier_uniform_(m.weight.data)
            if isinstance(m,nn.BatchNorm1d):
                init.xavier_uniform_(m.weight.data)

    def forward(self, s):
        # batch_size x feat_cnt x board_x x board_y
        s = s.view(-1, self.feat_cnt, self.board_x, self.board_y)  
        pi=s 
        for i in range(0,s.shape[0]):
            pi[i]=(s[i]-s[i].mean())/s[i].var()
        #print(pi.shape)
        pi=self.cv1(pi)
       # print(pi.shape)
        pi=self.cv2(pi)
        #print(pi.shape)
        #print(pi)
        pi=self.cv3(pi)
        pi=self.cv4(pi)
        pi=self.cv5(pi)
        pi=self.cv6(pi)
        pi=self.fc1(pi.view(pi.shape[0],-1))
        #print(pi)
        """
            TODO: Design your neural network architecture
            Return a probability distribution of the next play (an array of length self.action_size) 
            and the evaluation of the current state.

            pi = ...
            v = ...
        """
        v=torch.tensor([1])
        v=v.cuda()
        # Think: What are the advantages of using log_softmax ?
        return F.log_softmax(pi, dim=1), torch.tanh(v)
        #return pi, torch.tanh(v)