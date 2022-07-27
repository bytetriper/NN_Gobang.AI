from copy import deepcopy
from ctypes.wintypes import tagRECT
from functools import reduce
from multiprocessing import reduction
import torch.optim as optim
import torch.nn as nn
import torch
from time import time
from pytorch_classification.utils import Bar, AverageMeter
from utils import *
import os
import numpy as np
import math
import sys
sys.path.append('../../')

from network.NNetArchitecture import NNetArchitecture as nnetarch

"""
    TODO: Tune or add new arguments if you need
"""
args = dotdict({
    'lr': .003,
    'cuda': torch.cuda.is_available(),
    'feat_cnt': 3
})

class NNetWrapper():
    def __init__(self, game):
        self.nnet = nnetarch(game, args)
        self.feat_cnt = args.feat_cnt
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()

        """
            TODO: Choose a optimizer and scheduler

            self.optimzer = ...
            self.scheduler = ...
        """
        self.optimizer=optim.Adam(self.nnet.parameters(),lr=args['lr'])
        #self.scheduler=optim.lr_scheduler.MultiStepLR(self.optimizer,milestones=[30,40],gamma=0.1)
        self.scheduler=optim.lr_scheduler.ExponentialLR(self.optimizer,0.7)
        if args.cuda:# No cuda man
            self.nnet.cuda()

    def loss_pi(self, outputs, targets):
        """
            TODO: Design a policy loss function
        """
        #print(outputs)
        #print(targets)
        #lossf=nn.KLDivLoss()
        lossf=nn.KLDivLoss(reduction='batchmean')
        loss_pi=lossf(outputs,targets)
        return loss_pi

    def loss_v(self, outputs, targets):
        """
            TODO: Design a evaluation loss function
        """
        loss_v=torch.tensor([0])
        loss_v=loss_v.cuda()
        return loss_v

    def train(self, batches, train_steps):

        # Switch to train mode
        self.nnet.train()

        data_time = AverageMeter()
        batch_time = AverageMeter()
        pi_losses = AverageMeter()
        v_losses = AverageMeter()
        end = time()

        print(f"Current LR: {self.optimizer.state_dict()['param_groups'][0]['lr']}")
        bar = Bar(f'Training Net', max=train_steps)
        current_step = 0
        while current_step < train_steps:
            for batch_idx, batch in enumerate(batches):
                if current_step == train_steps:
                    break
                current_step += 1

                # Obtain targets from the dataset
                boards, target_pis, target_vs = batch
                if args.cuda:
                    boards, target_pis, target_vs = boards.contiguous().cuda(
                    ), target_pis.contiguous().cuda(), target_vs.contiguous().cuda()
                data_time.update(time() - end)
                out_pi,out_v=self.nnet(boards)
                l_pi=self.loss_pi(out_pi,target_pis)
                l_v=self.loss_v(out_v,target_vs)
                """
                    TODO: Compute output & loss
                    out_pi, out_v = ...
                    l_pi = ... 
                    l_v = ...
                """
                total_loss = l_pi + l_v

                pi_losses.update(l_pi.item(), boards.size(0))
                v_losses.update(l_v.item(), boards.size(0))

                """
                    TODO: Compute gradient (backward) and do optimizer step
                """
                self.optimizer.zero_grad()
                l_pi.backward()
                self.optimizer.step()
                batch_time.update(time() - end)
                end = time()
                bar.suffix = '({step}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss_pi: {lpi:.4f} | Loss_v: {lv:.3f}'.format(
                    step=current_step,
                    size=train_steps,
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    lpi=pi_losses.avg,
                    lv=v_losses.avg,
                )
                bar.next()

        """
            TODO: do scheduler step
        """
        self.scheduler.step()
        bar.finish()
        print()

        return pi_losses.avg, v_losses.avg

    def predict(self, board):
        # Preparing input
        board = torch.FloatTensor(board.astype(np.float64))
        if args.cuda:
            board = board.contiguous().cuda()

        with torch.no_grad():
            board = board.view(self.feat_cnt, self.board_x, self.board_y)

            # Switch to eval mode
            self.nnet.eval()

            """
                TODO: predict pi & v
            """
            pi,v=self.nnet(board)
            return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            os.mkdir(folder)
        state=deepcopy(self.nnet.state_dict())
        torch.save(state,filepath)
        """
            TODO: save the model (nnet, optimizer and scheduler) in the given filepath
        """

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise ("No model in path {}".format(filepath))
        paras=torch.load(filepath)
        self.nnet.load_state_dict(paras)
        self.nnet.eval()
        """
            TODO: load the model (nnet, optimizer and scheduler) from the given filepath
        """

            