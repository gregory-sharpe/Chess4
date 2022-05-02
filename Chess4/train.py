"""
An implementation of the training pipeline of AlphaZero adapted from gomoku from Junxiao Song
"""

#from __future__ import print_function
import torch.multiprocessing
try:
     torch.multiprocessing.set_start_method('spawn')
except RuntimeError:
    pass
import random
import numpy as np
from collections import defaultdict, deque
from ChessEngine import GameState
from ChessMain import Game
from Mcts import MCTSPlayer as MCTS_Pure
from MctsAlphaZero import MCTSPlayer
from PolicyValueNet import PolicyValueNet  # Theano and Lasagne
import os
import glob
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Process, Manager
import time
# from policy_value_net_pytorch import PolicyValueNet  # Pytorch
# from policy_value_net_tensorflow import PolicyValueNet # Tensorflow
# from policy_value_net_keras import PolicyValueNet # Keras



class TrainPipeline():
    def getBestFile(self):
        list_of_files = glob.glob('TrainedModels/BestModels/*')  # * means all if need specific format then *.csv
        # requires atleast 1 file to be in best Models
        if len(list_of_files) > 0:
            bestFile = max(list_of_files, key=os.path.getctime)

            return bestFile
        else:
            return None
    def setBestFile(self):
        list_of_files = glob.glob('TrainedModels/BestModels/*')  # * means all if need specific format then *.csv
        # requires atleast 1 file to be in best Models
        if len(list_of_files)>0:
            bestFile = max(list_of_files, key=os.path.getctime)
            print("here")
            self.bestPolicy_value_net = PolicyValueNet(self.board_width,
                                               self.board_height,
                                               model_file=bestFile)
        else:
            self.bestPolicy_value_net = None
    def __init__(self, init_model=None):
        #TODO
        # Currently it will start with a fresh model and the compare it to the best from previous iterations
        # once 100k simulations has been reached have it start from the best model and test whether it has improved less frequently
        # params of the board and the game
        self.board_width = 14
        self.board_height = 14
        self.board = GameState()
        self.game = Game(self.board)
        # training params
        self.learn_rate = 2e-3
        self.lr_multiplier = 1.0  # adaptively adjust the learning rate based on KL
        self.temp = 1.0  # the temperature param
        self.n_playout = 2  # num of simulations for each move
        self.c_puct = 3
        self.buffer_size = 10000 # change back to 10000
        self.batch_size = 512  # mini-batch size for training
        # typically
        self.data_buffer = deque(maxlen=self.buffer_size)

        self.play_batch_size = 1
        self.epochs = 5  # num of train_steps for each update
        self.kl_targ = 0.02
        self.check_freq = 10
        self.game_batch_num = 100
        self.best_win_ratio = 0.0
        self.CPUCount = 4#torch.multiprocessing.cpu_count()
        # num of simulations used for the pure mcts, which is used as
        # the opponent to evaluate the trained policy
        self.pure_mcts_playout_num = 100
        init_model = self.getBestFile()
        if init_model:
            print("here model used")
            # start training from an initial policy-value net
            self.policy_value_net = PolicyValueNet(self.board_width,
                                                   self.board_height,
                                                   model_file=init_model,use_gpu=torch.cuda.is_available()
                                                   )

        else:
            #print("here2")
            # start training from a new policy-value net
            self.policy_value_net = PolicyValueNet(self.board_width,
                                                   self.board_height,use_gpu=torch.cuda.is_available())

        self.mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                      c_puct=self.c_puct,
                                      n_playout=self.n_playout,
                                      is_selfplay=1)
        list_of_files = glob.glob('TrainedModels/BestModels/*')  # * means all if need specific format then *.csv
        # requires atleast 1 file to be in best Models

    def get_equi_data(self, play_data):
        """augment the data set by rotation
        and flipping
        play_data: [(state, mcts_prob, winner_z), ..., ...]
        """
        extend_data = []
        for state, mcts_porb, winner in play_data:
            for i in [1, 2, 3, 4]:
                # rotate counterclockwise
                equi_state = np.array([np.rot90(s, i) for s in state])
                equi_mcts_prob = mcts_porb#np.rot90(np.flipud(
                    #mcts_porb.reshape(self.board_height, self.board_width)), i)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
                ## flip horizontally
                #equi_state = np.array([np.fliplr(s) for s in equi_state])
                #equi_mcts_prob = equi_mcts_prob #np.fliplr(equi_mcts_prob)
                #extend_data.append((equi_state,
                #                   np.flipud(equi_mcts_prob).flatten(),
                #                    winner))
        return extend_data

    def collect_selfplay_data(self, n_games=1):
        """collect self-play data for training"""
        for i in range(n_games):
            winner, play_data = self.game.start_self_play(self.mcts_player,
                                                          temp=self.temp)
            play_data = list(play_data)[:]
            self.episode_len = len(play_data)
            # augment the data
            play_data = self.get_equi_data(play_data)
            self.data_buffer.extend(play_data)# needs a lock

    def policy_update(self):
        """update the policy-value net"""

        mini_batch = random.sample(self.data_buffer, self.batch_size)

        state_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        winner_batch = [data[2] for data in mini_batch]
        old_probs, old_v = self.policy_value_net.policy_value(state_batch)
        for i in range(self.epochs):
            loss, entropy = self.policy_value_net.train_step(
                    state_batch,
                    mcts_probs_batch,
                    winner_batch,
                    self.learn_rate*self.lr_multiplier)
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)
            kl = np.mean(np.sum(old_probs * (
                    np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                    axis=1)
            )
            if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly
                break
        # adaptively adjust the learning rate
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5
        explained_var_old = (1 -
                             np.var(np.array(winner_batch) - old_v.flatten()) /
                             np.var(np.array(winner_batch)))
        explained_var_new = (1 -
                             np.var(np.array(winner_batch) - new_v.flatten()) /
                             np.var(np.array(winner_batch)))
        print(("kl:{:.5f},"
               "lr_multiplier:{:.3f},"
               "loss:{},"
               "entropy:{},"
               "explained_var_old:{:.3f},"
               "explained_var_new:{:.3f}"
               ).format(kl,
                        self.lr_multiplier,
                        loss,
                        entropy,
                        explained_var_old,
                        explained_var_new))
        return loss, entropy
    def policy_evaluateinParralel(self,n_games =25):
        current_mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                         c_puct=self.c_puct,
                                         n_playout=self.n_playout)
        if self.bestPolicy_value_net == None:
            best_mcts_player = MCTS_Pure(c_puct=self.c_puct,
                                         n_playout=self.pure_mcts_playout_num)
        else:
            best_mcts_player = MCTSPlayer(self.bestPolicy_value_net.policy_value_fn,
                                         c_puct=self.c_puct,
                                         n_playout=self.n_playout) #self.n_playout)

        with torch.multiprocessing.Manager() as manager:
            executor = ProcessPoolExecutor(1)
            mwin_cnt = manager.dict()
            mwin_cnt[-1] = 0
            mwin_cnt[1] = 0
            mwin_cnt[2]=0
            futures = [executor.submit(self.game.start_play,current_mcts_player,best_mcts_player,mwin_cnt) for i in range(n_games)]
            executor.shutdown(wait=True)
            print(mwin_cnt)
            win_ratio = 1.0*(mwin_cnt[1] + 0.5*mwin_cnt[-1]) / n_games
            print("num_playouts:{}, win: {}, lose: {}, tie:{}".format(
                   self.pure_mcts_playout_num,
                    mwin_cnt[1], mwin_cnt[2], mwin_cnt[-1]))
            return win_ratio

    def policy_evaluate(self, n_games=10):
        current_mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                         c_puct=self.c_puct,
                                         n_playout=self.n_playout)
        if self.bestPolicy_value_net == None:
            best_mcts_player = MCTS_Pure(c_puct=5,
                                         n_playout=self.pure_mcts_playout_num)
        else:
            best_mcts_player = MCTSPlayer(self.bestPolicy_value_net.policy_value_fn,
                                         c_puct=self.c_puct,
                                         n_playout=self.n_playout) #self.n_playout)
        win_cnt = defaultdict(int)

        for i in range(n_games):
            winner = self.game.start_play(current_mcts_player,
                                          best_mcts_player)
            win_cnt[winner] += 1
        win_ratio = 1.0*(win_cnt[1] + 0.5*win_cnt[-1]) / n_games
        print("num_playouts:{}, win: {}, lose: {}, tie:{}".format(
                self.pure_mcts_playout_num,
                win_cnt[1], win_cnt[2], win_cnt[-1]))
        return win_ratio
    # @deprecated
    def runSelfPlayInParralel(self,i,saveNoneBestModels = False):
        self.collect_selfplay_data(self.play_batch_size)
        print(i)
        print(len(self.data_buffer))
        print()
        print("batch i:{}, episode_len:{}".format(
            i + 1, self.episode_len))
        if len(self.data_buffer) > self.batch_size:
            loss, entropy = self.policy_update()
        # check the performance of the current model,
        # and save the model params
        filePath = ""
        fileNumber = 0
        if saveNoneBestModels and not os.path.exists(
                "TrainedModels/EveryModel/policyModel%s.model" % fileNumber):  # this is for the first time
            filePath = "TrainedModels/EveryModel/policyModel" + str(fileNumber) + ".model"
        while saveNoneBestModels and os.path.exists("TrainedModels/EveryModel/policyModel%s.model" % fileNumber):
            fileNumber += 1
            filePath = "TrainedModels/EveryModel/policyModel" + str(fileNumber) + ".model"

        if i % 2 == 0 and saveNoneBestModels:
            self.policy_value_net.save_model(filePath)
        if (i + 1) % self.check_freq == 0:
            print("current self-play batch: {}".format(i + 1))
            if self.bestPolicy_value_net != None:
                win_ratio = self.policy_evaluate()
                # self.policy_value_net.save_model('./current_policy.model')
                if win_ratio > 0.55 or self.bestPolicy_value_net == None:
                    print("New best policy!!!!!!!!")
                    # self.best_win_ratio = win_ratio
                    # update the best_policy
                    bestFilePath = "TrainedModels/BestModels/policyModel" + str(fileNumber) + ".model"
                    self.policy_value_net.save_model(bestFilePath)
                    self.setBestFile()# deprecated
            else:
                bestFilePath = "TrainedModels/BestModels/policyModel" + str(fileNumber) + ".model"
                self.policy_value_net.save_model(bestFilePath)
                self.setBestFile()#deprecated
    # to be deprecated in favour a model that collects training data in parralel then trains using that training data later
    def collectDataInParralel(self,selfPlayData):
        self.collect_selfplay_data(self.play_batch_size)
        print(len(self.data_buffer))
        selfPlayData.extend(self.data_buffer)

        #unproxiedSharedData = list(selfPlayData)
        #if len(selfPlayData)>self.buffer_size*0.75:
        #    print(self.buffer_size/2)
        #    selfPlayData[:] =[]
        #else:
            #self.data_buffer = deque(unproxiedSharedData)

    def trainingData(self):
        if len(self.data_buffer) > self.batch_size:
            loss, entropy = self.policy_update()
    def evaluateData(self,fileNumber=0,i=0):
        print("evaluating stage")
        print("current self-play batch: {}".format(i + 1))
        if self.bestPolicy_value_net != None:
            win_ratio = self.policy_evaluateinParralel()
            # self.policy_value_net.save_model('./current_policy.model')
            if win_ratio > 0.55 or self.bestPolicy_value_net == None:
                print("New best policy!!!!!!!!")
                bestFilePath = "TrainedModels/BestModels/policyModel" + str(fileNumber) + ".model"
                self.policy_value_net.save_model(bestFilePath)
                #self.setBestFile()
        else:
            print("saving current policy because there is no other policy to play against")
            bestFilePath = "TrainedModels/BestModels/policyModel" + str(fileNumber) + ".model"
            self.policy_value_net.save_model(bestFilePath)
            #self.setBestFile()

    def runSelfPlayInParralelWithManager(self,selfPlayData,mBestPolicy,i, saveNoneBestModels=False):
        self.collect_selfplay_data(self.play_batch_size)
        selfPlayData.extend(self.data_buffer)
        unproxiedSharedData = list(selfPlayData)
        self.bestPolicy_value_net = mBestPolicy.value
        if len(selfPlayData)>self.buffer_size*0.75:
            print(self.buffer_size/2)
            selfPlayData[:] =[]
        else:
            self.data_buffer = deque(unproxiedSharedData)

        print("batch i:{}, episode_len:{}".format(
            i + 1, self.episode_len))
        if len(self.data_buffer) > self.batch_size:
            loss, entropy = self.policy_update()

        fileNumber = 0
        if(i+1) % self.check_freq == 0:
            print("evaluating stage")
            print("current self-play batch: {}".format(i + 1))
            if self.bestPolicy_value_net != None:
                win_ratio = self.policy_evaluate()
                # self.policy_value_net.save_model('./current_policy.model')
                if win_ratio > 0.55 or self.bestPolicy_value_net == None:
                    print("New best policy!!!!!!!!")
                    # self.best_win_ratio = win_ratio
                    # update the best_policy
                    bestFilePath = "TrainedModels/BestModels/policyModel" + str(fileNumber) + ".model"
                    self.policy_value_net.save_model(bestFilePath)

                    mBestPolicy.value = PolicyValueNet(self.board_width,
                                                               self.board_height,
                                                               model_file=bestFilePath)
                    #self.setBestFile()
            else:
                print("saving current policy because there is no other policy to play against" )
                bestFilePath = "TrainedModels/BestModels/policyModel" + str(fileNumber) + ".model"
                self.policy_value_net.save_model(bestFilePath)

                #self.setBestFile()  #may requires lock
                mBestPolicy.value = PolicyValueNet(self.board_width,
                                                   self.board_height,
                                                   model_file=bestFilePath)
    # collect data until the buffer is full
    # keep a recording of how many games were finished
    # train the data
    # evaluate the data
    def runWithManagerModularised(self):
        policyUpdateCycles = 10
        for x in range(10):
            with torch.multiprocessing.Manager() as manager:

                ## collecting data
                executor = ProcessPoolExecutor(self.CPUCount)
                mlist = manager.list()
                futures = [executor.submit(self.collectDataInParralel,mlist) for i in range(self.game_batch_num)]
                executor.shutdown(wait=True)
                unproxiedSharedData = list(mlist)
                self.data_buffer = deque(unproxiedSharedData)
                print(len(mlist))
                # train data


            for i in range(policyUpdateCycles):
                print(len(self.data_buffer))
                print("updating policy")
                self.policy_update()
            bestFilePath = "TrainedModels/BestModels/policyModel" + str(0) + ".model"
            self.policy_value_net.save_model(bestFilePath)
        #print("evaluating policy")
        #self.evaluateData() deterministic so no point

    # deprecated
    def runWithmanager(self):
        with torch.multiprocessing.Manager() as manager:
            executor = ProcessPoolExecutor(self.CPUCount)
            mlist = manager.list()
            mBestPolicy = manager.Value('A',self.bestPolicy_value_net)
            futures = [executor.submit(self.runSelfPlayInParralelWithManager,mlist,mBestPolicy, i) for i in range(self.game_batch_num)]
            executor.shutdown(wait=True)
    # deprecated
    def run(self):
        """run the training pipeline"""
        try:

            saveNoneBestModels = False
            executor = concurrent.futures.ProcessPoolExecutor(6)
            futures = [executor.submit(self.runSelfPlayInParralel, i) for i in range(self.game_batch_num)]
            concurrent.futures.wait(futures)

            """
            for i in range(self.game_batch_num):
                self.runSelfPlayInParralel(i)
            """
        except KeyboardInterrupt:
            print('\n\rquit')


if __name__ == '__main__':
    if not os.path.exists("TrainedModels"):
        os.mkdir("TrainedModels")
        os.mkdir("TrainedModels/BestModels")
        os.mkdir("TrainedModels/EveryModel")
    training_pipeline = TrainPipeline()
    #training_pipeline.run()
    training_pipeline.runWithManagerModularised()
