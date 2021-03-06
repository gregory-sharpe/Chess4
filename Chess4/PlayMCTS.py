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


def getBestFile():
    list_of_files = glob.glob('TrainedModels/BestModels/*')  # * means all if need specific format then *.csv
    # requires atleast 1 file to be in best Models
    if len(list_of_files) > 0:
        bestFile = max(list_of_files, key=os.path.getctime)

        return PolicyValueNet(14,
                              14,
                              model_file=bestFile)
    else:
        return None
if __name__ == "__main__" :
    bestPolicy = getBestFile()
    g = Game()
    players = g.gs.allPlayers
    for i in range(1,4):
        players[i].isHumanPlaying=False
        players[i].isMcts = True
        players[i].isBestMcts=True
        print(players[i].isHumanPlaying)
    if bestPolicy == None:
        g.startGame(DontCreateGS=True)
    else:
        c_punct = 1
        n_playout = 100
        temp = 1e-3
        bestMCTSPlayer = MCTSPlayer(bestPolicy.policy_value_fn,
                                         c_puct=c_punct,
                                         n_playout=n_playout)
        #bestMCTSPlayer = MCTS_Pure(c_puct=5,
        #                             n_playout=100)
    g.startGame(bestMCTSPlayer,DontCreateGS=True)
    #main()
