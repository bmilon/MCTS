# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 08:05:07 2021

@author: mbhattac
"""
#%%
import time
import tracemalloc

import numpy as np
from mctspy.tree.nodes import TwoPlayersGameMonteCarloTreeSearchNode
from mctspy.tree.search import MonteCarloTreeSearch
from mctspy.games.examples.tictactoe import TicTacToeGameState
import threading

simulations=[1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,15000,20000,25000,30000]
naive_MCTS_time=[]
naive_MCTS_peak_memory=[]
naive_MCTS_average_memory=[]

#%%%
for s in simulations:
    start =time.process_time()
    tracemalloc.start()
    
    state = np.zeros((3,3))
    initial_board_state = TicTacToeGameState(state = state, next_to_move=1)
    
    root = TwoPlayersGameMonteCarloTreeSearchNode(state = initial_board_state)
    mcts = MonteCarloTreeSearch(root)
    best_node = mcts.best_action(s)
    best_node = mcts.best_action(s)
    
    current, _peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()    

    naive_MCTS_time.append((time.process_time()- start))
    naive_MCTS_peak_memory.append((_peak / 10**6 ))    
    naive_MCTS_average_memory.append((current / 10**6 ))    
    
    print( 'Time taken (seconds): ',(time.process_time()- start),'Memory used (Average): (MB)' ,(current / 10**6),'Memory used (Peak): (MB)' ,(_peak / 10**6 ))
    
raise SystemExit   



#%%%

