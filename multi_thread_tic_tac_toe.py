# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 07:29:44 2021

@author: mbhattac
"""
import time 
import numpy as np
import itertools
from collections import defaultdict
import copy
import threading
import math
import random
import string 
import time
import tracemalloc


def getRandomId(N=4):
    return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(N))

class Player():    
    def __init__(self,id):
        self.id= id
    def __str__(self):
        return 'PLayer nos:' + str(self.id)

    
class sharedGameTree():
    def __init__(self,state):
        self.state = state 
    
            
class Game():    
    def __init__(self,actions,players):
        self.actions = actions
        self.players = players        
        
    def getPayoff(self,state,action):
        return 0     
    def getAllPossibleMoves(self,state,player):
        valid_moves = []
        for a in self.actions:
            if state[a.x][a.y] ==  -1 :
                valid_moves.append(a)
        return valid_moves
    
    def getBestMove(self,state,player):
        moves = []
        payoffs =[]
        for a in self.getAllPossibleMoves(state,player):
                payoffs.append(self.getPayoff(state,a))
                moves.append(a)
        return moves[payoffs.index(max(payoffs))] 

    
    def GameResult(self,state):
        board_size = state.shape[0]
        # check if game is over
        #print(state)
        #state=np.array([[-1,-1,0], [0,0,0], [0,0,0]])
        for p in self.players:
            pass
            pid = p.id
            #print('checking for player',pid)
            for rows in state:
                if (rows == [pid , pid , pid]).all():
                    #print('row returning',pid)
                    return pid 
            for cols in state.T:
                if (cols == [pid , pid , pid]).all():
                    #print('col returning',pid)                    
                    return pid
            if (np.diag(state)  == [pid , pid , pid]).all()  or( np.diag(np.flip(state,1))  == [pid , pid , pid]  ).all():
                    #print('diag returning ',pid)                
                    return pid
                
        if np.all(state != -1):
            #print('returning :',np.count_nonzero(state == self.players[0].id ) / np.count_nonzero(state == self.players[1].id )* 0.001) 
            return -1
            #return np.count_nonzero(state == self.players[0].id ) / np.count_nonzero(state == self.players[1].id ) * 0.001                
        
        return None
    #state=np.array([[0., 1., 1.], [1., 0., 0.],[1., 0. ,0.]])
    def move(self,action,state,player):
        #print('in move for ',player)                
        new_state = state.copy()
        new_state[action.x][action.y]=player
#        print(new_state)
        return new_state 
    
class  action():
    def __init__(self,x,y):
        self.x = x
        self.y = y
        

class MCTS_Node():

    def __init__(self, state,player,count_of_players,game,actions, parent=None):
        self.state = state
        self.parent = parent
        self.children = []        
        self._number_of_visits = 0.
        self._results = defaultdict(int)
        self._untried_actions = actions
        self.game= game
        self.wins = 0 
        self.whos_turn = player  
        self.players_c = count_of_players
        self.id = getRandomId()

    def __str__(self):
        return self.id 

    def is_terminal_node(self):
        return len(self.children) == 0 

    def is_fully_expanded(self):
        return len(self._untried_actions) == 0

    def q(self):
        wins = self._results[self.parent.state.next_to_move]
        loses = self._results[-1 * self.parent.state.next_to_move]
        return wins - loses

    
    def expand(self):
        #print('in expand------------------------------------')
        if len(self._untried_actions)  == 0 :
            return self
        action = self._untried_actions.pop()
        next_state = self.game.move(action,self.state,self.whos_turn)
        next_player = (self.whos_turn+1)%self.players_c 
        child_node = MCTS_Node(state = next_state,game=self.game,count_of_players=self.players_c ,parent=self,player=next_player,actions=self.game.getAllPossibleMoves(next_state,next_player) )
        self.children.append(child_node)
        #print(len(self.children))
        return child_node
    
    def rollout_policy(self, possible_moves):    
        #print('rollout_policy')        
        return possible_moves[np.random.randint(len(possible_moves))]

    def rollout(self):
        #print('rollout')                
        current_rollout_state = self.state
        who_played = self.whos_turn
        while not self.game.GameResult(current_rollout_state) :
            #time.sleep(0.1)
            #print('current state\n',current_rollout_state)
            possible_moves = self.game.getAllPossibleMoves(current_rollout_state ,who_played)
            if len(possible_moves) == 0 :
                break
            action = self.rollout_policy(possible_moves)
            #print('next action',action.x,action.y)
            current_rollout_state = game.move(action,current_rollout_state,who_played)            
            who_played=(who_played+1)%self.players_c
        return self.game.GameResult(current_rollout_state)
    
    def backpropagate(self, result):
        #print('backpropagate')                
        
        self._number_of_visits += 1.
        self._results[result] += 1.
        if self.parent:
            self.parent.backpropagate(result)

    def best_child(self, player_id,c_param=0.5):
        #print('best_child')               
        
        #print(self._number_of_visits)
        #print([c._results[player_id] for c in self.children])
        # if np.any([c._number_of_visits for c in self.children]):
        #     print('returning')
        #     return random.choice(self.children)
        choices_weights = [(c._results[player_id])/(c._number_of_visits) + c_param*math.sqrt(math.log(self._number_of_visits,2)/(c._number_of_visits))  for c in self.children]
        
        #print(choices_weights)
        #print([c._number_of_visits for c in self.children])
        return self.children[np.argmax(choices_weights)]
    


class ParallelMonteCarloTreeSearch(object):

    def __init__(self, node,player):
        self.root = node
        self.player = player 
    def best_action(self, simulations_number,player_id):
        for _ in range(0, simulations_number):            
            #print('simulation nos ',_)
            v = self._tree_policy(player_id)
            reward = v.rollout()
            #print('reward',reward)
            v.backpropagate(reward)
        # to select best child go for exploitation only
        return self.root.best_child(c_param=0.,player_id=player_id)
    def _tree_policy(self,player_id):
        #print('_tree_policy')                                

        current_node = self.root        
        while current_node.is_fully_expanded():
            current_node = current_node.best_child(player_id=player_id)
        current_node=current_node.expand()

        return current_node    
        
        # while not current_node.is_fully_expanded() or not current_node.is_terminal_node() :            
        #     if not current_node.is_fully_expanded():
        #         current_node.expand()
            
        # while not current_node.is_terminal_node():                
        #     current_node = current_node.best_child(player_id=player_id)
        
        # if current_node.is_fully_expanded():
        #     while not current_node.is_terminal_node():            
        #         current_node = current_node.best_child(player_id=player_id)
        # else:
        #     pass
        
        
        # current_node = self.root
        # while not self.root.is_fully_expanded():
        #     #print( self.root._untried_actions)
        #     current_node.expand()
            
    
def print_node(root):
    s = root.id + '>'
    for c in root.children:
        s +=  "-" + c.id
        print_node(c)
        
    print(s)

def thread_wrapper(root,runs,player_id):
    mcts = ParallelMonteCarloTreeSearch(root,player_id)    
    mcts.best_action(runs,player_id=player_id)
    
#if __name__  ==  '__main__'    :
    
    
players = 2
all_players=[]

# create all players
for _ in range(players):
    all_players.append(Player(_))

game_size = 3

current_state = np.zeros((game_size ,game_size ))
current_state[:]=-1

possible_tta =  [action(a,b) for (a,b) in list(itertools.product(range(3),range(3)))]

game = Game(actions=possible_tta,players=all_players)
root  =  MCTS_Node(state = current_state,game=game,count_of_players=len(all_players),player=0,actions=game.actions.copy())
#mcts = ParallelMonteCarloTreeSearch(root,all_players[0])
#mcts.best_action(10000,player_id='0')
start = time.process_time()
tracemalloc.start()


simulations=[1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,15000,20000,25000,30000]
parallel_MCTS_time=[]
parallel_MCTS_peak_memory=[]
parallel_MCTS_average_memory=[]

for s in simulations:
    start =time.process_time()
    tracemalloc.start()
    
    for p in all_players:
        mcts = ParallelMonteCarloTreeSearch(root,p)
        x = threading.Thread(target=thread_wrapper, args=(root,s,str(p.id)))    
        x.start()
        x.join()            
    current, _peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()    

    parallel_MCTS_time.append((time.process_time()- start))
    parallel_MCTS_peak_memory.append((_peak / 10**6 ))    
    parallel_MCTS_average_memory.append((current / 10**6 ))    
    
    print( 'Time taken (seconds): ',(time.process_time()- start),'Memory used (Average): (MB)' ,(current / 10**6),'Memory used (Peak): (MB)' ,(_peak / 10**6 ))
   
raise SystemExit     

def display_matrix(mat):
    def local_map(a):
        m={}
        m[0]='O'
        m[-1]='_'
        m[1]='X'        
        return m[a]
    
    local = mat.copy()
    local = np.vectorize(local_map)(local)
    print(local)
    pass

for p in all_players:
    print('For Player ',p.id)
    temp=root
    while len(temp.children)> 0 :
        if temp.whos_turn == p.id:
            print("Player's turn : ",temp.whos_turn)
            display_matrix(temp.state)
        
        temp=temp.best_child(player_id=str(p.id))
    
    
    
# vary C        
