#!/usr/bin/python

'''Random Player - For reference'''

from Env import HexGameEnv
import random
import numpy as np
import time
from tqdm import trange

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BOARD_SIZE = 11

class RandomHexPlayer(object):
    '''Hex Player that makes random legal moves'''
    def __init__(self):
        pass
    
    def runEp(self, opponent = 'random'):
        env = HexGameEnv(opponent)
        board = env.get_board()
        end = env.game_finished(board)
        while not end:
            possible_acts = env.get_possible_actions(board)
            _, rw, end, _ = env.step(random.choice(possible_acts))
        return rw
    
    def as_func(self, board):
        '''shit, board[2] is 2d. change.'''
        blank = np.ndarray.flatten(board[2])
        poss_moves = [i for i in xrange(BOARD_SIZE**2) if (blank[i] == 1)]
        try:
            return random.choice(poss_moves)
        except IndexError:
            return BOARD_SIZE**2

class HumanHexPlayer(object):
    '''Human Hex Player'''
    def __init__(self):
        pass
    
    def runEp(self, opponent = 'random'):
        env = HexGameEnv(opponent)
        env.render()
        board = env.get_board()
        end = env.game_finished(board)
        while not end:
            ver_num, hor_num = raw_input('Human\'s move? ').split(',')
            ver_num = int(ver_num, 16)
            hor_num = int(hor_num, 16)
            _, rw, end, _ = env.step(11*ver_num+hor_num)
            env.render()
        return rw
    
    def as_func(self, board):
        pass

def logGames(player1, player2, game_num = 1000):
    '''plays game_num games with player1 and player2. 
    returns a tuple where the first entry is the number of times player1 won,
    and the second entry is the number of times player2 won. '''
    res = {1.:0, -1.:0} # accumulates results here
    res_order = []
    t = trange(game_num, desc='Bar desc', leave = True)
    for i in t:
        if i % 20 == 0:
            t.set_description('%03d:%03d ' % (res[1], res[-1]))
        game_res = player1.runEp(opponent = player2.as_func)
        res[game_res] += 1
        res_order.append(game_res)
    return (res[1.], res[-1.]), res_order

def graphWins(res_order, games_over = 20, title = ''):
    '''will plot the win rate from RES_ORDER over GAMES_OVER games.'''
    mod = [max(0, res) for res in res_order]
    avg_res = [float(sum(mod[i:i+games_over]))/games_over
               for i in xrange(len(res_order)-(games_over-1))]
    plt.plot(range(games_over, len(res_order)+1), avg_res)
    axes = plt.gca()
    axes.set_ylim([0., 1.])
    plt.xlabel('Games')
    plt.ylabel('Win rate over past %d games' % games_over)
    plt.title('Win rate wrt. time for %s' % title)
    plt.savefig(title+'.png')

## Example usage of logGames (TODO: delete when you have main.py implemented).
#if __name__ == '__main__':
    #a = RandomHexPlayer()
    #b = RandomHexPlayer()
    #print logGames(a, b)