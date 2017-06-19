#!/usr/bin/python

'''Random Player - For reference'''

from Env import HexGameEnv, SimHexEnv
import random
import numpy as np
from tqdm import trange
from itertools import compress

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
    
    def as_func(self, board, hist = None):
        '''shit, board[2] is 2d. change.'''
        blank = np.ndarray.flatten(board[2])
        poss_moves = list(compress(xrange(BOARD_SIZE**2), blank))
        try:
            return random.choice(poss_moves)
        except IndexError:
            return BOARD_SIZE**2

class RolloutHexPlayer01(RandomHexPlayer):
    def __init__(self):
        pass
    
    def runEp(self, opponent = 'random'):
        env = HexGameEnv(opponent)
        board = env.get_board()
        end = env.game_finished(board)
        while not end:
            _, rw, end, _ = env.step(self.as_func(board, env.move_history))
        return rw
    
    def as_funco(self, board, hist):
        '''Blocks "connected" territory'''
        nearby = [(-1, 0), (-1, 1), (0, 1), (1, 0), (1, -1), (0, -1)]
        if len(hist) > 0:
            nearby_1d = [hist[-1] + BOARD_SIZE*x + y for x, y in nearby]    
            past_move = (hist[-1] / BOARD_SIZE, hist[-1] % BOARD_SIZE)
            my_color = int(1 - board[1][past_move])
            around = [(past_move[0] + x, past_move[1] + y) for x, y in nearby]
            answs = []
            for idx in xrange(len(nearby_1d)):
                m_idx = (idx + 1) % 6
                e_idx = (idx + 2) % 6
                if around[idx][0] < 0 or around[idx][1] < 0: continue
                if around[idx][0] > 10 or around[idx][1] > 10: continue
                if around[e_idx][0] < 0 or around[e_idx][1] < 0: continue
                if around[e_idx][0] > 10 or around[e_idx][1] > 10: continue                
                if (board[my_color][around[idx]] == 1 and
                    board[2][around[m_idx]] == 1 and
                    board[my_color][around[e_idx]] == 1):
                    answs.append(nearby_1d[m_idx])
            if len(answs) != 0:
                return random.choice(answs)
        return super(RolloutHexPlayer01, self).as_func(board)

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
            try:
                ver_num, hor_num = raw_input('Human\'s move? ').split(',')
                ver_num = int(ver_num, 16)
                hor_num = int(hor_num, 16)
                _, rw, end, _ = env.step(BOARD_SIZE*ver_num+hor_num)
                env.render()
                print env.move_history
            except (IndexError, ValueError, TypeError) as e:
                print 'Try again.'
        return rw
    
    def as_func(self, board):
        pass

def logGames(player1, player2, game_num = 100):
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
    t.set_description('%03d:%03d ' % (res[1], res[-1]))
    return (res[1.], res[-1.]), res_order

def graphWins(res_order, games_over = 20, title = ''):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt    
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
if __name__ == '__main__':
    a = RolloutHexPlayer01()
    b = RolloutHexPlayer01()
    print logGames(a, b)
