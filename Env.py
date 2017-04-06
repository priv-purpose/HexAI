#!/usr/bin/python

'''HexEnv is insufficient as it is. Need wrapper to make it functional'''

import sys
import numpy as np
from gym import spaces
from gym.envs.board_game import HexEnv

class ModHexEnv(HexEnv):
    def __init__(self, player_color, opponent, observation_type, illegal_move_mode, board_size):
        """
        Args:
            player_color: Stone color for the agent. Either 'black' or 'white'
            opponent: An opponent policy
            observation_type: State encoding
            illegal_move_mode: What to do when the agent makes an illegal move. Choices: 'raise' or 'lose'
            board_size: size of the Hex board
        """
        assert isinstance(board_size, int) and board_size >= 1, 'Invalid board size: {}'.format(board_size)
        self.board_size = board_size

        colormap = {
            'black': HexEnv.BLACK,
            'white': HexEnv.WHITE,
        }
        try:
            self.player_color = colormap[player_color]
            self.real_player_color = player_color
        except KeyError:
            raise error.Error("player_color must be 'black' or 'white', not {}".format(player_color))

        self.opponent = opponent
        self._seed()

        assert observation_type in ['numpy3c']
        self.observation_type = observation_type

        assert illegal_move_mode in ['lose', 'raise']
        self.illegal_move_mode = illegal_move_mode

        if self.observation_type != 'numpy3c':
            raise error.Error('Unsupported observation type: {}'.format(self.observation_type))

        # One action for each board position and resign
        self.action_space = spaces.Discrete(self.board_size ** 2 + 1)
        observation = self.reset()
        self.observation_space = spaces.Box(np.zeros(observation.shape), np.ones(observation.shape))
    
    def get_board(self):
        '''Gets board state back
        funny that this simple function does not exist...'''
        return self.state
    
    def _render(self, mode='human', close=False):
        if close:
            return
        board = self.state
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        outfile.write(' ' * 5)
        for j in range(board.shape[1]):
            outfile.write(' ' +  hex(j)[2:] + '  | ')
        outfile.write('\n')
        outfile.write(' ' * 5)
        outfile.write('-' * (board.shape[1] * 6 - 1))
        outfile.write('\n')
        for i in range(board.shape[1]):
            outfile.write(' ' * (2 + i * 3) +  hex(i)[2:] + '  |')
            for j in range(board.shape[1]):
                if board[2, i, j] == 1:
                    outfile.write('     ')
                elif board[0, i, j] == 1:
                    outfile.write('  \033[96mB\033[0m  ')
                else:
                    outfile.write('  \033[1mW\033[0m  ')
                outfile.write('|')
            outfile.write('\n')
            outfile.write(' ' * (i * 3 + 1))
            outfile.write('-' * (board.shape[1] * 7 - 1))
            outfile.write('\n')

        if mode != 'human':
            return outfile    

def HexGameEnv(opponent):
    '''returns ModHexEnv set with player_color random and 
    opponent set to opponent'''
    player_color = 'black' if np.random.random() < .5 else 'white'
    return ModHexEnv(player_color = player_color,
                     opponent = opponent,
                     observation_type = 'numpy3c',
                     illegal_move_mode = 'raise',
                     board_size = 11)