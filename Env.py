#!/usr/bin/python

'''HexEnv is insufficient as it is. Need wrapper to make it functional'''

import sys
import numpy as np
from gym import spaces
from gym.envs.board_game import HexEnv
# for randomEp func
import theano

BOARD_SIZE = 11

class ModHexEnv(HexEnv):
    '''Modified in many ways from original HexEnv...
    1. Render changed
    2. Any player can come first
    3. action sequence also part of state'''
    
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
        self.move_history = []

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
    
    def _reset(self):
        self.state = np.zeros((3, self.board_size, self.board_size))
        self.state[2, :, :] = 1.0
        self.to_play = HexEnv.BLACK
        self.done = False

        # Let the opponent play if it's not the agent's turn
        if self.player_color != self.to_play:
            a = self.opponent_policy(self.state, self.move_history)
            HexEnv.make_move(self.state, a, HexEnv.BLACK)
            self.move_history.append(a)
            self.to_play = HexEnv.WHITE
        return self.state
    
    def get_move_hist(self):
        return self.move_history
    
    def _step(self, action):
        assert self.to_play == self.player_color
        # If already terminal, then don't do anything
        if self.done:
            return self.state, 0., True, {'state': self.state}

        # if HexEnv.pass_move(self.board_size, action):
        #     pass
        if HexEnv.resign_move(self.board_size, action):
            return self.state, -1, True, {'state': self.state}
        elif not HexEnv.valid_move(self.state, action):
            if self.illegal_move_mode == 'raise':
                raise
            elif self.illegal_move_mode == 'lose':
                # Automatic loss on illegal move
                self.done = True
                return self.state, -1., True, {'state': self.state}
            else:
                raise error.Error('Unsupported illegal move action: {}'.format(self.illegal_move_mode))
        else:
            HexEnv.make_move(self.state, action, self.player_color)
            self.move_history.append(action)

        # Opponent play
        a = self.opponent_policy(self.state, self.move_history)

        # if HexEnv.pass_move(self.board_size, action):
        #     pass

        # Making move if there are moves left
        if a is not None:
            if HexEnv.resign_move(self.board_size, a):
                return self.state, 1, True, {'state': self.state}
            else:
                HexEnv.make_move(self.state, a, 1 - self.player_color)
                self.move_history.append(a)

        reward = HexEnv.game_finished(self.state)
        if self.player_color == HexEnv.WHITE:
            reward = - reward
        self.done = reward != 0
        return self.state, reward, self.done, {'state': self.state}    
    
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
    

class SimHexEnv(ModHexEnv):
    '''HexEnv used for simulation in MCTS.'''
    def __init__(self, player_color, opponent, observation_type, \
                 illegal_move_mode, board_size):
        super(SimHexEnv, self).__init__(player_color, opponent, observation_type, \
                                        illegal_move_mode, board_size)
        self._rand_stream = theano.tensor.shared_randomstreams.RandomStreams()
    
    def set_start(self, state, history = None):
        '''This function shall not be in normal HexEnvs, as it should not be
        allowed for actors to be able to change the state as they want. Thus,
        this function is only valid for simulation.'''
        self.done = False
        self.state = np.copy(state)
        self.move_history = [] if history is None else history[:]
    
    def runEp(self, players, turn, lgl_mvs):
        giveup_move = self.state.shape[1]**2
        lgl_mvs_cpy = lgl_mvs[:]
        while True:
            new_move = players[turn].as_func(self.state, self.move_history, lgl_mvs_cpy)
            if new_move == giveup_move: break
            self.make_move(self.state, new_move, turn)
            self.move_history.append(new_move)
            lgl_mvs_cpy.remove(new_move)
            turn = 1-turn
        return self.game_finished(self.state)
    
    def randomEp(self, turn, lgl_mvs):
        '''Optimized version of runEp for MC'''
        np.random.shuffle(lgl_mvs)
        first = np.zeros((BOARD_SIZE**2,))
        first[lgl_mvs[::2]] = 1
        sec = np.zeros((BOARD_SIZE**2,))
        sec[lgl_mvs[1::2]] = 1
        self.state[turn] += first.reshape((BOARD_SIZE, BOARD_SIZE))
        self.state[1-turn] += sec.reshape((BOARD_SIZE, BOARD_SIZE))
        self.state[2] = np.zeros((BOARD_SIZE, BOARD_SIZE))
        return self.rand_game_finished(self.state)
    
    @staticmethod
    def get_turn(state):
        '''In normal HexEnvs, it should not be necessary for one to access
        who's turn it is, as it is always the player's turn. However, this 
        is a necessary function in simulation. '''
        return int(np.sum(state[0]) != np.sum(state[1]))
    
    @staticmethod
    def rand_game_finished(board):
        # Returns 1 if player 1 wins, -1 if player 2 wins
        d = board.shape[1]

        inpath = set()
        newset = set()
        for i in range(d):
            if board[0, 0, i] == 1:
                newset.add(i)

        while len(newset) > 0:
            for i in range(len(newset)):
                v = newset.pop()
                inpath.add(v)
                cx = v // d
                cy = v % d
                # Left
                if cy > 0 and board[0, cx, cy - 1] == 1:
                    v = cx * d + cy - 1
                    if v not in inpath:
                        newset.add(v)
                # Right
                if cy + 1 < d and board[0, cx, cy + 1] == 1:
                    v = cx * d + cy + 1
                    if v not in inpath:
                        newset.add(v)
                # Up
                if cx > 0 and board[0, cx - 1, cy] == 1:
                    v = (cx - 1) * d + cy
                    if v not in inpath:
                        newset.add(v)
                # Down
                if cx + 1 < d and board[0, cx + 1, cy] == 1:
                    if cx + 1 == d - 1:
                        return 1
                    v = (cx + 1) * d + cy
                    if v not in inpath:
                        newset.add(v)
                # Up Right
                if cx > 0 and cy + 1 < d and board[0, cx - 1, cy + 1] == 1:
                    v = (cx - 1) * d + cy + 1
                    if v not in inpath:
                        newset.add(v)
                # Down Left
                if cx + 1 < d and cy > 0 and board[0, cx + 1, cy - 1] == 1:
                    if cx + 1 == d - 1:
                        return 1
                    v = (cx + 1) * d + cy - 1
                    if v not in inpath:
                        newset.add(v)
        return -1

def HexGameEnv(opponent):
    '''returns ModHexEnv set with player_color random and 
    opponent set to opponent'''
    player_color = 'black' if np.random.random() < .5 else 'white'
    return ModHexEnv(player_color = player_color,
                     opponent = opponent,
                     observation_type = 'numpy3c',
                     illegal_move_mode = 'raise',
                     board_size = BOARD_SIZE)
