#!/usr/bin/python

'''Hex REINFORCE implementation
But really, it's annoying that network implementation seems so hard'''

from Env import HexGameEnv
import theano
import theano.tensor as T
from theano.tensor.nnet import conv2d, relu
import numpy as np
import itertools

from MLP import HiddenLayer
from Random import *

## Global variable (somewhat nuisance, later make it all come from main)
BOARD_SIZE = 11

class HexConvLayer(object):
    '''Hex Convolutional layer imitating alphago
    filter_shape should be odd, otherwise board size changes'''
    def __init__(self, inpt, filter_shape, image_shape):
        '''Make HexConvLayer with internal params.'''
        assert filter_shape[2] == filter_shape[3]
        assert filter_shape[2] % 2 == 1
        assert image_shape[1] == filter_shape[1]
        
        fan_in = np.prod(filter_shape[1:])
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:]))
        W_bound = np.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            np.asarray(
                rng.uniform(
                    low = -W_bound,
                    high = W_bound,
                    size = filter_shape
                ),
                dtype = theano.config.floatX
            ),
            borrow = True
        )
        # elig trace
        self.W_e = theano.shared(
            value = np.zeros(filter_shape),
            borrow = True
        )
        b_values = np.zeros((filter_shape[0],), dtype = theano.config.floatX)
        self.b = theano.shared(value = b_values, borrow = True)
        b_e_values = np.zeros((filter_shape[0],), dtype = theano.config.floatX)
        self.b_e = theano.shared(value = b_e_values, borrow = True)
        
        conv_out = conv2d(
            input = inpt,
            filters = self.W,
            filter_shape = filter_shape,
            input_shape = image_shape,
            border_mode = 'half'
        )
        
        self.output = relu(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        
        # cleanup
        self.params = [self.W, self.b]
        self.eligs = [self.W_e, self.b_e]
        self.inpt = inpt

class HexEndConvLayer(object):
    def __init__(self, inpt, image_shape):
        '''1x1 convolution (also in alphago paper)
        along with different biases per point
        This layer yields a 1d array with size BOARD_SIZE**2
        
        After this network, you add rule-keeping softmax'''
        filter_shape = (1, image_shape[1], 1, 1)
        
        fan_in = np.prod(filter_shape[1:])
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:]))
        W_bound = np.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            np.asarray(
                rng.uniform(
                    low = -W_bound,
                    high = W_bound,
                    size = filter_shape
                ),
                dtype = theano.config.floatX
            ),
            borrow = True
        )
        self.W_e = theano.shared(
            value = np.zeros(filter_shape),
            borrow = True
        )
        # now things get different
        b_values = np.zeros((BOARD_SIZE**2,), dtype = theano.config.floatX)
        self.b = theano.shared(value = b_values, borrow = True)
        b_e_values = np.zeros((BOARD_SIZE**2,), dtype = theano.config.floatX)
        self.b_e = theano.shared(value = b_e_values, borrow = True)
        
        conv_out = conv2d(
            input = inpt,
            filters = self.W,
            filter_shape = filter_shape,
            input_shape = image_shape,
            border_mode = 'half'
        )
        
        final_values = conv_out.flatten(1) # now final_values should be (121,)
        # note that the above assumes minibatch_size = 1. Not sure how one can
        # do minibatch learning in reinforcement tasks. 
        self.output = final_values + self.b # this is also slightly diff
        # above DOES NOT use relu
        
        # cleanup
        self.params = [self.W, self.b]
        self.eligs = [self.W_e, self.b_e]
        self.inpt = inpt

class RuleKeepSoftmaxLayer(object):
    '''returns the probability of actions with rule-keeping in mind'''
    def __init__(self, inpt, legal_moves):
        '''inpt should be the output of the final layer, 
        legal_moves should be a vector in which the legal moves are set to 1, 
        illegal moves are set to 0'''
        # Assumes inpt has not been subject to softmax
        middle_softmax = T.nnet.softmax(inpt)
        legal_pre_prob = legal_moves * middle_softmax
        self.output = legal_pre_prob / T.sum(legal_pre_prob)
        
        # cleanup
        self.params = []
        self.inpt = inpt

class REINFORCEHexPlayer(object):
    '''Hex Player that makes random legal moves'''
    def __init__(self, filter_num, layer_num, lmbda = 1., learn_rate = .0003,
                 base_r = 0.):
        '''3s below are hardcoded...'''
        self.lmbda = lmbda
        self.learn_rate = learn_rate
        self.base_r = base_r
        # below is network building
        inpt = T.tensor3()
        inpt_shape = (1, 3, BOARD_SIZE, BOARD_SIZE)
        re_inpt = inpt.reshape(inpt_shape)
        
        # below is hardcode of ALPHAGO parameters
        board_shape = (BOARD_SIZE, BOARD_SIZE)
        first_k_size = (5, 5)
        later_k_size = (3, 3)
        
        # Finally we actually start building the layer. 
        self.layers = []
        layer0 = HexConvLayer(
            inpt = re_inpt,
            image_shape = inpt_shape,
            filter_shape = (filter_num, 3)+first_k_size,
        )
        
        self.layers.append(layer0) # add first layer, and then..
        # add the consecutive layers
        for i in xrange(1, layer_num):
            tmp_layer = HexConvLayer(
                inpt = self.layers[-1].output,
                image_shape = (1, filter_num) + board_shape,
                filter_shape = (filter_num, filter_num) + later_k_size,
            )
            self.layers.append(tmp_layer)
        # final layer sums up result to 1
        fin_layer = HexEndConvLayer(
            inpt = self.layers[-1].output,
            image_shape = (1, filter_num) + board_shape,
        )
        self.layers.append(fin_layer)
        
        # Policy-learning stuff
        self.legal = T.dvector()
        prob_layer = RuleKeepSoftmaxLayer(self.layers[-1].output, self.legal)
        prob = prob_layer.output
        self.selected = T.scalar(dtype = 'int32')
        self.g = prob[(0, self.selected)]
        reward = T.dscalar()
        
        # stuff for functions
        param_pre_list = [layer.params for layer in self.layers]
        self.params = list(itertools.chain.from_iterable(param_pre_list))
        elig_pre_list = [layer.eligs for layer in self.layers]
        self.eligs = list(itertools.chain.from_iterable(elig_pre_list))
        
        # functions
        self.f_pass = theano.function(
            inputs = [inpt, self.legal],
            outputs = prob
        )
        
        self.elig_updator = theano.function(
            inputs = [inpt, self.legal, self.selected],
            outputs = None,
            updates = [(elig, lmbda*elig + self.elig_calc(param))
                       for elig, param in zip(self.eligs, self.params)]
        )
        
        self.val_updator = theano.function(
            inputs = [reward],
            outputs = None,
            updates = [(param, self.R_update(param, reward, elig))
                       for elig, param in zip(self.eligs, self.params)]
        )
        
        self.elig_clear = theano.function(
            inputs = [],
            outputs = None,
            updates = [(elig, 0*elig) for elig in self.eligs]
        )
    
    def elig_calc(self, theta):
        '''calculates the elig related to theta'''
        return T.grad(T.log(self.g), wrt=theta)
    
    def R_update(self, theta, reward, elig):
        return theta + self.learn_rate*(reward-self.base_r)*elig
    
    def legal_moves(self, env_input):
        '''computes legal moves from the env input. 
        (this is actually already done by the environment.)'''
        return np.ndarray.flatten(env_input[2])
    
    def make_move(self, env_input):
        lgl_mvs = self.legal_moves(env_input)
        probs = self.f_pass(env_input, lgl_mvs)[0]
        select = np.random.choice(range(BOARD_SIZE**2), p = probs)
        self.elig_updator(env_input, lgl_mvs, select)
        return select
    
    def runEp(self, opponent = 'random'):
        '''not finished runEp'''
        env = HexGameEnv(opponent)
        board = env.get_board()
        end = env.game_finished(board)
        
        ns = board
        while not end:
            move = self.make_move(ns)
            ns, rw, end, _ = env.step(move)
        self.val_updator(rw)
        self.elig_clear()
        return rw
    
    def as_func(self, board):
        return self.make_move(board)
    
    def export_val(self, fname):
        import cPickle
        f = open(fname, 'wb')
        cPickle.dump([p.get_value() for p in self.params], f)
        f.close()
    
    def import_val(self, fname):
        import cPickle
        f = open(fname, 'rb')
        vals = cPickle.load(f)
        for p, val in zip(self.params, vals):
            p.set_value(val)
        f.close()

    
inpt = T.tensor3()
baba = inpt.reshape((1, 3, 11, 11))
rng = np.random.RandomState(1236) # I'm okay with this

print ' ... building REINFORCE '
ba = REINFORCEHexPlayer(filter_num = 50, layer_num = 2, learn_rate = .0003) 
ba.import_val('HexBrain.pkl')
print ' ... initializing env'

cho = RandomHexPlayer()
cho.runEp(opponent = ba.as_func)
#res, res_order = logGames(ba, cho, game_num = 1000)
#print res
#graphWins(res_order, games_over = 50, title='continue_train3')
ba.export_val('HexBrain.pkl')