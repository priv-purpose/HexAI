#!/usr/bin/python

'''Hex REINFORCE implementation
But really, it's annoying that network implementation seems so hard'''

import theano
import theano.tensor as T
from theano.tensor.nnet import conv2d, relu
import numpy as np
import itertools, time, random, os

from Env import HexGameEnv
from MLP import HiddenLayer
from Random import *

## Global variable (somewhat nuisance, later make it all come from main)
BOARD_SIZE = 11
LAYER_NUM = 2

## Shortcuts
dbg = False
mode = 1
brain_dir = 'brains/'
init_brain = 'GTX2B02_25000.pkl'
dbg_oppo_brain = 'GTX2B01__final.pkl'
new_batch_prefix = 'GTX3B01_'
train_num = 30000
minibatch_size = 50

print 'Initialized with', init_brain
print 'new_batch_prefix:', new_batch_prefix
print 'train_num:', train_num
print 'batch_size:', minibatch_size

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
            value = np.zeros(filter_shape, dtype = theano.config.floatX),
            borrow = True,
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
        filter_shape = (1, image_shape[1], 1, 1) # TODO > Inappropriate for batch
        
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
            value = np.zeros(filter_shape, dtype = theano.config.floatX),
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
    def __init__(self, filter_num, layer_num, lmbda = 1., learn_rate = .0,
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
        reward = T.scalar()
	batch_size = T.scalar()
        
        # stuff for functions
        param_pre_list = [layer.params for layer in self.layers]
        self.params = list(itertools.chain.from_iterable(param_pre_list))
        elig_pre_list = [layer.eligs for layer in self.layers]
        self.accum_eligs = list(itertools.chain.from_iterable(elig_pre_list))
        self.temp_eligs = [theano.shared(
                               value = np.zeros(
                                   elig.get_value().shape,
                                   dtype = theano.config.floatX), 
                               borrow = True)
                           for elig in self.accum_eligs]
        
        # functions
        self.f_pass = theano.function(
            inputs = [inpt, self.legal],
            outputs = prob,
            allow_input_downcast = True
        )
        
        self.elig_updator = theano.function(
            inputs = [inpt, self.legal, self.selected],
            outputs = None,
            updates = [(elig, lmbda*elig + self.elig_calc(param))
                       for elig, param in zip(self.temp_eligs, self.params)],
            allow_input_downcast = True
        )
	
	self.elig_finalize = theano.function(
	    inputs = [reward],
	    outputs = None,
	    updates = [(elig, (reward-self.base_r)*elig)
	               for elig in self.temp_eligs]
	)
        
        self.Aelig_updator = theano.function(
            inputs = [],
            outputs = None,
            updates = [(Aelig, Aelig + Telig)
                       for Aelig, Telig in zip(self.accum_eligs, self.temp_eligs)]
        )

        self.val_updator = theano.function(
            inputs = [batch_size],
            outputs = None,
            updates = [(param, self.val_calc(param, elig, batch_size))
                       for elig, param in zip(self.accum_eligs, self.params)] #TODO
        )

        self.Aelig_clear = theano.function(
            inputs = [],
            outputs = None,
            updates = [(Aelig, 0*Aelig) for Aelig in self.accum_eligs]
        )
        
        self.elig_clear = theano.function(
            inputs = [],
            outputs = None,
            updates = [(elig, 0*elig) for elig in self.temp_eligs]
        )
    
    def elig_calc(self, theta):
        '''calculates the elig related to theta'''
        return T.grad(T.log(self.g), wrt=theta)
    
    def val_calc(self, theta, elig, batch_size):
	return theta + (self.learn_rate/batch_size)*elig
    
    def legal_moves(self, env_input):
        '''computes legal moves from the env input. 
        (this is actually already done by the environment.)'''
        return np.ndarray.flatten(env_input[2])
    
    def make_move(self, env_input):
        lgl_mvs = self.legal_moves(env_input)
        probs = self.f_pass(env_input, lgl_mvs)[0]
        #print np.max(probs)
        select = np.random.choice(range(BOARD_SIZE**2), p = probs)
        self.elig_updator(env_input, lgl_mvs, select)
        return select
    
    def runEp(self, opponent = 'random', verbose = False):
        '''not finished runEp'''
        env = HexGameEnv(opponent)
        if verbose:
            print 'REINFORCE is %s' % env.real_player_color
        board = env.get_board()
        end = env.game_finished(board)
        
        ns = board
        while not end:
            move = self.make_move(ns)
            ns, rw, end, _ = env.step(move)
            if verbose:
                env.render()
                lgl_mvs = self.legal_moves(ns)
                probs = self.f_pass(ns, lgl_mvs)[0]
                np.set_printoptions(formatter={'float': '{: .2f}'.format})
                print np.array(probs.reshape((BOARD_SIZE, BOARD_SIZE)))
                raw_input('Enter to see next move')
        #self.val_updator(rw) # gone, because of batch learning
        self.elig_finalize(rw) # finalize elig with reward
        self.Aelig_updator()
        self.elig_clear()
        return rw
    
    def runBatch(self, oppos, batch_size = minibatch_size):
	res = {1.: 0, -1.: 0}
	for i in xrange(batch_size):
	    oppo = random.choice(oppos) # choose randomly from opponent list
	    tmp_res = self.runEp(opponent = oppo.as_func)
	    res[tmp_res] += 1
	# now, batch_size-worth elig is kept in eligs. this is resolved:
	self.val_updator(batch_size)
        self.Aelig_clear()
	# now elig is clear, and we are ready for next batch. 
	# return res to show progress. 
	return res
    
    def as_func(self, board, hist):
        lgl_mvs = self.legal_moves(board)
        probs = self.f_pass(board, lgl_mvs)[0]
        select = np.random.choice(range(BOARD_SIZE**2), p = probs)
        return select
    
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
	    print val.shape
            p.set_value(val)
        f.close()

class REINFORCEHexPriorGenerator(REINFORCEHexPlayer):
    '''Class to generate prior distribution for MCTS, identical to REINFORCE
    except for below function'''
    def prior_dist(self, board):
        '''returns probs'''
        lgl_mvs = self.legal_moves(board)
        return self.f_pass(board, lgl_mvs)[0]

def ensembleTrain(re_player, seeds, save_prefix,
                  batch_num = 1000, save_interval = 100):
    # with batch training, this function serves to making opponents as needed
    t = trange(batch_num, desc='Bar desc', leave = True)
    t.set_description('%03d:%03d ' % (0, 0))
    for i in t:
	batch_res = re_player.runBatch(seeds)
	t.set_description('%03d:%03d ' % (batch_res[1], batch_res[-1]))
        if i != 0 and i % save_interval == 0:
            re_player.export_val(save_prefix + str(i) + '.pkl')
            new = REINFORCEHexPlayer(filter_num = 50, layer_num = LAYER_NUM, 
	                             learn_rate = 0.)
            new.import_val(save_prefix + str(i) + '.pkl')
            seeds.append(new)
    re_player.export_val(save_prefix + '_final.pkl')
    # doesn't return anything

rng = np.random.RandomState(1236) # I'm okay with this

def test(my_name, oppo_name):
    ################### NOTE: BELOW LEARN RATE 0
    ba = REINFORCEHexPlayer(filter_num = 50, layer_num = 2, learn_rate = .0000) 
    ba.import_val(brain_dir + my_name)
    
    ref_oppo = REINFORCEHexPlayer(filter_num = 50, layer_num = 2, learn_rate = .0)
    ref_oppo.import_val(brain_dir + oppo_name)
    #ref_oppo = RandomHexPlayer()
    
    if dbg:
        #ba = RandomHexPlayer()
        res = ba.runEp(opponent = ref_oppo.as_func, verbose = True)
        print res
    else:
        res, res_order = logGames(ba, ref_oppo, game_num = 100)
        print res
        graphWins(res_order, games_over = 50, title = 'tmp_monitor')

def train():
    ba = REINFORCEHexPlayer(filter_num = 50, layer_num = LAYER_NUM, 
                            lmbda = 0.98, learn_rate = .001) 
    #ba.import_val(brain_dir + init_brain)
    print 'BRAIN NOT IMPORTED'

    ref_oppo = REINFORCEHexPlayer(filter_num = 50, layer_num = 2)
    ref_oppo.import_val(brain_dir + dbg_oppo_brain)
    #ref_oppo = RandomHexPlayer()
    #print 'USING RANDOM HEX PLAYER AS REFERENCE'

    oppos = [RolloutHexPlayer01()]
    '''print '... generating opponents'
    for f_name in os.listdir(brain_dir):
	if f_name[-3:] != 'pkl': continue
	op = REINFORCEHexPlayer(filter_num = 50, layer_num = 2)
	op.import_val(brain_dir + f_name)
	oppos.append(op)'''
    print 'NO OPPONENTS LOADED FROM BRAINS/'
    print len(oppos), 'opponents generated'
    _ = ensembleTrain(ba, oppos, brain_dir + new_batch_prefix, train_num, 500)
    #print res
    #graphWins(res_order, games_over = 100, title='tmp_monitor')
    #print logGames(ba, oppos[-10])
    #print logGames(ba, oppos[-50])
    print logGames(ba, ref_oppo)   

def humanPlay():
    comp = REINFORCEHexPlayer(filter_num = 50, layer_num = 2) 
    human = HumanHexPlayer()
    human.runEp(opponent = comp.as_func)

if __name__ == '__main__':
    if mode == 1:
    	train()
    elif mode == 2:
        test(init_brain, dbg_oppo_brain)
    elif mode == 3:
        humanPlay()
