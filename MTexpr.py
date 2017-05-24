#!/usr/bin/python

'''Python Multiprocessing: Experiments for efficient "threading"
(quotation marks because I'm actually spawning multiple processes, not threads)

This experiments start from my MCTS code, as I wanted to search multiple
strands at the same time with a multi-core system.'''

from multiprocessing import Manager, Pool, Process
from multiprocessing.managers import BaseManager, BaseProxy
import numpy as np
from time import time, sleep
from Env import SimHexEnv
from Random import RandomHexPlayer, HumanHexPlayer

class DummyClass01(BaseProxy):
    class TreeNode(object):
        class TreeEdge(object):
            def __init__(self, pos, action, parent):
                self._pos = pos
                self._act = action
                self._N = 0.
                self._W = 0.
                self._leaf = None
                self._parent = parent
                self._post_pos = None
                self.load = False # is self._post_pos ready?

            def N(self):
                return self._N
            
            def Q(self):
                if self._N != 0:
                    return self._W / self._N
                else:
                    return 0.
        
            def update(self, sim_res):
                self._N += 1
                self._W += sim_res
            
            def lazy_postpos(self):
                self._post_pos = np.copy(self._pos)
                self._parent._tree._sim_env.make_move(
                    self._post_pos, 
                    self._act, 
                    self._parent._turn
                )
                
                self._lgl_mvs = self._parent._tree._sim_env.get_possible_actions(
                    self._post_pos,
                )
                self.load = True
            
            def add_leaf(self, new_node):
                '''adds new branching node, yeah this is confusing...'''
                self._leaf = new_node
            
            def has_leaf(self):
                '''checks if branching node below edge'''
                return self._leaf is not None
            
            def get_leaf(self):
                '''Gets leaf if exists, else makes one and returns'''
                if self._leaf is not None:
                    return self._leaf
                else:
                    print 'SIM!'
                    sim_pos = np.copy(self._pos)
                    self._parent._tree._sim_env.make_move(
                        sim_pos, self._act, self._parent._turn
                    )
                    self._leaf = self._parent._tree.TreeNode(
                        pos = sim_pos,
                        tree = self._parent._tree,
                        turn = 1-self._parent._turn
                    )
                    return self._leaf
        
        def __init__(self, pos, tree, turn):
            self._pos = pos
            self._tree = tree
            self._turn = turn
            self._sa_dict = {a: self.TreeEdge(pos, a, self)
                             for a in tree._sim_env.get_possible_actions(pos)}
            self._node_n = 0.
            # performance
            self._prev_best = 0
            self._granul = 10
        
        def _u(self, action, s_val):
            '''s_val: speedup val'''
            return (
                s_val/(1+self._sa_dict[action].N())
            )
        
        def max_select(self):
            '''this is for simulation!!'''
            #sqrt_val = np.sqrt(self._node_n) # speedup
            if self._node_n % self._granul == 0:
                s_val = self._tree._c_puct * np.sqrt(self._node_n)
                best = max([(self._sa_dict[a].Q()+self._u(a, s_val), a)
                            for a in self._sa_dict.keys()])
                self._prev_best = best[1]
            return self._prev_best
        
        def max_act(self):
            '''this is to extract the best action'''
            print [(a, self._sa_dict[a].N(), self._sa_dict[a].Q()) 
                   for a in self._sa_dict.keys()]
            best = max([(a, self._sa_dict[a].N())
                        for a in self._sa_dict.keys()], 
                       key = lambda x: x[1])
            print best + (self._sa_dict[best[0]].Q(),)
            return best[0]
            
        def sim(self, turn = None):
            '''runs the random simulation for MC estimation'''
            if turn is None:
                turn = self._turn #tree._sim_env.get_turn(self._pos)
            a = self.max_select()
            chosen_edge = self._sa_dict[a]
            if chosen_edge.has_leaf():
                res = -chosen_edge.get_leaf().sim(1-turn)
                # above must be - because what good to child is bad for me
            else:
                ## below two lines are needed for generic case
                #me = self._tree._rollout_policy()
                #op = self._tree._rollout_policy()
                if not self._sa_dict[a].load: # if lazy loading not already done
                    self._sa_dict[a].lazy_postpos()
                sim_pos = self._sa_dict[a]._post_pos
                lgl_mvs = self._sa_dict[a]._lgl_mvs
                if turn == 0:
                    '''I'm simulating first right now'''
                    self._tree._sim_env.set_start(sim_pos)
                    res = self._tree._sim_env.randomEp(1, lgl_mvs)
                elif turn == 1:
                    '''I'm simulating later right now'''
                    self._tree._sim_env.set_start(sim_pos)
                    res = -self._tree._sim_env.randomEp(0, lgl_mvs) # note the -
                
                # check if edge needs expansion (don't expand when end-pos)
                if (chosen_edge.N() > self._tree._n_thr and 
                    not self._tree._sim_env.game_finished(sim_pos)):
                    new_node = self._tree.make_node(sim_pos, self._tree, 1-turn)
                    chosen_edge.add_leaf(new_node)

            # now res is always defined, so let's update stuff
            assert res != 0
            self._node_n += 1.
            chosen_edge.update(res)
            return res
        
        def show(self, level):
            '''this is for visualization, for debugging'''
            translate = ["X", ".", "O"]
            print '    '*level,
            print ''.join(map(lambda x: translate[int(x)+1], self._pos.flatten())), 
            print self._node_n, self._turn
            for a in self._sa_dict.keys():
                #print a, self._sa_dict[a].Q(), self._sa_dict[a].N()
                if self._sa_dict[a].has_leaf():
                    self._sa_dict[a].get_leaf().show(level+1)
    
    def __init__(self, c_puct, n_thr, sim_env, rollout_policy):
        self._root = None
        self._c_puct = c_puct
        self._n_thr = n_thr
        # given env must have the following functions:
        assert hasattr(sim_env, 'game_finished'), \
               'No game_finished method @ env'
        assert hasattr(sim_env, 'get_possible_actions'), \
               'No get_possible_actions method @ env'
        assert hasattr(sim_env, 'make_move'), 'No make_move method @ env'
        assert hasattr(sim_env, 'get_turn'), 'No get_Turn method @ env'
        self._sim_env = sim_env
        self._rollout_policy = rollout_policy
        # ---------------------------------------------------------------------
        self.inners = [0, 0, 0, 0]
        self.nono = None
        self.added = 0
        self.ok = False
    
    def make_node(self, pos, tree, turn):
        '''stupid hack to make Nodes from within nodes'''
        return self.TreeNode(pos, tree, turn)
    
    def sim(self):
        l_i = 0
        for i in trange(250000):
            self._root.sim()
        #self.show()
    
    def show(self):
        self._root.show(0)
    
    def make_move(self, board):
        # root-moving algorithm
        if self._root == None:
            self._root = self.TreeNode(board, self, self._sim_env.get_turn(board))
        else:
            diff = self._root._pos[2].flatten() == board[2].flatten()
            new_a = list(diff).index(False)
            # TODO - below is temporary. beware. 
            prev_n = self._root._node_n
            print 'MCTS thinks your best move is %d' % self._root.max_act(),
            print '(your move was %d)' % new_a
            self._root = self._root._sa_dict[new_a].get_leaf()
            new_n = self._root._node_n
            print ('MCTS expected your move with %.2f %% confidence' % \
                   (float(100*new_n)/prev_n))
        self.ok = True
        while self.added < 10: continue
        raise Exception('DONE!')
        #self.sim()
        a = self._root.max_act()
        # below is temporary too. (you shouldn't directly access _sa_dict)
        try:
            self._root = self._root._sa_dict[a].get_leaf()
        except AssertionError:
            self._root = None # shouldn't play after this point
        return a
    
    def update(self, ns, reward):
        return
    
    def as_func(self, board):
        return self.make_move(board)
    
    def hoo(self, i):
        self.inners[i] += 1
        self.added += 1
        return self.inners
    
    def check_inners(self):
        print self.inners
    
    def ready(self):
        return self.added
    
    def okay(self):
        self.ok = True
    
    def okq(self):
        return self.ok

class DarkManager(BaseManager): pass

env = SimHexEnv('black', 'random', 'numpy3c', 'raise', 3)
dc = DummyClass01(0.03, 20, env, RandomHexPlayer)

def serv():
    global dc
    DarkManager.register('get_dc', callable=lambda:dc)
    m = DarkManager(address=('', 50000), authkey='abracadabra')
    s = m.get_server()
    s.serve_forever()

def cli(x):
    DarkManager.register('get_dc')
    m = DarkManager(address=('', 50000), authkey='abracadabra')
    m.connect()
    oo = m.get_dc()
    
    #while not oo.okq():
        #pass # yeah, yeah, busy waiting... 
    
    while oo.ready() < 1000:
        res = oo.hoo(x)
    print x
    if x == 0: print res

def look(x):
    DarkManager.register('get_dc')
    m = DarkManager(address=('', 50000), authkey='abracadabra')
    m.connect()
    oo = m.get_dc()
    oo.check_inners()

if __name__ == '__main__':
    me = HumanHexPlayer()
    serv = Process(target = serv)
    serv.start()
    print 'serv complete'
    clients = []
    for i in xrange(4):
        clien = Process(target = cli, args = (i,))
        clien.start()
        clients.append(clien)
    #me.runEp(opponent = dc.as_func)
    sleep(2)
    for c in clients:
        c.terminate()
    serv.terminate()
    