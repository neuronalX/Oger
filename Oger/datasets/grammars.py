from numpy import random

if __name__ == '__main__':
    print 'This module is meant to be imported by grammar_task.py'

class simple_pcfg:
    """A very simple grammar that allows for variable vocabulary sizes.
    """
    
    def __init__(self, nouns=None, verbs=None, coding='string'):
        self.nouns = nouns
        self.verbs = verbs
        self.type = type
        if coding == 'string':
            if self.nouns == None:
                self.nouns = ['hamsters', 'boys', 'girls', 'pigs', 'computers', 'clowns', 'engineers']
            if self.verbs == None:
                self.verbs = ['follow', 'simulate', 'eat', 'collect', 'love', 'help']
            self.THAT = 'that'
            self.END = '.'
            self.WITH = 'with'
            self.FROM = 'from'
            self.THE = 'the'
        else:
            self.THAT = 0
            self.END = 1
            self.WITH = 2
            self.FROM = 3
            self.nouns = range(4, 4 + len(nouns))
            self.verbs = range(4 + len(nouns), 4 + len(nouns) + len(verbs))
    
    def S(self):
        return self.NP() + self.V() + self.NP() + [self.END]
    
    def SRC(self):
        NP = self.NP
        V = self.V
        return [self.THAT] + V() + NP()
    
    def ORC(self):
        N = self.N
        V = self.V
        return [self.THAT] + N() + V()
    
    def NP(self):
        f = 4 * [[]]
        f[0] = lambda: self.N()
        f[1] = lambda: self.N() + self.SRC()
        f[2] = lambda: self.N() + self.ORC()
        f[3] = lambda: self.N() + self.PP()
        return f[random.random_integers(0, 3)]()
    
    def PP(self):
        if random.sample() > .5:
            return [self.FROM] + self.N()
        else:
            return [self.WITH] + self.N()
    
    def N(self):
        if random.sample() > .5:
            return [self.nouns[random.random_integers(0, len(self.nouns) - 1)]]
        else:
            return [self.THE] + [self.nouns[random.random_integers(0, len(self.nouns) - 1)]]
    
    def V(self):
        return [self.verbs[random.random_integers(0, len(self.verbs) - 1)]]



