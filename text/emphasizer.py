import os
import sys
import numpy as np
import re
from collections import defaultdict
import re

class Emphasizer:
    def __init__(self, path):
        if '.npy' in path:
            self.db = np.load(path, allow_pickle=True).item()
        else:
            self.db = {}
            self.duplicates = defaultdict(list)
            with open(path) as fp:
                for l in fp:
                    t,e = l.strip().split('|')
                    ex = self.db.get(t)
                    if '+' not in e.lower():
                        print(e)
                    if not ex:
                        self.db[t] = e
                    else:
                        if ex.strip() != e.strip():
                            if '+' in ex:
                                self.duplicates[t].append(ex)
                            if '+' in e:
                                self.duplicates[t].append(e)
            # print(self.duplicates,len(self.duplicates))

    def find(self, name, tries=4):
        l = len(name)
        for i in range(tries):
            r = self.db.get(name[:l-i])
            if r:
                return r + name[l-i:]
        return name

    def add_stress(self, text):
        words = text.split(' ')
        res = []
        for w in words:
            if "+" in w:
                res.append(w)
                continue
            w = self.find(w)
            res.append(w)
        return ' '.join(res)

if __name__ == '__main__':
    r = Emphasizer('ru_emphasize.dict')
    print(r.add_stress('привет мама'))

    # w = open('a1.txt','w')
    # err = open('err.txt','w')
    # add = open('add.txt','w')
    # with open('ru_emphasize.dict') as fp:
    #     for l in fp:
    #         t,e = l.strip().split('|')
    #         idx = e.index('+')
    #         e = e.replace('+','')
    #         e = e[:idx-1]+'+'+e[idx-1:]
    #         w.write('%s|%s\n' % (t,e))
