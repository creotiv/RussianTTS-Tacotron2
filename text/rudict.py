import os
import sys
import numpy as np
import re
from collections import defaultdict

class RuDict:
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
        if "+" in text:
            return text
        words = text.split(' ')
        res = []
        for w in words:
            w = self.find(w)
            res.append(w)
        return ' '.join(res)


def preprocess(path, save_path):
    res = {}
    with open(path) as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue

            name, apostrof, _ = line.split('|')
            res[name.strip()] = apostrof.strip()
        np.save(save_path, res)

if __name__ == '__main__':
    # preprocess(sys.argv[1], sys.argv[2])
    r = RuDict('ru_emphasize.dict')
    print(r.add_stress('привет мама'))