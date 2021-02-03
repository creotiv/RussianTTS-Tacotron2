import os
import sys
import numpy as np
import re

class RuDict:
    def __init__(self, path):
        self.db = np.load(path, allow_pickle=True).item()

    def find(self, name, tries=4):
        l = len(name)
        for i in range(tries):
            r = self.db.get(name[:l-i])
            if r:
                return r + name[l-i:]
        return name

    def text_to_text(self, text):
        words = re.split('[\s]+',text)
        res = []
        for w in words:
            cl = w.strip('!\'(),.:;?')
            cl2 = self.find(cl)
            w = w.replace(cl,cl2)
            res.append(w)
        res = ' '.join(res)
        return res.replace("'", "*")


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
    preprocess(sys.argv[1], sys.argv[2])