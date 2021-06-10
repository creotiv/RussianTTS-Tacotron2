import numpy as np
from collections import defaultdict

class Emphasizer:
    def __init__(self, path):
        if '.npy' in path:
            self.db = np.load(path, allow_pickle=True).item()
        else:
            self.db = {}
            self.parts = {}
            self.duplicates = defaultdict(list)
            with open(path) as fp:
                for l in fp:
                    t,e = l.strip().split('|')
                    ex = self.db.get(t)
                    if '+' not in e.lower():
                        raise Exception("No stress found in:%s" % t)
                    idx = e.rfind('+')
                    self.parts[t[0:idx+1]] = (e[0:idx+2], len(e))
                    if not ex:
                        self.db[t] = e
                    else:
                        if ex.strip() != e.strip():
                            if '+' in ex:
                                self.duplicates[t].append(ex)
                            if '+' in e:
                                self.duplicates[t].append(e)
            # print(self.duplicates,len(self.duplicates))

    # TODO: handle different stress position for different root of the word
    def find(self, name, tries=50):
        l = len(name)
        _name = name.replace('э','е').replace('ё','е')
        out = []
        for i in range(tries):
            x = self.parts.get(_name[:l-i])
            if x:
                r, loss = x
                loss -= len(name)
                out.append((name[0:r.rfind('+')] + '+' +name[r.rfind('+'):], abs(loss)))
        if len(out) > 1:
            print(out)
        if out:
            return out[0][0]
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
    import time
    r = Emphasizer('ru_emphasize.dict')
    st = time.perf_counter()
    print(r.add_stress('В Минздраве отметили, что вакцинация второй дозой тех, кого первой дозой прививали мобильные бригады, проводится по такому же алгоритму, что и вакцинация первой дозой. Списки, сформированные во время первой прививки, используются для получения следующей дозы.'.lower()))
    print(time.perf_counter() - st)