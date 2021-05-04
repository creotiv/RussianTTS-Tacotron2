import sys
import re

def fix(path, split='|', max_length=180):
    out = open('out.csv','w')
    with open(path, encoding='utf-8') as f:
        for line in f:
            el = line.strip().split(split)
            if len(el[1]) > max_length:
                continue
            if len(re.split('[\.\!\?]+',el[1])) > 2:
                continue
            out.write(line)
    
fix(sys.argv[1])