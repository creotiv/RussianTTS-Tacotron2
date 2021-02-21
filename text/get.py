w = open('dict2.txt','w')
with open('slov2.txt') as fp:
    for l in fp:
        t,e  = l.split("|")
        e = e.strip().replace("'","+").replace('/','').split(' ')[0]
        t = t.strip()
        w.write('%s|%s\n' % (t,e))