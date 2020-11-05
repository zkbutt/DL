import numpy as np

'''
基本求导
    c'=0(c为常数）
    (x^a)'=ax^(a-1),a为常数且a≠0
    (a^x)'=a^xlna
    (e^x)'=e^x
    (logax)'=1/(xlna),a>0且 a≠1
    (lnx)'=1/x
    (sinx)'=cosx
    (cosx)'=-sinx
    (tanx)'=(secx)^2
    (secx)'=secxtanx
    (cotx)'=-(cscx)^2
    (cscx)'=-csxcotx
    (arcsinx)'=1/√(1-x^2)
    (arccosx)'=-1/√(1-x^2)
    (arctanx)'=1/(1+x^2)
    (arccotx)'=-1/(1+x^2)
    (shx)'=chx
    (chx)'=shx
    （uv)'=uv'+u'v
    (u+v)'=u'+v'
    (u/)'=(u'v-uv')/^2
'''


def ecludDist(x, y):
    '''欧式距离'''
    return np.sqrt(sum(np.square(np.array(x) - np.array(y))))


def manhattanDist(x, y):
    '''曼哈顿距离'''
    return np.sum(np.abs(x - y))


def cos(x, y):
    '''夹角余弦'''
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
