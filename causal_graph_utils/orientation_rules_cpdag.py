
import numpy as np
import pandas as pd

#  Orientation rules
#  Author: kbiza@csd.uoc.gr, based on the matlab code by striant@csd.uoc.gr


def R0(cpdag, unshielded_triples, dag, verbose):

    n_nodes = cpdag.shape[1]
    dnc = {}

    for c in range(n_nodes):
        curtriples = unshielded_triples[c]
        ntriples = len(curtriples[0])

        if ntriples > 0:
            sep=np.zeros(ntriples, dtype=bool)

            for i in range(ntriples):
                triple = [curtriples[0][i], curtriples[1][i]]

                if np.logical_or(dag[triple[0],c]!=2 , dag[triple[1],c]!=2):
                    sep[i]=True

            dnc[c] = [curtriples[0][sep], curtriples[1][sep]]

            cpdag[curtriples[0][~sep],c] = 2
            cpdag[curtriples[1][~sep],c] = 2
            cpdag[c, curtriples[0][~sep]] = 3
            cpdag[c, curtriples[1][~sep]] = 3

            if verbose:
                idx=np.nonzero(~sep)[0]
                for i in idx:
                    print('R0: Orienting %d-->%d<--%d' %(curtriples[0][i],c,curtriples[1][i]))

        else:
            dnc[c]=[]

    return cpdag, dnc

def R1(cpdag, flag, verbose):

    # If a*->bo-*c and a,c not adjacent ==> a*->b->c
    [c, b] = np.where(cpdag == 1)
    len = c.size

    for i in range(len):
        if (cpdag[c[i], b[i]] == 1) and np.any(np.logical_and(cpdag[:, b[i]] == 2, cpdag[:, c[i]] == 0)) :
            if verbose:
                print('R1: Orienting %d-->%d' %(b[i],c[i]))

            cpdag[b[i], c[i]] = 2
            cpdag[c[i], b[i]] = 3
            flag = True

    return cpdag, flag

def R2(cpdag, flag, verbose):
    #If a->b*->c or a*->b->c, and a*-oc ==> a*->c

    [a, c] = np.where(cpdag == 1)
    len = a.size

    for i in range(len):
        r0 = cpdag[a[i], c[i]] == 1
        r1 = cpdag[a[i], :] == 2
        r2 = cpdag[:, c[i]] == 2
        r3 = cpdag[:, a[i]] == 3
        r4 = cpdag[c[i], :] == 3

        if r0 and np.any(np.logical_and(np.logical_and(r1,r2), np.logical_or(r3,r4))):
            if verbose:
                print('R2: Orienting %d-->%d' %(a[i],c[i]))
            cpdag[a[i], c[i]] = 2
            cpdag[c[i], a[i]] = 3
            flag = True

    return cpdag, flag

def R3(cpdag, flag, verbose):
    #If a*->b<-*c, a*-o8o-*c, a,c not adjacent, 8*-ob ==> 8*->b


    [th, b] = np.where(cpdag == 1)
    nedges = th.size

    for i in range(nedges):
        r1 = cpdag[:, th[i]] == 1
        r2 = cpdag[:, b[i]] == 2
        a = np.where(np.logical_and(r1, r2))[0]
        len_ = len(a)
        f = False
        for j in range(len_):
            for k in range(j+1, len_):
                r3 = cpdag[a[j], a[k]] == 0
                r4 = cpdag[th[i], b[i]] == 1
                if np.logical_and(r3, r4):
                    if verbose:
                        print('R3: Orienting %d-->%d' %(th[i], b[i]))
                cpdag[th[i], b[i]] = 2
                cpdag[b[i], th[i]] = 3
                flag = True
                f = True
                break
            if f:
                break

    return cpdag, flag
