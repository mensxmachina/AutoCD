import numpy as np
import networkx as nx
from AutoCD.causal_graph_utils.one_directed_path import *
from AutoCD.causal_graph_utils.find_ancestors_nx import *

#  Orientation rules
#  Author: kbiza@csd.uoc.gr, based on the matlab code by striant@csd.uoc.gr


def R0(pag, unshielded_triples, mag, verbose):

    n_nodes = pag.shape[1]
    dnc = {}

    for c in range(n_nodes):
        curtriples = unshielded_triples[c]
        ntriples = len(curtriples[0])

        if ntriples > 0:
            sep=np.zeros(ntriples, dtype=bool)

            for i in range(ntriples):
                triple = [curtriples[0][i], curtriples[1][i]]

                if np.logical_or(mag[triple[0],c]!=2 , mag[triple[1],c]!=2):
                    sep[i]=True

            dnc[c] = [curtriples[0][sep], curtriples[1][sep]]

            pag[curtriples[0][~sep],c] = 2
            pag[curtriples[1][~sep],c] = 2

            if verbose:
                idx=np.nonzero(~sep)[0]
                for i in idx:
                    print('R0: Orienting %d*->%d<-*%d' %(curtriples[0][i],c,curtriples[1][i]))

        else:
            dnc[c]=[]

    return pag, dnc

def R1(Pag, flag, verbose):

    # If a*->bo-*c and a,c not adjacent ==> a*->b->c
    [c, b] = np.where(Pag == 1)
    len = c.size

    for i in range(len):
        if (Pag[c[i], b[i]] == 1) and np.any(np.logical_and(Pag[:, b[i]] == 2, Pag[:, c[i]] == 0)) :
            if verbose:
                print('R1: Orienting %d->%d' %(b[i],c[i]))

            Pag[b[i], c[i]] = 2
            Pag[c[i], b[i]] = 3
            flag = True

    return Pag, flag


def R2_(Pag, flag, verbose):
    #If a->b*->c or a*->b->c, and a*-oc ==> a*->c

    [a, c] = np.where(Pag == 1)
    len = a.size

    for i in range(len):
        r0 = Pag[a[i], c[i]] == 1
        r1 = Pag[a[i], :] == 2
        r2 = Pag[:, c[i]] == 2
        r3 = Pag[:, a[i]] == 3
        r4 = Pag[c[i], :] == 3

        if r0 and np.any(np.logical_and(np.logical_and(r1,r2), np.logical_or(r3,r4))):
            if verbose:
                print('R2: Orienting %d*->%d' %(a[i],c[i]))
            Pag[a[i], c[i]] = 2
            flag = True

    return Pag, flag

def R3(Pag, flag, verbose):
    #If a*->b<-*c, a*-o8o-*c, a,c not adjacent, 8*-ob ==> 8*->b


    [th, b] = np.where(Pag == 1)
    nedges = th.size

    for i in range(nedges):
        r1 = Pag[:, th[i]] == 1
        r2 = Pag[:, b[i]] == 2
        a = np.where(np.logical_and(r1, r2))[0]
        len_ = len(a)
        f = False
        for j in range(len_):
            for k in range(j+1, len_):
                r3 = Pag[a[j], a[k]] == 0
                r4 = Pag[th[i], b[i]] == 1
                if np.logical_and(r3, r4):
                    if verbose:
                        print('R3: Orienting %d*->%d' %(th[i], b[i]))
                Pag[th[i], b[i]] = 2
                flag = True
                f = True
                break
            if f:
                break

    return Pag, flag

def R4(pag, mag, flag, verbose):

    '''
    Start from some node X, for node Y
    Visit all possible nodes X*->V & V->Y
    For every neighbour that is bi-directed and a parent of Y, continue
    For every neighbour that is bi-directed and o-*Y, orient and if parent continue
    Total: n*n*(n+m)

     For each node Y, find all orientable neighbours W
     For each node X, non-adjacent to Y, see if there is a path to some node in W
     Create graph as follows:
     for X,Y
     edges X*->V & V -> Y --> X -> V
     edges A <-> B & A -> Y --> A -> B
     edges A <-* W & A -> Y --> A->W
     discriminating: if path from X to W

    '''

    n_nodes = pag.shape[1]
    pag_t = np.matrix.transpose(pag)
    dir = np.logical_and(pag == 2, pag_t == 3)
    bidir = np.logical_and(pag == 2, pag_t == 2)

    for curc in range(n_nodes):
        b = np.where(pag[curc,:] == 1)[0]
        if len(b) == 0:
            continue

        th = np.where(pag[curc,:] == 0)[0]
        if len(th)==0:
            continue

        cur_dir = dir[:, curc]

        G = np.zeros((n_nodes, n_nodes), dtype=int)
        for curth in th:
            r1 = np.logical_and(pag[curth, :] == 2, cur_dir)
            idx = np.nonzero(r1)
            G[curth, idx] = 1

        ds = np.nonzero(cur_dir)[0]
        for d in ds:
            idx = np.nonzero(bidir[d,:])
            G[idx, d] = 1

        Gnx = nx.from_numpy_array(G, create_using=nx.MultiDiGraph())
        TC = nx.transitive_closure(Gnx)
        closure = np.zeros((n_nodes, n_nodes), dtype=int)
        for node in range(n_nodes):
            edges = list(TC.edges(node))
            for edge in edges:
                closure[edge[0], edge[1]] = 1

        a = np.nonzero(np.any(closure[th,:], axis=0))[0]

        if len(a)==0:
            continue

        for curb in b:
            for cura in a:
                if pag[curb, cura] == 2:
                    r1 = closure[th, cura]
                    r2 = mag[curb, curc] == 2
                    r3 = mag[curc, curb] == 3
                    if np.any(np.logical_and(np.logical_and(r1,r2),r3)):
                        if verbose:
                            print('R4: Orienting %d->%d' %(curb,curc))

                        pag[curb, curc] = 2
                        pag[curc, curb] = 3

                    else:
                        if verbose:
                            print('R4: Orienting %d<->%d' %(curb, curc))

                        pag[curb, curc] = 2
                        pag[curc, curb] = 2
                        pag[cura, curb] = 2

                    flag=True
                    break

    return pag, flag

def R8(G, flag, verbose):

    G_t = np.transpose(G)
    [r, c] = np.where(np.logical_and(G == 2, G_t == 1))

    n_edges = len(r)

    for i in range(n_edges):
        out = np.where(G[:, r[i]]==3)[0]
        if np.any(np.logical_and(G[out, c[i]] == 2, G[c[i], out] == 3)):
            if verbose:
                print('R8: Orienting %d->%d' %(r[i], c[i]))
            G[c[i], r[i]] = 3
            flag = True

    return G, flag

def R9_R10(G, dnc, flag, verbose):

    n_nodes = G.shape[1]
    G_t = np.transpose(G)
    [r, c] = np.where(np.logical_and(G == 2, G_t == 1))
    n_edges = len(r)

    # R9: Equivalent to orienting X <-o Y as X <-> Y and checking if Y is an
    # ancestor of X (i.e. there is an almost directed cycle)

    for i in range(n_edges):
        # Can it be bidirected? (R9)
        G_ = G.copy()
        G_[c[i], r[i]] = 2

        # isReachablePag : assume that we consider only '-->' edges
        path = one_directed_path(G_, r[i], c[i])
        if path:
            if verbose:
                print('R9: Orienting %d*--%d' %(c[i], r[i]))
            G[c[i], r[i]] = 3
            flag = True


    # R10: Equivalent to checking if for some definite non collider V - X - W
    # and edge X o-> Y, X->V and X->W both create a directed path to Y after
    # oriented
    r1 = np.logical_and(G == 1, G_t != 2)
    r2 = np.logical_and(G == 2, G_t == 3)
    r3 = np.logical_or(r1, r2)

    G_ones = r3 * 1
    Gnx = nx.from_numpy_array(G_ones, create_using=nx.MultiDiGraph())
    TC = nx.transitive_closure(Gnx)
    possible_closure = np.zeros((n_nodes, n_nodes), dtype=int)
    for node in range(n_nodes):
        edges = list(TC.edges(node))
        for edge in edges:
            possible_closure[edge[0], edge[1]] = 1

    closures = np.zeros(G.shape, dtype=int)
    tested = np.zeros(G.shape[1], dtype=bool)
    for s in range(G.shape[1]):
        tested[:] = False
        curdnc = dnc[s]
        if curdnc:
            ndnc = curdnc[0].size
        else:
            ndnc = 0

        tt = np.where(np.logical_and(G[:, s] == 1, G[s, :] == 2))[0]
        for t in tt:
            for j in range(ndnc):
                a = curdnc[0][j]
                b = curdnc[1][j]

                r1 = np.logical_or(possible_closure[a, t] == 0, possible_closure[b, t] == 0)
                r2 = np.logical_or(G[a, s] == 2, G[b, s] == 2)
                r3 = np.logical_or(a == t, b == t)
                if np.logical_or(np.logical_or(r1, r2), r3):
                    continue

                if tested[a] == False:
                    G_ = G.copy()
                    G_[s, a] = 2
                    G_[a, s] = 3
                    G_t_ = np.transpose(G_)
                    anc = find_ancestors_nx(G_t_, s)
                    closures[a, anc] = 1
                    tested[a] = True

                if closures[a, t] == 0:
                    continue

                if tested[b] == 0:
                    G_ = G.copy()
                    G_[s, b] = 2
                    G_[b, s] = 3
                    G_t_ = np.transpose(G_)
                    anc = find_ancestors_nx(G_t_, s)
                    closures[b, anc] = 1
                    tested[b] = True

                if closures[b, t] == 0:
                    continue

                if verbose:
                    print('R10: Orienting %d*--%d' % (t, s))

                G[t, s] = 3
                flag = True
                break

    return G, flag
