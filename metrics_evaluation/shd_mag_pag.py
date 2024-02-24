
def shd_mag_pag(G1, G2):

    '''
    Computes the structural hamming distance as appeared in
    S. Triantafillou and I. Tsamardinos,  UAI 2016
    Author : kbiza@csd.uoc.gr, based on matlab code by striant@csd.uoc.gr

    Args:
        G1(numpy array): a matrix of a graph (mag or pag)
        G2(numpy array): a matrix of a graph (mag or pag, must be the same type with G1)

    Returns:
        shd(int): the value of the metric
    '''

    n_nodes = G1.shape[0]
    shd = 0
    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            # o-o
            if G1[i,j] == 1 and G1[j,i] == 1:
                # o-o
                if G2[i,j] == 1 and G2[j,i] == 1:
                    shd = shd + 0
                # o->
                if G2[i,j] == 2 and G2[j,i] == 1:
                    shd = shd + 1
                # <-o
                if G2[i,j] == 1 and G2[j,i] == 2:
                    shd = shd + 1
                # <->
                if G2[i,j] == 2 and G2[j,i] == 2:
                    shd = shd + 2
                # -->
                if G2[i,j] == 2 and G2[j,i] == 3:
                    shd = shd + 2
                # <--
                if G2[i,j] == 3 and G2[j,i] == 2:
                    shd = shd + 2
                # 'empty'
                if G2[i,j] == 0 and G2[j,i] == 0:
                    shd = shd + 1

            # o->
            if G1[i,j] == 2 and G1[j,i] == 1:
                # o-o
                if G2[i,j] == 1 and G2[j,i] == 1:
                    shd = shd + 1
                # o->
                if G2[i,j] == 2 and G2[j,i] == 1:
                    shd = shd + 0
                # <-o
                if G2[i,j] == 1 and G2[j,i] == 2:
                    shd = shd + 2
                # <->
                if G2[i,j] == 2 and G2[j,i] == 2:
                    shd = shd + 1
                # -->
                if G2[i,j] == 2 and G2[j,i] == 3:
                    shd = shd + 1
                # <--
                if G2[i,j] == 3 and G2[j,i] == 2:
                    shd = shd + 2
                # 'empty'
                if G2[i,j] == 0 and G2[j,i] == 0:
                    shd = shd + 2

            # <-o
            if G1[i,j] == 1 and G1[j,i] == 2:
                # o-o
                if G2[i,j] == 1 and G2[j,i] == 1:
                    shd = shd + 1
                # o->
                if G2[i,j] == 2 and G2[j,i] == 1:
                    shd = shd + 2
                # <-o
                if G2[i,j] == 1 and G2[j,i] == 2:
                    shd = shd + 0
                # <->
                if G2[i,j] == 2 and G2[j,i] == 2:
                    shd = shd + 1
                # -->
                if G2[i,j] == 2 and G2[j,i] == 3:
                    shd = shd + 2
                # <--
                if G2[i,j] == 3 and G2[j,i] == 2:
                    shd = shd + 1
                # 'empty'
                if G2[i,j] == 0 and G2[j,i] == 0:
                    shd = shd + 2

            # <->
            if G1[i,j] == 2 and G1[j,i] == 2:
                # o-o
                if G2[i,j] == 1 and G2[j,i] == 1:
                    shd = shd + 2
                # o->
                if G2[i,j] == 2 and G2[j,i] == 1:
                    shd = shd + 1
                # <-o
                if G2[i,j] == 1 and G2[j,i] == 2:
                    shd = shd + 1
                # <->
                if G2[i,j] == 2 and G2[j,i] == 2:
                    shd = shd + 0
                # -->
                if G2[i,j] == 2 and G2[j,i] == 3:
                    shd = shd + 1
                # <--
                if G2[i,j] == 3 and G2[j,i] == 2:
                    shd = shd + 1
                # 'empty'
                if G2[i,j] == 0 and G2[j,i] == 0:
                    shd = shd + 3


            # -->
            if G1[i,j] == 2 and G1[j,i] == 3:
                # o-o
                if G2[i,j] == 1 and G2[j,i] == 1:
                    shd = shd + 2
                # o->
                if G2[i,j] == 2 and G2[j,i] == 1:
                    shd = shd + 1
                # <-o
                if G2[i,j] == 1 and G2[j,i] == 2:
                    shd = shd + 2
                # <->
                if G2[i,j] == 2 and G2[j,i] == 2:
                    shd = shd + 1
                # -->
                if G2[i,j] == 2 and G2[j,i] == 3:
                    shd = shd + 0
                # <--
                if G2[i,j] == 3 and G2[j,i] == 2:
                    shd = shd + 2
                # 'empty'
                if G2[i,j] == 0 and G2[j,i] == 0:
                    shd = shd + 3

            # <--
            if G1[i,j] == 3 and G1[j,i] == 2:
                # o-o
                if G2[i,j] == 1 and G2[j,i] == 1:
                    shd = shd + 2
                # o->
                if G2[i,j] == 2 and G2[j,i] == 1:
                    shd = shd + 2
                # <-o
                if G2[i,j] == 1 and G2[j,i] == 2:
                    shd = shd + 1
                # <->
                if G2[i,j] == 2 and G2[j,i] == 2:
                    shd = shd + 1
                # -->
                if G2[i,j] == 2 and G2[j,i] == 3:
                    shd = shd + 2
                # <--
                if G2[i,j] == 3 and G2[j,i] == 2:
                    shd = shd + 0
                # 'empty'
                if G2[i,j] == 0 and G2[j,i] == 0:
                    shd = shd + 3


            # 'empty'
            if G1[i,j] == 0 and G1[j,i] == 0:
                # o-o
                if G2[i,j] == 1 and G2[j,i] == 1:
                    shd = shd + 1
                # o->
                if G2[i,j] == 2 and G2[j,i] == 1:
                    shd = shd + 2
                # <-o
                if G2[i,j] == 1 and G2[j,i] == 2:
                    shd = shd + 2
                # <->
                if G2[i,j] == 2 and G2[j,i] == 2:
                    shd = shd + 3
                # -->
                if G2[i,j] == 2 and G2[j,i] == 3:
                    shd = shd + 3
                # <--
                if G2[i,j] == 3 and G2[j,i] == 2:
                    shd = shd + 3
                # 'empty'
                if G2[i,j] == 0 and G2[j,i] == 0:
                    shd = shd + 0


    return shd