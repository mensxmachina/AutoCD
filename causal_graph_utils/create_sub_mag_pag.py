import pandas as pd
from AutoCD.causal_graph_utils.dag_to_mag_removeL import *
from AutoCD.causal_graph_utils.mag_to_pag import *
from AutoCD.causal_graph_utils.enforce_stationarity import *
from AutoCD.data_utils.functions_variable_names import *

def create_sub_mag_pag(dag_pd, selected_vars, n_lags=None):

    '''
    Given a DAG and a set of latent variables,
    we marginalize out the latent variables and
    create the corresponding MAG and PAG.
    For time-lagged causal DAGs, we enforce the stationarity assumption.

    Parameters
    ----------
        dag_pd (pandas Dataframe) : the matrix of the DAG
        selected_vars (list) : name of nodes to be latent
        n_lags (None or int) : if int is the maximum number of previous time lags and the dag_pd must be a time-lagged graph

    Returns
    -------
        mag_noL_pd (pandas Dataframe) : the matrix of the MAG (the latent variables are removed)
        pag_noL_pd (pandas Dataframe) : the matrix of the PAG (the latent variables are removed)
    '''

    if isinstance(n_lags, int):

        is_latent_pd = pd.DataFrame(np.ones((1,dag_pd.shape[1]), dtype=bool), columns=dag_pd.columns)
        sel_vars_lagged = lagnames_from_names(selected_vars, n_lags)
        is_latent_pd[sel_vars_lagged] = False
        is_latent_np = is_latent_pd.to_numpy()
        _, mag_noL_pd_ = dag_to_mag_removeL(dag_pd, is_latent_np.reshape(-1))

        # enforce stationarity on MAG
        mag_noL_st = enforce_stationarity_add_edge(mag_noL_pd_.copy().to_numpy(), mag_noL_pd_, n_lags, False)
        mag_noL_pd = pd.DataFrame(mag_noL_st, columns=mag_noL_pd_.columns, index=mag_noL_pd_.index)

        # convert MAG to PAG
        pag_noL_pd = mag_to_pag(mag_noL_pd, False, n_lags)

    else:
        is_latent_pd = pd.DataFrame(np.ones((1, dag_pd.shape[1]), dtype=bool), columns=dag_pd.columns)
        is_latent_pd[selected_vars] = False
        is_latent_np = is_latent_pd.to_numpy()
        _, mag_noL_pd = dag_to_mag_removeL(dag_pd, is_latent_np.reshape(-1))

        # convert MAG to PAG
        pag_noL_pd = mag_to_pag(mag_noL_pd, False)

    return mag_noL_pd, pag_noL_pd
