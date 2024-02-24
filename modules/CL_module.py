
from AutoCD.causal_discovery.select_with_OCT_parallel import *
from AutoCD.causal_discovery.class_causal_config import *
from AutoCD.data_utils.class_data_object import *

def CL_module(dataObj):

    '''
    Applies the CL module
    Authors: kbiza@csd.uoc.gr
    Parameters
    ----------
        dataObj (class object) : contains the preprocessed reduced data

    Returns
    -------
        opt_causal_config (dictionary) : the selected causal configuration
        opt_mec_graph_pd (pandas Dataframe) : the matrix of the estimated MEC causal graph
    '''


    # Create causal configurations
    causal_configs = CausalConfigurator().create_causal_configs(dataObj, False)

    # Run OCT tuning method
    library_results, opt_mec_graph_pd, opt_graph_pd, opt_causal_config = (
        OCT_parallel(dataObj, 2).select(causal_configs))


    return opt_causal_config, opt_mec_graph_pd