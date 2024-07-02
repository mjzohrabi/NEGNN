import os
from utils.train_nc import main
from utils.train_nc_de import main_de
from utils.utils import set_seed, set_device
from tqdm import tqdm

# by having verbose=True, we can see the timing results in the specified folder 

def run_original(model_name, dataset_name):
    set_seed(seed)
    _, _, _, _ = main(seed=seed, dataset_name=dataset_name, extend_metric='None', model_name=model_name, alpha=0, beta=0, gamma=0, explore=0, verbose=True)
    _, _, _, _ = main_de(seed=seed, dataset_name=dataset_name, extend_metric='None', model_name=model_name, alpha=0, beta=0, gamma=0, explore=0, verbose=True)




if __name__ == "__main__":

    seeds = [1]
    # seeds = [1, 2, 3, 4, 5]
    dataset_names = ['cora']
    model_names = ['gat', 'gatv2', 'gcn', 'sage']
    
    num_nodes_cora = 2708
    # num_nodes_citeseer = 3327
    # num_nodes_pubmed = 19717
    # num_nodes_amazonphoto = 7650
    # num_nodes_amazoncomputer = 13752
    
    
    # run original
    print("Running original")
    for seed in seeds:
        for dataset in dataset_names:
            for model in tqdm(model_names):
                run_original(model, dataset)
                
    
