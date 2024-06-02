import os
from utils.train_nc import main
from utils.utils import set_seed, set_device
from tqdm import tqdm

# by having verbose=True, we can see the timing results in the specified folder 

def run_original(model_name, dataset_name):
    set_seed(seed)
    _, _, _, _ = main(seed=seed, dataset_name=dataset_name, extend_metric='None', model_name=model_name, alpha=0, beta=0, gamma=0, explore=0, verbose=True)


def run_extended(seed, dataset_name, model_name, extend_metric, alpha, beta, gamma, explore):
    set_seed(seed)
    _, _, _, _ = main(seed=seed, dataset_name=dataset_name, model_name=model_name, alpha=alpha, beta=beta, gamma=gamma, explore=explore, extend_metric=extend_metric, verbose=True)


if __name__ == "__main__":

    seeds = [1]
    # seeds = [1, 2, 3, 4, 5]
    dataset_names = ['cora']
    model_names = ['gat', 'gatv2', 'gcn', 'sage']
    extend_metrics_centrality = ["degree", "eigenvector"]
    extend_metrics_similarity = ["adamic_adar", "jaccard", "resource_allocation"]
    
    num_nodes_cora = 2708
    # num_nodes_citeseer = 3327
    # num_nodes_pubmed = 19717
    # num_nodes_amazonphoto = 7650
    # num_nodes_amazoncomputer = 13752
    
    alpha_percents = [0.2, 0.4, 0.6, 0.8, 1]

    
    # run original
    print("Running original")
    for seed in seeds:
        for dataset in dataset_names:
            for model in tqdm(model_names):
                run_original(model, dataset)
                
    
    alpha_list = [int(num*num_nodes_cora) for num in alpha_percents]
    
    # run centrality
    print("Running centrality")
    beta_list = [1, 5, 10]    
    gamma = 0
    explore = 0
    for seed in seeds:
        for dataset in dataset_names:
            for model in tqdm(model_names):
                for extend_metric in extend_metrics_centrality:
                    for alpha in alpha_list:
                        for beta in beta_list:
                            run_extended(seed, dataset, model, extend_metric, alpha, beta, gamma=gamma, explore=explore)

    
    # run similarity
    print("Running similarity")
    gamma_list = [int(i) for i in range(2000, 11000, 2000)]
    explore = 0.01
    beta = 0
    for seed in seeds:
        for dataset in dataset_names:
            for model in tqdm(model_names):
                for alpha in alpha_list:
                    for extend_metric in extend_metrics_similarity:
                        for gamma in gamma_list:
                            run_extended(seed, dataset, model, extend_metric, alpha, beta=beta, gamma=gamma, explore=explore)
                        
