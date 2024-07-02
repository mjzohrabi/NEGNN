import os
import sys

from tqdm import tqdm
from utils.train_lp_de import main_de
from utils.train_lp import main
from utils.utils import set_seed, test_settings_similarity



def run_experiments(seed, dataset_name, model_name):
    set_seed(seed)

    gamma = explore = alpha = beta = gamma = 1
    results_file = f"results/link_prediction/dropedge/{dataset_name}_{model_name}_{seed}.txt"

    org_auc, org_precision, org_recall, org_f1 = main(seed=seed, dataset_name=dataset_name, extend_metric='None', model_name=model_name, alpha=alpha, beta=beta, gamma=gamma, explore=explore)
    with open(results_file, 'w') as f:
        f.write(f"Original {model_name} on {dataset_name} with seed: {seed}")
        f.write("\n")
        f.write(f"auc, precision, recall, f1, alpha, gamma, explore")
        f.write("\n")
        f.write("{:.3f},{:.3f},{:.3f},{:.3f},{},{}".format(org_auc, org_precision, org_recall, org_f1, alpha, gamma, explore))
        f.write("\n")
        f.write("Experiments started")
        f.write("\n--------------------------------------------\n")
        
    org_auc, org_precision, org_recall, org_f1 = main_de(seed=seed, dataset_name=dataset_name, extend_metric='None', model_name=model_name, alpha=alpha, beta=beta, gamma=gamma, explore=explore)
    with open(results_file, 'a') as f:
        f.write(f"DropEdge {model_name} on {dataset_name} with seed: {seed}")
        f.write("\n")
        f.write(f"auc, precision, recall, f1, alpha, gamma, explore")
        f.write("\n")
        f.write("{:.3f},{:.3f},{:.3f},{:.3f},{},{}".format(org_auc, org_precision, org_recall, org_f1, alpha, gamma, explore))
        f.write("\n")
        f.write("Experiments started")
        f.write("\n--------------------------------------------\n")
                
    




if __name__ == "__main__":
    # Centrality runs 
    os.makedirs('results/link_prediction/dropedge/', exist_ok=True)

    seeds = [1]
    # seeds = [1, 2, 3, 4, 5]
    dataset_names = ['cora', 'citeseer', 'pubmed', 'amazonphoto', 'amazoncomputer']
    # dataset_names = ['amazonphoto', 'amazoncomputer']
    model_names = ['gat', 'gatv2', 'gcn', 'sage']

    for seed in seeds:
        for dataset_name in tqdm(dataset_names):
            for model_name in model_names:
                run_experiments(seed, dataset_name, model_name)