import os
import sys

from tqdm import tqdm
from utils.train_lp import main
from utils.utils import set_seed, test_settings_similarity



def run_experiments(seed, dataset_name, model_name, extend_metric):
    set_seed(seed)

    gamma = explore = alpha = beta = gamma = 1
    results_file = f"results/link_prediction/similarity/{dataset_name}_{model_name}_{seed}_{extend_metric}.txt"

    org_auc, org_precision, org_recall, org_f1 = main(seed=seed, dataset_name=dataset_name, extend_metric='None', model_name=model_name, alpha=alpha, beta=beta, gamma=gamma, explore=explore)
    with open(results_file, 'w') as f:
        f.write(f"Original {model_name} on {dataset_name} with seed: {seed}, extend metric: {extend_metric}")
        f.write("\n")
        f.write(f"auc, precision, recall, f1, alpha, gamma, explore")
        f.write("\n")
        f.write("{:.3f},{:.3f},{:.3f},{:.3f},{},{}".format(org_auc, org_precision, org_recall, org_f1, alpha, gamma, explore))
        f.write("\n")
        f.write("Experiments started")
        f.write("\n--------------------------------------------\n")

    test_list_alpha_similarity, test_list_gamma, test_list_explore = test_settings_similarity(dataset_name)
    beta = 0
    for explore in test_list_explore:
        for alpha in test_list_alpha_similarity:
            for gamma in test_list_gamma:
                try:
                    auc, precision, recall, f1 = main(seed=seed, dataset_name=dataset_name, model_name=model_name, alpha=alpha, beta=beta, gamma=gamma, explore=explore, extend_metric=extend_metric)

                    if (auc > org_auc) or (f1 > org_f1):
                        with open(results_file, 'a') as f:
                            f.write("{:.3f},{:.3f},{:.3f},{:.3f},{},{}".format(auc, precision, recall, f1,  alpha, gamma, explore))
                            f.write("\n")
                except:
                    print(f"Error in {dataset_name} {model_name} {extend_metric} {alpha} {gamma} {explore}")
                    continue


if __name__ == "__main__":
    # Centrality runs 
    os.makedirs('results/link_prediction/similarity/', exist_ok=True)

    seeds = [1]
    # seeds = [1, 2, 3, 4, 5]
    dataset_names = ['cora', 'citeseer', 'pubmed', 'amazonphoto', 'amazoncomputer']
    model_names = ['gat', 'gatv2', 'gcn', 'sage']
    extend_metrics = ["adamic_adar", "jaccard", "resource_allocation"]

    for seed in seeds:
        for dataset_name in tqdm(dataset_names):
            for model_name in model_names:
                for extend_metric in extend_metrics:
                    run_experiments(seed, dataset_name, model_name, extend_metric)