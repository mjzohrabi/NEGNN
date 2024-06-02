import numpy as np
import torch
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def set_device():
    device_list = [f'cuda:{i}' for i in range(torch.cuda.device_count())]
    return device_list[0]


def test_settings_centrality(dataset_name):
    match dataset_name:
        case 'cora':
            num_nodes = 2708
            test_list_alpha_centrality = []
            test_list_beta = []
            percents = [0.5, 0.6, 0.7, 0.8, 0.9, 1]
            percents = [int(num*num_nodes) for num in percents]
            # print(percents)
            for i in range(1, num_nodes):
                if i < 100:
                    if i % 20 == 0:
                        test_list_alpha_centrality.append(i)
                if 100 <= i < 1000:
                    if i % 80 == 0:
                        test_list_alpha_centrality.append(i)
                if i < 6:
                    test_list_beta.append(i)
                if 10 <= i < 100:
                    if i % 20 == 0: 
                        test_list_beta.append(i)
                if i in percents:
                    test_list_alpha_centrality.append(i)
            return test_list_alpha_centrality, test_list_beta
        
        case 'citeseer':
            num_nodes = 3327
            test_list_alpha_centrality = []
            test_list_beta = []
            percents = [0.5, 0.6, 0.7, 0.8, 0.9, 1]
            percents = [int(num*num_nodes) for num in percents]
            # print(percents)
            for i in range(1, num_nodes):
                if i < 100:
                    if i % 20 == 0:
                        test_list_alpha_centrality.append(i)
                if 100 <= i < 1200:
                    if i % 100 == 0:
                        test_list_alpha_centrality.append(i)
                if i < 5:
                    test_list_beta.append(i)
                if 10 <= i < 100:
                    if i % 20 == 0: 
                        test_list_beta.append(i)
                if i in percents:
                    test_list_alpha_centrality.append(i)
            return test_list_alpha_centrality, test_list_beta 
        
        case 'pubmed':
            num_nodes = 19717
            test_list_alpha_centrality = []
            test_list_beta = []
            percents = [0.5, 0.6, 0.7, 0.8]
            percents = [int(num*num_nodes) for num in percents]
            # print(percents)
            for i in range(1, num_nodes):
                if i < 300:
                    if i % 50 == 0:
                        test_list_alpha_centrality.append(i)
                if 300 < i < 10000:
                    if i % 700 == 0:
                        test_list_alpha_centrality.append(i)
                if i < 5:
                    test_list_beta.append(i)
                if i in percents:
                    test_list_alpha_centrality.append(i)       
            return test_list_alpha_centrality, test_list_beta  
        
        case 'amazonphoto':
            num_nodes = 7650
            test_list_alpha_centrality = []
            test_list_beta = []
            percents = [0.5, 0.6, 0.7, 0.8, 0.9, 1]
            percents = [int(num*num_nodes) for num in percents]
            # print(percents)
            for i in range(1, num_nodes):
                if i < 300:
                    if i % 50 == 0:
                        test_list_alpha_centrality.append(i)
                if 500 < i < 10000:
                    if i % 800 == 0:
                            test_list_alpha_centrality.append(i)
                if i < 5:
                    test_list_beta.append(i)
                if i in percents:
                    test_list_alpha_centrality.append(i)
            return test_list_alpha_centrality, test_list_beta
        
        case 'amazoncomputer':
            num_nodes = 13752
            test_list_alpha_centrality = []
            test_list_beta = []
            percents = [0.5, 0.6, 0.7, 0.8, 0.9]
            percents = [int(num*num_nodes) for num in percents]
            # print(percents)
            for i in range(1, num_nodes):
                if i < 500:
                    if i % 50 == 0:
                        test_list_alpha_centrality.append(i)
                if 500 < i < 8000:
                    if i % 1000 == 0:
                        test_list_alpha_centrality.append(i)
                if i < 5:
                    test_list_beta.append(i)
                if i in percents:
                    test_list_alpha_centrality.append(i)
            return test_list_alpha_centrality, test_list_beta
        
        case _:
            raise ValueError("Dataset is not found!")


def test_settings_similarity(name):
    match name:
        case 'cora':
            num_nodes = 2708
            test_list_gamma = []
            test_list_explore = [0.1]
            percents = [0.4, 0.6, 0.8, 1]
            test_list_alpha_similarity = [int(num*num_nodes) for num in percents]
            for i in range(num_nodes):
                if i > 300: 
                    if i % 200 == 0:
                        test_list_gamma.append(i)
            return test_list_alpha_similarity, test_list_gamma, test_list_explore

        case 'citeseer':
            num_nodes = 3327
            test_list_gamma = []
            test_list_explore = [0.1]
            percents = [0.4, 0.6, 0.8, 1]
            test_list_alpha_similarity = [int(num*num_nodes) for num in percents]
            for i in range(num_nodes):
                if i > 300: 
                    if i % 200 == 0:
                        test_list_gamma.append(i)
            return test_list_alpha_similarity, test_list_gamma, test_list_explore

        case 'pubmed':
            num_nodes = 19717
            test_list_gamma = []
            test_list_explore = [0.1]
            percents = [0.2]
            test_list_alpha_similarity = [int(num*num_nodes) for num in percents]
            for i in range(num_nodes):
                if 799 < i < 10000: 
                    if i % 800 == 0:
                        test_list_gamma.append(i)
            return test_list_alpha_similarity, test_list_gamma, test_list_explore

        case 'amazonphoto':
            num_nodes = 7650
            test_list_gamma = []
            test_list_explore = [0.004, 0.01]
            percents = [0.6]
            test_list_alpha_similarity = [int(num*num_nodes) for num in percents]
            for i in range(1, num_nodes):
                if i % 800 == 0:
                    test_list_gamma.append(i)
            return test_list_alpha_similarity, test_list_gamma, test_list_explore

        case 'amazoncomputer':
            num_nodes = 13752
            test_list_gamma = []
            test_list_explore = [0.01]
            percents = [0.3]
            test_list_alpha_similarity = [int(num*num_nodes) for num in percents]
            for i in range(1, num_nodes):
                if i % 800 == 0:
                    test_list_gamma.append(i)
            return test_list_alpha_similarity, test_list_gamma, test_list_explore

        case _:
            raise ValueError("Dataset is not defined!")