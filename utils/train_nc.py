import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
    
from utils.dataset import load_dataset
from extension.extend_nc import ExtendNeighborhood
from models.gat import GAT
from models.gatv2 import GATv2
from models.gcn import GCN
from models.sage import GraphSAGE
from utils.utils import set_device, set_seed
import os
import time


class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def step(self, acc, model):
        score = acc
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score:
            self.counter += 1
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model):
        '''Saves model when validation loss decrease.'''
        torch.save(model.state_dict(), 'es_checkpoint.pt')
        



def evaluate(g, model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
        logits = logits[mask]
        labels = labels[mask]
        predictions = torch.argmax(logits, dim=1)
        labels_np = labels.cpu().numpy()
        predictions_np = predictions.cpu().numpy()

        accuracy = accuracy_score(labels_np, predictions_np)
        precision = precision_score(labels_np, predictions_np, average='macro')
        recall = recall_score(labels_np, predictions_np, average='macro')
        f1 = f1_score(labels_np, predictions_np, average='macro')

        return accuracy, precision, recall, f1


    
def main(seed, dataset_name, model_name, alpha, beta, gamma, explore, extend_metric, verbose=False):

    add_self_loop=True
    fast_mode=False
    early_stop=False
    epochs=200
    lr=0.005
    weight_decay=5e-4

    set_seed(seed)
    device = set_device()
    
    
    g, graph_nx, graph_undirected, features, labels, train_mask, val_mask, test_mask, num_feats, n_classes, n_edges = load_dataset(dataset_name, verbose=True)

    if model_name == 'sage':
        # print("SAGE model is used")
        model = GraphSAGE(g.ndata['feat'].shape[1], 16)
    elif model_name == 'gcn':
        # print("GCN model is used")
        model = GCN(g.ndata['feat'].shape[1], 16)
    elif model_name == 'gat':
        # print("GAT model is used")
        num_heads, num_layers, num_out_heads = 8, 2, 1
        num_hidden = n_classes = 16
        in_drop, attn_drop, negative_slope, residual = 0.6, 0.6, 0.2, False
        heads = ([num_heads] * num_layers) + [num_out_heads]
        model = GAT(num_layers, g.ndata['feat'].shape[1], num_hidden, n_classes, heads, F.elu, in_drop, attn_drop, negative_slope, residual)
    elif model_name == 'gatv2':
        # print("GATv2 model is used")
        num_heads, num_layers, num_out_heads = 8, 2, 1
        num_hidden = n_classes = 16
        in_drop, attn_drop, negative_slope, residual = 0.6, 0.6, 0.2, False
        heads = ([num_heads] * num_layers) + [num_out_heads]
        model = GATv2(num_layers, g.ndata['feat'].shape[1], num_hidden, n_classes, heads, F.elu, in_drop, attn_drop, negative_slope, residual)
    else:
        raise("Not a model!")


    if extend_metric != 'None':
        t1 = time.time()
        num_edge_before_extention = g.number_of_edges()
        # print(f'Total number of edges before extension: {num_edge_before_extention}')
        extend_neighborhood = ExtendNeighborhood(g, graph_undirected, alpha=alpha, beta=beta, gamma=gamma, explore=explore, extend_metric=extend_metric)
        # print(extend_metric)
        if extend_metric == 'degree' or extend_metric == 'eigenvector':
            extended_graph = extend_neighborhood.add_edges_centrality_based()
        else: 
            extended_graph = extend_neighborhood.add_edges_similarity_based()
        g = extended_graph
        t_total = time.time() - t1
        num_edge_after_extention = g.number_of_edges()

        if verbose:
            os.makedirs('results/timing', exist_ok=True)
            with open('results/timing/extension_timing.txt', 'a+') as f:
                f.write(f"seed: {seed}\t")
                f.write(f"dataset_name: {dataset_name}\t")
                f.write(f"model_name: {model_name}\t")
                f.write(f"alpha: {alpha}\t")
                f.write(f"beta: {beta}\t")
                f.write(f"gamma: {gamma}\t")
                f.write(f"explore: {explore}\t")
                f.write(f"extend_metric: {extend_metric}\t")
                f.write(f"add_self_loop: {add_self_loop}\t")
                f.write(f"fast_mode: {fast_mode}\t")
                f.write(f"early_stop: {early_stop}\t")
                f.write(f"epochs: {epochs}\t")
                f.write(f"lr: {lr}\t")
                f.write(f"weight_decay: {weight_decay}\n")
                f.write(f'Total number of edges after extension: {num_edge_after_extention}\n')
                f.write(f'Total number of added edges: {num_edge_after_extention - num_edge_before_extention}\n')
                f.write(f"*** Total construction time in seconds: {t_total:.2f} ***\n")
                f.write(f"------------------------------------------------------------\n")

    if add_self_loop == True:
        # print(f"Total edges before adding self-loop {g.number_of_edges()}")
        g = g.remove_self_loop().add_self_loop()
        # print(f"Total edges after adding self-loop {g.number_of_edges()}")

    model = model.to(device)
    g, labels, features, train_mask, val_mask, test_mask = map(lambda x: x.to(device), (g, labels, features, train_mask, val_mask, test_mask))
    
    loss_fcn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay)

    # if early_stop:
    #     stopper = EarlyStopping(patience=100)

    train_loss = []
    val_loss = []
    start_time = time.time()
    for epoch in range(epochs):
        model.train()

        # forward
        logits = model(g, features)
        loss = loss_fcn(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        if epoch % 10 == 0:
            train_loss.append(loss.item())
            model.eval()
            with torch.no_grad():
                val_logits = model(g, features)
                val_loss.append(loss_fcn(val_logits[val_mask], labels[val_mask]).item())

    end_time = time.time()
    training_time = end_time - start_time

    # Save training time
    if verbose:
        os.makedirs('results/timing', exist_ok=True)
        with open('results/timing/training_time.txt', 'a+') as f:
            f.write(f"seed: {seed}\t")
            f.write(f"dataset_name: {dataset_name}\t")
            f.write(f"model_name: {model_name}\t")
            f.write(f"alpha: {alpha}\t")
            f.write(f"beta: {beta}\t")
            f.write(f"gamma: {gamma}\t")
            f.write(f"explore: {explore}\t")
            f.write(f"extend_metric: {extend_metric}\n")
            f.write(f"Training Time: {training_time} seconds\n")
            f.write(f"------------------------------------------------------------\n")

        os.makedirs('results/loss_plots', exist_ok=True)
        # Number of epochs
        interval = 10  # Interval at which you saved the loss values
        # Generate the x-values based on the interval
        plot_epochs = list(range(0, len(train_loss) * interval, interval))
        # Plotting the raw losses
        plt.plot(plot_epochs, train_loss, label='Training Loss')
        plt.plot(plot_epochs, val_loss, label='Validation Loss')

        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f'results/loss_plots/loss_plot_{seed}_{dataset_name}_{model_name}_{extend_metric}_{alpha}_{beta}_{gamma}_{explore}.png')
        plt.close()

        # train_acc = accuracy(logits[train_mask], labels[train_mask])

        # if fast_mode:
        #     val_acc = accuracy(logits[val_mask], labels[val_mask])
        # else:
        #     val_acc = evaluate(model, features, labels, val_mask)
            # if early_stop:
            #     if stopper.step(val_acc, model):
            #         break
        
        # if epoch % 10 == 0:
        #     print("Epoch {:05d} | Loss {:.4f} | TrainAcc {:.4f} |"
        #         " ValAcc {:.4f}".
        #         format(epoch, loss.item(), train_acc,
        #                 val_acc))

    # val_acc = evaluate(model, features, labels, val_mask)
    accuracy, precision, recall, f1 = evaluate(g, model, features, labels, test_mask)

    # Calculate classification metrics


    return accuracy, precision, recall, f1