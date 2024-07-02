import itertools
import time
from utils.utils import set_seed, set_device
from extension.extend_lp import *
from models.gat_de import GAT
from models.gatv2_de import GATv2
from models.gcn_de import GCN
from models.sage_de import GraphSAGE
import scipy.sparse as sp
from utils.dataset import load_dataset_link_prediction

def evaluate(scores, labels, threshold=0.5):
    # Convert scores to predictions based on threshold
    predictions = (scores >= threshold).astype(int)
    predictions = torch.from_numpy(predictions)
    
    # Calculate confusion matrix
    device = predictions.device
    confusion_matrix = torch.tensor([[torch.sum((predictions == 1) & (labels == 1)).to(device), torch.sum((predictions == 0) & (labels == 1)).to(device)],
                                    [torch.sum((predictions == 1) & (labels == 0)).to(device), torch.sum((predictions == 0) & (labels == 0)).to(device)]])
    
    # Calculate precision and recall
    precision = confusion_matrix[1, 1] / (confusion_matrix[1, 1] + confusion_matrix[0, 1])
    recall = confusion_matrix[1, 1] / (confusion_matrix[1, 1] + confusion_matrix[1, 0])

    # Calculate F1 score
    F1 = 2 * (precision * recall) / (precision + recall)

    return precision, recall, F1


def main_de(seed, dataset_name, extend_metric, model_name, alpha, beta, gamma, explore, add_self_loop=True, verbose=False):
    set_seed(seed)
    set_device()
    
    dataset = load_dataset_link_prediction(dataset_name)
    g = dataset[0]

    # Split edge set for training and testing
    u, v = g.edges()

    eids = np.arange(g.number_of_edges())
    eids = np.random.permutation(eids)
    test_size = int(len(eids) * 0.2)
    train_size = g.number_of_edges() - test_size
    test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]]
    train_pos_u, train_pos_v = u[eids[test_size:]], v[eids[test_size:]]

    # Find all negative edges and split them for training and testing
    adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())), shape=(g.number_of_nodes(), g.number_of_nodes()))
    adj_neg = 1 - adj.todense() - np.eye(g.number_of_nodes())
    neg_u, neg_v = np.where(adj_neg != 0)

    neg_eids = np.random.choice(len(neg_u), g.number_of_edges())
    test_neg_u, test_neg_v = neg_u[neg_eids[:test_size]], neg_v[neg_eids[:test_size]]
    train_neg_u, train_neg_v = neg_u[neg_eids[test_size:]], neg_v[neg_eids[test_size:]]
    train_g = dgl.remove_edges(g, eids[:test_size])
    graph_nx = dgl.to_networkx(train_g)
    graph_undirected = nx.Graph(graph_nx)
    train_pos_g = dgl.graph((train_pos_u, train_pos_v), num_nodes=g.number_of_nodes())
    train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=g.number_of_nodes())

    test_pos_g = dgl.graph((test_pos_u, test_pos_v), num_nodes=g.number_of_nodes())
    test_neg_g = dgl.graph((test_neg_u, test_neg_v), num_nodes=g.number_of_nodes())

    # build a two-layer GNN model
    if model_name == 'sage':
        print("SAGE model is used")
        model = GraphSAGE(train_g.ndata['feat'].shape[1], 16)

    elif model_name == 'gcn':
        print("GCN model is used")
        model = GCN(train_g.ndata['feat'].shape[1], 16)

    elif model_name == 'gat':
        print("GAT model is used")
        num_heads, num_layers, num_out_heads = 8, 2, 1
        num_hidden = n_classes = 16
        in_drop, attn_drop, negative_slope, residual = 0.6, 0.6, 0.2, False

        heads = ([num_heads] * num_layers) + [num_out_heads]
        model = GAT(num_layers, train_g.ndata['feat'].shape[1], num_hidden, n_classes, heads, F.elu, in_drop, attn_drop, negative_slope, residual)

    elif model_name == 'gatv2':
        print("GATv2 model is used")
        num_heads, num_layers, num_out_heads = 8, 2, 1
        num_hidden = n_classes = 16
        in_drop, attn_drop, negative_slope, residual = 0.6, 0.6, 0.2, False

        heads = ([num_heads] * num_layers) + [num_out_heads]
        model = GATv2(num_layers, train_g.ndata['feat'].shape[1], num_hidden, n_classes, heads, F.elu, in_drop, attn_drop, negative_slope, residual)

    else:
        raise("Not a model!")
    

    # pred = DotPredictor()
    pred = MLPPredictor(16)

    if extend_metric != 'None':
        t1 = time.time()
        num_edge_before_extention = train_g.number_of_edges()
        if verbose: 
            print(f'Total number of edges before extension: {num_edge_before_extention}')
        
        train_g = extend_neighborhood(train_g, graph_nx, graph_undirected,  alpha, beta, gamma, explore, extend_metric)
        
        t_total = time.time() - t1
        num_edge_after_extention = train_g.number_of_edges()
        if verbose: 
            print(f'Total number of edges after extension: {num_edge_after_extention}')
            print(f'Total number of added edges: {num_edge_after_extention - num_edge_before_extention}')
            print(f"*** Total construction time in seconds: {t_total:.2f} ***")
    else:
        print("Neighborhood is not extended!")

    if add_self_loop == True:
        if verbose: 
            print(f"Total edges before adding self-loop {train_g.number_of_edges()}")
        train_g = train_g.remove_self_loop().add_self_loop()
        if verbose: 
            print(f"Total edges after adding self-loop {train_g.number_of_edges()}")

    
    optimizer = torch.optim.Adam(itertools.chain(model.parameters(), pred.parameters()), lr=0.01)

    all_logits = []
    for e in range(100):
        # forward
        h = model(train_g, train_g.ndata['feat'])
        pos_score = pred(train_pos_g, h)
        neg_score = pred(train_neg_g, h)
        loss = compute_loss(pos_score, neg_score)
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # if e % 5 == 0:
        #     print('In epoch {}, loss: {}'.format(e, loss))



    # ----------- 5. check results ------------------------ #
    from sklearn.metrics import roc_auc_score
    with torch.no_grad():
        pos_score = pred(test_pos_g, h)
        neg_score = pred(test_neg_g, h)

        scores = torch.cat([pos_score, neg_score]).numpy()
        labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).numpy()

        auc = roc_auc_score(labels, scores)
        precision, recall, F1 = evaluate(scores, labels)
        
        # print('AUC', auc)
        # print('Precision', precision)
        # print('Recall', recall)
        # print('F1-score', F1)
        return auc, precision, recall, F1
            
