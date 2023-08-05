import argparse

import torch
from torch_geometric.nn.conv.gcn_conv import gcn_norm

from torch_geometric_autoscale.models import GCN
from torch_geometric_autoscale import metis, permute, SubgraphLoader
from torch_geometric_autoscale import get_data, compute_micro_f1

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, required=True,
                    help='Root directory of dataset storage.')
parser.add_argument('--device', type=int, default=0)
args = parser.parse_args()

torch.manual_seed(12345)
device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'

data, in_channels, out_channels = get_data(args.root, name='ogbn-products')

# Pre-process adjacency matrix for GCN:
data.adj_t = gcn_norm(data.adj_t, add_self_loops=True)

# Pre-partition the graph using Metis:
num_parts = 8
perm, ptr = metis(data.adj_t, num_parts=num_parts, log=True)
data = permute(data, perm, log=True)
print("1", data.x[0].size)
x = data.x[0]
print(x.numel() )
print ("1", x.numel() * x.element_size())
# print(perm)
# print(ptr)
# print(data.adj_t)
mapping = [0 for i in range(perm.shape[0])]
for i in range(num_parts):
    start, end = ptr[i], ptr[i+1]
    for j in range(start, end):
        mapping[perm[j]] = i

traffic = [[0 for i in range(num_parts)] for j in range(num_parts)]
node_idx = 0
row, col, _ = data.adj_t.coo()
edge_indices = (row == node_idx).nonzero(as_tuple=False).squeeze()
num_edges = len(edge_indices)
#只算发的. 算p1 到p2而不是所有的. 4 x4 的矩阵
row = row.tolist()
for i , x in enumerate(row):
    # print(x, col[i], mapping[x], mapping[col[i]])
    traffic[mapping[x]][mapping[col[i]]] += 1

for i in range(num_parts):
    for j in range(num_parts):
        print(traffic[i][j], end=' ')
    print()

