import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax
# from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
from torch_geometric.nn import MLP, SumAggregation, MeanAggregation, MaxAggregation, Set2Set, AttentionalAggregation, SortAggregation, GraphMultisetTransformer, GRUAggregation, MLPAggregation, DeepSetsAggregation, SetTransformerAggregation
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.nn.inits import glorot, zeros

from torch import nn, einsum
from torch.nn import Parameter
from collections import defaultdict as ddict
from convs import *
from jumpingknowledge import *
from searchspace import *


num_atom_type = 120 #including the extra mask tokens
num_chirality_tag = 3

num_bond_type = 6 #including aromatic and self-loop edge, and extra masked tokens
num_bond_direction = 3


class GNN(torch.nn.Module):
    """


    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        JK (str): last, concat, max or sum.
        max_pool_layer (int): the layer from which we use max pool rather than add pool for neighbor aggregation
        drop_ratio (float): dropout rate
        gnn_type: gin, gcn, graphsage, gat

    Output:
        node representations

    """

    def __init__(self, num_layer, emb_dim, drop_ratio=0, gnn_type="gin", ft_mode='fully', adapters=None, adapter_ws=None):
        super(GNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        # self.JK = JK
        self.ft_mode = ft_mode

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.x_embedding1 = torch.nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = torch.nn.Embedding(num_chirality_tag, emb_dim)

        torch.nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        ###List of MLPs
        self.gnns = torch.nn.ModuleList()
        for layer in range(num_layer):
            if gnn_type == "gin":
                self.gnns.append(GINConv(emb_dim, aggr="add"))
            elif gnn_type == "gcn":
                self.gnns.append(GCNConv(emb_dim))
            elif gnn_type == "gat":
                self.gnns.append(GATConv(emb_dim))
            elif gnn_type == "graphsage":
                self.gnns.append(GraphSAGEConv(emb_dim))

        ###List of batchnorms
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

        ###List of adapters
        self.adapters = adapters
        self.adapter_ws = adapter_ws


    # early version
    # def adapter_fusion(self, layer, h_in, edge_index, edge_attr):
    #     h_out_list = []
    #     for adapter_mode in ADAPTER_CANDIDATES:
    #         if adapter_mode == 'skip':
    #             h_out = self.gnns[layer](h_in, edge_index, edge_attr)
    #         elif adapter_mode == 'sequential':
    #             h_out = self.gnns[layer](h_in, edge_index, edge_attr)
    #             h_out = self.adapters[layer](h_out)
    #         elif adapter_mode == 'parallel':
    #             h_out = self.gnns[layer](h_in, edge_index, edge_attr) + self.adapters[layer](h_in)
    #         h_out_list.append(h_out)
    #     h_out = einsum('i,ijk->jk', self.adapter_ws[layer], torch.stack(h_out_list))
    #     return h_out


    # 0620 version
    # def adapter_fusion(self, layer, h_in, edge_index, edge_attr):
    #     assert len(ADAPTER_CANDIDATES) <= 2 # se; se/none search (freeze ginconvs)
    #     h_out_list = []
    #     for adapter_mode in ADAPTER_CANDIDATES:
    #         if adapter_mode == 'skip':
    #             h_out = self.gnns[layer](h_in, edge_index, edge_attr)
    #         elif adapter_mode == 'sequential':
    #             h_out = self.gnns[layer](h_in, edge_index, edge_attr)
    #             h_out = self.adapters[layer](h_out)
    #         else:
    #             raise NotImplementedError
    #         h_out_list.append(h_out)
    #     h_out = einsum('i,ijk->jk', self.adapter_ws[layer], torch.stack(h_out_list))
    #     return h_out


    # # 0621 version1
    # def adapter_fusion(self, layer, h_in, h_out_temp, ft_mode='auto_adapter_pa'):
    #     # se; se/none search (freeze ginconvs)
    #     assert len(ADAPTER_CANDIDATES) <= 2 and ft_mode in ['auto_adapter_pa', 'auto_adapter_se']
    #     h_out_list = []
    #     adapter_modes = ['skip', 'parallel'] if ft_mode == 'auto_adapter_pa' else ['skip', 'sequential']
    #     for adapter_mode in adapter_modes:
    #         if adapter_mode == 'skip':
    #             h_out = h_out_temp
    #         if adapter_mode == 'parallel':
    #             h_out = h_out_temp + self.adapters[layer](h_in)
    #         if adapter_mode == 'sequential':
    #             h_out = self.adapters[layer](h_out_temp)
    #         h_out_list.append(h_out)
    #     h_out = einsum('i,ijk->jk', self.adapter_ws[layer], torch.stack(h_out_list))
    #     return h_out


    # 0621 version2
    def adapter_fusion(self, layer, h_in, h_out_temp, ft_mode='auto_adapter_pa'):
        assert ft_mode in ['adapter_pa', 'adapter_se', 'auto_adapter_pa', 'auto_adapter_se',
                           'fully_adapter_pa', 'fully_adapter_se', 'fully_auto_adapter_pa', 'fully_auto_adapter_se',
                           'fully_skip',
                           'fully_latest']
        h_out_list = []

        if ft_mode in ['adapter_pa', 'fully_adapter_pa']:
            return (h_out_temp + self.adapters[layer](h_in))

        elif ft_mode in ['adapter_se', 'fully_adapter_se']:
            return (self.adapters[layer](h_out_temp))

        elif ft_mode in ['auto_adapter_pa', 'fully_auto_adapter_pa']:
            # adapter_modes = ['skip', 'parallel']
            h_out_list.extend([h_out_temp, h_out_temp + self.adapters[layer](h_in)])
            return einsum('i,ijk->jk', self.adapter_ws[layer], torch.stack(h_out_list))

        elif ft_mode in ['auto_adapter_se', 'fully_auto_adapter_se']:
            # adapter_modes = ['skip', 'sequential']
            h_out_list.extend([h_out_temp, self.adapters[layer](h_out_temp)])
            return einsum('i,ijk->jk', self.adapter_ws[layer], torch.stack(h_out_list))

        elif ft_mode in ['fully_skip']:
            h_out_list.extend([h_out_temp, h_out_temp + h_in])
            return einsum('i,ijk->jk', self.adapter_ws[layer], torch.stack(h_out_list))

        elif ft_mode in ['fully_latest']:
            h_out_list.extend([h_out_temp,
                               h_out_temp + h_in,
                               h_out_temp + self.adapters[layer](h_in)])
            return einsum('i,ijk->jk', self.adapter_ws[layer], torch.stack(h_out_list))

        else:
            raise NotImplementedError


    # def forward(self, x, edge_index, edge_attr):
    def forward(self, *argv):
        if len(argv) == 3:
            x, edge_index, edge_attr = argv[0], argv[1], argv[2]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        else:
            raise ValueError("unmatched number of arguments.")

        x = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1])

        h_list = [x]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)

            if self.ft_mode in ['fully', 'decoder_only', 'last_1', 'last_2', 'last_3', 'last_4']:
                pass

            elif self.ft_mode in ['adapter_pa', 'adapter_se', 'auto_adapter_pa', 'auto_adapter_se',
                                  'fully_adapter_pa', 'fully_adapter_se', 'fully_auto_adapter_pa', 'fully_auto_adapter_se',
                                  'fully_skip',
                                  'fully_latest']:
                h = self.adapter_fusion(layer, h_in=h_list[layer], h_out_temp=h, ft_mode=self.ft_mode)

            else:
                raise NotImplementedError

            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
            h_list.append(h)
        return h_list


class GNN_graphpred(torch.nn.Module):
    """
    Extension of GIN to incorporate edge information by concatenation.

    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        num_tasks (int): number of tasks in multi-task learning scenario
        drop_ratio (float): dropout rate
        JK (str): last, concat, max or sum.
        graph_pooling (str): sum, mean, max, attention, set2set
        gnn_type: gin, gcn, graphsage, gat
        
    See https://arxiv.org/abs/1810.00826
    JK-net: https://arxiv.org/abs/1806.03536
    """
    def __init__(self, num_layer, emb_dim, num_tasks, ft_mode='fully', adapter_dim=None, drop_ratio = 0, gnn_type = "gin", temp = 0.1, mode = "search"):
        super(GNN_graphpred, self).__init__()

        self.mode = mode
        self.ft_mode = ft_mode
        self.temp = temp
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        # self.JK = JK
        self.emb_dim = emb_dim
        self.adapter_dim = adapter_dim
        self.num_tasks = num_tasks
        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.setup_supernet()

        self.gnn = GNN(num_layer, emb_dim, drop_ratio, gnn_type = gnn_type, ft_mode=self.ft_mode, adapters=self.adapters)
        #For graph-level binary classification
        # self.mult = 2 if graph_pooling[:-1] == "set2set" else 1
        # self.graph_pred_linear = torch.nn.Linear(self.mult * self.emb_dim, self.num_tasks)
        self.graph_pred_linear = torch.nn.Linear(self.emb_dim, self.num_tasks)


    def from_pretrained(self, model_file):
        self.gnn.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')), strict=False)


    def setup_supernet(self):
        self.MAP = {'ADAPTER': ADAPTER_CANDIDATES, 'JK': JK_CANDIDATES, 'POOL': POOL_CANDIDATES}
        # self.search_jk, self.search_pool = len(JK_CANDIDATES) > 1, len(POOL_CANDIDATES) > 1
        self.searched_arch = ddict(list)

        self.get_adapters()
        self.get_jks()
        self.get_pools()
        self.init_alphas()


    def get_adapters(self):
        self.adapters = nn.ModuleList([])
        if ('adapter' in self.ft_mode) or ('fully' in self.ft_mode):
            for k in range(self.num_layer):
                self.adapters.append(MLP([self.emb_dim, self.adapter_dim, self.emb_dim]))
        else:
            pass


    def get_jks(self):
        self.jks = nn.ModuleList([])
        self.jk_concat_ln = nn.Linear((self.num_layer + 1) * self.emb_dim, self.emb_dim)

        for jk_mode in JK_CANDIDATES:
            if jk_mode == 'concat':
                self.jks.append(nn.Sequential(JumpingKnowledge(mode='concat'), self.jk_concat_ln))
            elif jk_mode == 'last':
                self.jks.append(JumpingKnowledge(mode='last'))
            elif jk_mode == 'max':
                self.jks.append(JumpingKnowledge(mode='max'))
            elif jk_mode == 'mean':
                self.jks.append(JumpingKnowledge(mode='mean'))
            elif jk_mode == 'att':
                self.jks.append(JumpingKnowledge(mode='att', K=self.num_layer))
            elif jk_mode == 'gpr':
                self.jks.append(JumpingKnowledge(mode='gpr', K=self.num_layer))
            elif jk_mode == 'lstm':
                self.jks.append(JumpingKnowledge(mode='lstm', channels=self.emb_dim, num_layers=2))
            elif jk_mode == 'node_adaptive':
                self.jks.append(JumpingKnowledge(mode='node_adaptive', channels=self.emb_dim))
            else:
                raise NotImplementedError


    def get_pools(self):
        self.pools = nn.ModuleList([])
        self.processing_steps = 2
        self.k = 10
        self.pool_set2set_ln = nn.Linear(self.processing_steps * self.emb_dim, self.emb_dim)
        self.pool_sort_ln = nn.Linear(self.k * self.emb_dim, self.emb_dim)

        for pool_mode in POOL_CANDIDATES:
            #Different kind of graph pooling
            if pool_mode == 'sum':
                self.pools.append(SumAggregation())
            elif pool_mode == 'mean':
                self.pools.append(MeanAggregation())
            elif pool_mode == 'max':
                self.pools.append(MaxAggregation())

            elif pool_mode == 'set2set':
                self.pools.append(Set2Set(in_channels=self.emb_dim, processing_steps=self.processing_steps))

            elif pool_mode in 'att':
                self.pools.append(AttentionalAggregation(gate_nn=nn.Linear(self.emb_dim, 1)))

            elif pool_mode in 'sort':
                self.pools.append(SortAggregation(k=self.k))

            elif pool_mode == 'gmt':
                self.pools.append(GraphMultisetTransformer(channels=self.emb_dim, k=self.k))

            elif pool_mode == 'gru':
                self.pools.append(GRUAggregation(in_channels=self.emb_dim, out_channels=self.emb_dim))
            # elif pool_mode == 'mlp':
            #     self.pools.append(MLPAggregation(in_channels=self.emb_dim, out_channels=self.emb_dim))
            elif pool_mode == 'ds':
                self.pools.append(DeepSetsAggregation(local_nn=MLP([self.emb_dim, self.emb_dim, self.emb_dim]),
                                                      global_nn=MLP([self.emb_dim, self.emb_dim, self.emb_dim])))
            # elif pool_mode == 'st':
            #     self.pools.append(SetTransformerAggregation(channels=self.emb_dim, concat=False))

            else:
                raise ValueError("Invalid graph pooling type.")


    def init_alphas(self):
        loc_mean, loc_std = 1, 0.01
        self.adapter_alphas = Parameter(torch.ones((self.num_layer, len(ADAPTER_CANDIDATES))).normal_(loc_mean, loc_std))
        self.jk_alphas = Parameter(torch.ones((1, len(JK_CANDIDATES))).normal_(loc_mean, loc_std))
        self.pool_alphas = Parameter(torch.ones((1, len(POOL_CANDIDATES))).normal_(loc_mean, loc_std))


    def get_categ_masks(self):
        def relaxation(alphas, temp):
            log_alphas = alphas
            u = torch.zeros_like(log_alphas).uniform_()
            softmax = torch.nn.Softmax(-1)
            ws = softmax((log_alphas + (-((-(u.log())).log()))) / temp)

            values, indices = ws.max(dim=1)
            ws_onehot = torch.zeros_like(ws).scatter_(1, indices.view(-1, 1), 1)
            ws_onehot = (ws_onehot - ws).detach() + ws
            return ws_onehot

        self.adapter_ws = relaxation(self.adapter_alphas, self.temp)
        self.jk_ws = relaxation(self.jk_alphas, self.temp)
        self.pool_ws = relaxation(self.pool_alphas, self.temp)


    def fusion_trans(self, h_list):
        # h_list with shape: (K+1) * N * d
        # this function maps: (K+1) * N * d -> N * d (representations for nodes)
        node_representation_list = [jk(h_list) for jk in self.jks]
        node_representation = einsum('i,ijk->jk', self.jk_ws[0], torch.stack(node_representation_list))
        return node_representation


    def readout_trans(self, node_representation, batch):
        # h_fused with shape: N * d
        # this function maps: N * d -> d (representations for graphs)
        graph_representation_list = []
        for pool_mode, pool in zip(POOL_CANDIDATES, self.pools):
            hg = pool(node_representation, batch)
            if pool_mode == 'set2set':
                hg = self.pool_set2set_ln(hg)
            if pool_mode == 'sort':
                hg = self.pool_sort_ln(hg)
            graph_representation_list.append(hg)
        graph_representation = einsum('i,ijk->jk', self.pool_ws[0], torch.stack(graph_representation_list))
        return graph_representation


    def derive_arch(self):
        def get_alphas_onehot(alphas, key):  # _get_searched_ops
            # MAP = {'JK': JK_CANDIDATES, 'POOL': POOL_CANDIDATES}
            values, indices = alphas.max(dim=1)
            alphas_onehot = torch.zeros_like(alphas).scatter_(1, indices.view(-1, 1), 1)
            searched_arch = np.array(self.MAP[key])[indices.cpu()]
            return alphas_onehot, searched_arch

        self.adapter_ws, self.searched_arch['ADAPTER'] = get_alphas_onehot(self.adapter_alphas.clone().detach(), 'ADAPTER')
        self.jk_ws, self.searched_arch['JK'] = get_alphas_onehot(self.jk_alphas.clone().detach(), 'JK')
        self.pool_ws, self.searched_arch['POOL'] = get_alphas_onehot(self.pool_alphas.clone().detach(), 'POOL')


    def from_searched(self, model):
        self.adapter_alphas, self.adapter_ws = Parameter(model.adapter_alphas, requires_grad=False), Parameter(model.adapter_ws, requires_grad=False)
        self.jk_alphas, self.jk_ws = Parameter(model.jk_alphas, requires_grad=False), Parameter(model.jk_ws, requires_grad=False)
        self.pool_alphas, self.pool_ws = Parameter(model.pool_alphas, requires_grad=False), Parameter(model.pool_ws, requires_grad=False)
        self.searched_arch = model.searched_arch
        self.best_test_w_val_epoch, self.best_test_wo_val_epoch = None, None
        self.best_test_w_val, self.best_test_wo_val = None, None


    def forward(self, *argv):
        if len(argv) == 4:
            x, edge_index, edge_attr, batch = argv[0], argv[1], argv[2], argv[3]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        else:
            raise ValueError("unmatched number of arguments.")

        if self.mode == 'search':
            self.get_categ_masks()
        self.gnn.adapter_ws = self.adapter_ws
        # else: see self.from_searched()

        h_list = self.gnn(x, edge_index, edge_attr)
        node_representation = self.fusion_trans(h_list)
        graph_representation = self.readout_trans(node_representation, batch)
        return self.graph_pred_linear(graph_representation)


if __name__ == "__main__":
    pass

