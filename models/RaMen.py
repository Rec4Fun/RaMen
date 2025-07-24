import numpy as np
import os 
from typing import Optional, Tuple, Union
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor
from torch.nn import Parameter
from models.utils import TransformerEncoder
from collections import OrderedDict
import scipy.sparse as sp
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import (
    Adj,
    OptTensor,
    PairTensor,
    SparseTensor,
)
from torch_geometric.utils import (
    is_torch_sparse_tensor,
    softmax,
)
from torch_geometric.utils.sparse import set_sparse_value

eps = 1e-9 

class AsymMatrix(MessagePassing):
    _alpha: OptTensor

    def __init__(
            self,
            in_channels: Union[int, Tuple[int, int]],
            out_channels: int,
            heads: int = 1,
            concat: bool = True,
            negative_slope: float = 0.2,
            dropout: float = 0.0,
            add_self_loops: bool = True,
            edge_dim: Optional[int] = None,
            fill_value: Union[float, Tensor, str] = 'mean',
            bias: bool = True,
            extra_layer=False,
            **kwargs,
    ):
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.edge_dim = edge_dim
        self.fill_value = fill_value
        self.extra_layer = extra_layer

        if self.extra_layer:
            self.lin_l = Linear(in_channels, heads * out_channels, bias=bias,
                                weight_initializer='glorot')
            self.lin_r = Linear(in_channels, heads * out_channels, bias=bias,
                                weight_initializer='glorot')

        self.att = Parameter(torch.Tensor(1, heads, out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self._alpha = None

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        if self.extra_layer:
            self.lin_l.reset_parameters()
            self.lin_r.reset_parameters()
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
                edge_attr: OptTensor = None,
                return_attention_weights: bool = None):
        H, C = self.heads, self.out_channels
        x_l: OptTensor = None
        x_r: OptTensor = None
        if isinstance(x, Tensor):
            assert x.dim() == 2
            if self.extra_layer:
                x_l = self.lin_l(x).view(-1, H, C)
                x_r = self.lin_r(x).view(-1, H, C)
            else:
                x = x.expand(self.heads, x.shape[0], x.shape[1]).transpose(0, 1)
                x_l = x_r = x.view(-1, H, C)

        assert x_l is not None
        assert x_r is not None

        out = self.propagate(edge_index, x=(x_l, x_r), edge_attr=edge_attr,
                             size=None)

        alpha = self._alpha
        assert alpha is not None
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias

        if isinstance(return_attention_weights, bool):
            if isinstance(edge_index, Tensor):
                if is_torch_sparse_tensor(edge_index):
                    adj = set_sparse_value(edge_index, alpha)
                    return out, (adj, alpha)
                else:
                    return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def message(self, x_j: Tensor, x_i: Tensor, edge_attr: OptTensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        x = x_i + x_j
        x = F.leaky_relu(x, self.negative_slope)
        alpha = (x * self.att).sum(dim=-1)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)
    
class Amatrix(nn.Module):
    def __init__(self, in_dim, out_dim, n_layer=3, dropout=0.0, heads=2, concat=False, self_loop=True,
                 extra_layer=False):
        super(Amatrix, self).__init__()
        self.num_layer = n_layer
        self.dropout = dropout
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.concat = concat
        self.self_loop = self_loop
        self.extra_layer = extra_layer
        self.convs = nn.ModuleList([AsymMatrix(in_channels=self.in_dim,
                                               out_channels=self.out_dim,
                                               dropout=self.dropout,
                                               heads=self.heads,
                                               concat=self.concat,
                                               add_self_loops=self.self_loop,
                                               extra_layer=self.extra_layer)
                                    for _ in range(self.num_layer)])

    def forward(self, x, edge_index, return_attention_weights=True):
        feats = [x]
        attns = []

        for conv in self.convs:
            x, attn = conv(x, edge_index, return_attention_weights=return_attention_weights)
            feats.append(x)
            attns.append(attn)

        feat = torch.stack(feats, dim=1)
        x = torch.mean(feat, dim=1)
        return x, attns

def laplace_transform(graph):
    rowsum_sqrt = sp.diags(1/(np.sqrt(graph.sum(axis=1).A.ravel()) + 1e-8))
    colsum_sqrt = sp.diags(1/(np.sqrt(graph.sum(axis=0).A.ravel()) + 1e-8))
    graph = rowsum_sqrt @ graph @ colsum_sqrt
    return graph 

def to_tensor(graph):
    graph = graph.tocoo()
    values = graph.data
    indices = np.vstack((graph.row, graph.col))
    graph = torch.sparse_coo_tensor(torch.LongTensor(indices), torch.FloatTensor(values), torch.Size(graph.shape))
    return graph

def init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Parameter):
        nn.init.xavier_uniform_(m)
        
def np_edge_dropout(values, dropout_ratio):
    mask = np.random.choice([0, 1], size=(len(values),), p=[dropout_ratio, 1-dropout_ratio])
    values = mask * values
    return values

def recon_loss_function(recon_x, x):
    negLogLike = torch.sum(F.log_softmax(recon_x, 1) * x, -1) / x.sum(dim=-1)
    negLogLike = -torch.mean(negLogLike)
    return negLogLike

infonce_criterion = nn.CrossEntropyLoss()

def cl_loss_function(a, b, temp=0.2):
    a = nn.functional.normalize(a, dim=-1)
    b = nn.functional.normalize(b, dim=-1)
    logits = torch.mm(a, b.T)
    logits /= temp
    labels = torch.arange(a.shape[0]).to(a.device)
    return infonce_criterion(logits, labels)

class HGNNLayer(nn.Module):
    def __init__(self, n_hyper_layer):
        super(HGNNLayer, self).__init__()
        self.h_layer = n_hyper_layer

    def forward(self, i_hyper, u_hyper, item_embs, bundle_embs=None):
        i_ret = item_embs
        for _ in range(self.h_layer):
            lat = torch.mm(i_hyper.T, i_ret)
            i_ret = torch.mm(i_hyper, lat)
            u_ret = torch.mm(u_hyper, lat)
        return u_ret, i_ret
    
class MLP_multimodal(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_modal = 2 
        self.embedding_size = 64
        self.linear_1 = nn.Linear(self.n_modal * self.embedding_size, self.embedding_size)
        self.relu_1 = nn.ReLU()
        self.linear_2 = nn.Linear(self.embedding_size, self.embedding_size)
        self.relu_2 = nn.ReLU()

    def forward(self, x):
        x = self.linear_1(x)
        x = self.relu_1(x)
        x = self.linear_2(x)
        # x = self.relu_2(x)
        return x 

class HierachicalEncoder(nn.Module):
    def __init__(self, conf, raw_graph, features):
        super(HierachicalEncoder, self).__init__()
        self.conf = conf
        device = self.conf["device"]
        self.device = device
        self.num_user = self.conf["num_users"]
        self.num_bundle = self.conf["num_bundles"]
        self.num_item = self.conf["num_items"]
        self.embedding_size = 64
        self.ui_graph_full, self.bi_graph_train, self.bi_graph_seen = raw_graph
        self.attention_components = self.conf["attention"]
        
        self.content_feature, self.text_feature, self.cf_feature = features

        items_in_train = self.bi_graph_train.sum(axis=0, dtype=bool)
        self.warm_indices = torch.LongTensor(
            np.argwhere(items_in_train)[:, 1]).to(device)
        self.cold_indices = torch.LongTensor(
            np.argwhere(~items_in_train)[:, 1]).to(device)

        # MM >>>
        self.content_feature = nn.functional.normalize(
            self.content_feature, dim=-1)
        self.text_feature = nn.functional.normalize(self.text_feature, dim=-1)
        
        def dense(feature):
            module = nn.Sequential(OrderedDict([
                ('w1', nn.Linear(feature.shape[1], feature.shape[1])),
                ('act1', nn.ReLU()),
                ('w2', nn.Linear(feature.shape[1], 256)),
                ('act2', nn.ReLU()),
                ('w3', nn.Linear(256, 64)),
            ]))

            for m in module:
                init(m)
            return module

        # encoders for media feature
        self.c_encoder = dense(self.content_feature)
        self.t_encoder = dense(self.text_feature)
        # MM <<<

        # BI >>>
        self.item_embeddings = nn.Parameter(
            torch.FloatTensor(self.num_item, self.embedding_size))
        init(self.item_embeddings)
        # BI <<<

        # Multimodal Fusion:
        self.w_q = nn.Linear(self.embedding_size,
                             self.embedding_size, bias=False)
        init(self.w_q)
        self.w_k = nn.Linear(self.embedding_size,
                             self.embedding_size, bias=False)
        init(self.w_k)
        self.w_v = nn.Linear(self.embedding_size,
                             self.embedding_size, bias=False)
        init(self.w_v)
        self.ln = nn.LayerNorm(self.embedding_size, elementwise_affine=False)

        # characteristic setting 
        self.v_feat, self.t_feat = None, None
        v_feat_file_path = f"{conf['data_path']}/{conf['dataset']}/content_feature.pt"
        t_feat_file_path = f"{conf['data_path']}/{conf['dataset']}/description_feature.pt"
        if os.path.isfile(v_feat_file_path):
            self.v_feat = torch.load(v_feat_file_path, weights_only=True).type(torch.FloatTensor).to(self.device).squeeze(1)
        if os.path.isfile(t_feat_file_path):
            self.t_feat = torch.load(t_feat_file_path, weights_only=True).type(torch.FloatTensor).to(self.device).squeeze(1)
        self.mlp_multimodal = MLP_multimodal() # encode concat modality feature 

        # hypergraph setting 
        self.hyper_num = conf["hyper_num"]
        self.n_hyper_layer = 1
        self.tau = 0.2
        self.bi_graph = self.bi_graph_seen
        self.get_item_level_graph_ori()
        self.get_bundle_agg_graph_ori()
        self.hgnnLayer = HGNNLayer(self.n_hyper_layer)
        self.keep_rate = 0.3
        self.drop = nn.Dropout(p=1-self.keep_rate)

        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=True)
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.embedding_size)
            self.v_hyper = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(64, self.hyper_num)))
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=True)
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.embedding_size)
            self.t_hyper = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(64, self.hyper_num)))

        # asymmetric
        self.init_emb_asym()
        self.n_head = 2
        self.iui_edge_index = torch.tensor(
            np.load(f"{conf['data_path']}/{conf['dataset']}/{conf['iui_path']}.npy", allow_pickle=True)).to(self.device)
        self.n_layer = conf["n_layer_gat"]
        self.iui_gat_conv = Amatrix(in_dim=64, out_dim=64, n_layer=self.n_layer, dropout=0.1, heads=self.n_head, concat=False,
                                    self_loop=False, extra_layer=True)
        self.alpha_residual = conf["alpha_residual"]
        self.residual_type = conf["residual_type"]

        # fusion
        self.item_alpha = conf["item_alpha"]
                          
    def scipy_sparse_to_torch_sparse(self, matrix):
        coo = matrix.tocoo() 
        values = coo.data
        indices = np.vstack((coo.row, coo.col)) 
        indices_tensor = torch.LongTensor(indices)  
        values_tensor = torch.FloatTensor(values)
        shape = coo.shape
        shape_tuple = tuple(shape)
        return torch.sparse_coo_tensor(indices_tensor, values_tensor, torch.Size(shape_tuple))
            
    def init_emb_asym(self):
        self.items_embedding = nn.Parameter(torch.FloatTensor(self.num_item, self.embedding_size))
        nn.init.xavier_normal_(self.items_embedding)
        
    def get_item_level_graph_ori(self):
        # create sparse adj matrix 
        bi_graph = self.bi_graph
        item_level_graph = sp.bmat([[sp.csr_matrix((bi_graph.shape[0], bi_graph.shape[0])), bi_graph], [bi_graph.T, sp.csr_matrix((bi_graph.shape[1], bi_graph.shape[1]))]])
        self.item_level_graph_ori = to_tensor(laplace_transform(item_level_graph)).to(self.device)
        
        # create 2 x N adj matrix 
        row, col = bi_graph.nonzero()
        col += (max(row) + 1)
        self.n2_indices = torch.tensor(np.stack([row,col]), dtype=torch.int64) 

    def get_bundle_agg_graph_ori(self):
        bi_graph = self.bi_graph
        device = self.device
        bundle_size = bi_graph.sum(axis=1) + 1e-8 # calculate size for each bundle 
        # print(f"bundle size: {bundle_size.shape}")
        # print(f"diag bundle: {sp.diags(1/bundle_size.A.ravel()).shape}")
        bi_graph = sp.diags(1/bundle_size.A.ravel()) @ bi_graph # sp.diags(1/bundle_size.A.ravel()): D^-1 
        self.bundle_agg_graph_ori = to_tensor(bi_graph).to(device)  
        
    def selfAttention(self, features):
        # features: [bs, #modality, d]
        if "layernorm" in self.attention_components:
            features = self.ln(features)
        q = self.w_q(features)
        k = self.w_k(features)
        if "w_v" in self.attention_components:
            v = self.w_v(features)
        else:
            v = features
        # [bs, #modality, #modality]
        attn = q.mul(self.embedding_size ** -0.5) @ k.transpose(-1, -2)
        attn = attn.softmax(dim=-1)

        features = attn @ v  # [bs, #modality, d]
        # average pooling
        y = features.mean(dim=-2)  # [bs, d]

        return y
  
    def forward_all(self, item_asymmetric, fusion_hypergraph_item, test=False):
        c_feature = self.c_encoder(self.content_feature) # MLP projector
        t_feature = self.t_encoder(self.text_feature) # MLP projector 
        mm_feature_cat = torch.cat([c_feature, t_feature], dim=1) 
        mm_feature_full = self.mlp_multimodal(mm_feature_cat)

        features = [mm_feature_full]
        
        bi_feature_full = self.item_embeddings
        features.append(bi_feature_full)

        # stack -> tensor 
        features = torch.stack(features, dim=-2) # [bs, #modality, d]

        # modality fusion 
        final_feature = self.selfAttention(F.normalize(features, dim=-1)) # [bs, d]

        # fusion with asymmetric embs 
        final_feature = (self.item_alpha)*final_feature + (1-self.item_alpha)*item_asymmetric

        return final_feature, fusion_hypergraph_item, item_asymmetric
     
    def forward(self, seq_modify, all=False, test=False, discrete=False):
        cb_item_feat_iui, _ = self.iui_gat_conv(self.items_embedding, self.iui_edge_index, return_attention_weights=True)

        # add residual connect 
        if self.residual_type == 0: # best performance 
            cb_item_feat_iui = cb_item_feat_iui + self.alpha_residual * self.items_embedding
        else:
            cb_item_feat_iui = self.alpha_residual * cb_item_feat_iui + (1 - self.alpha_residual) * self.items_embedding

        cb_bundle_feat = self.bundle_agg_graph_ori @ cb_item_feat_iui
        cb_item_feat = cb_item_feat_iui

        # using in clhe & hypergraph module
        c_feature = self.c_encoder(self.content_feature) # MLP projector
        t_feature = self.t_encoder(self.text_feature) # MLP projector
       
        # hypergraph module 
        bi_graph_torch_sparse = self.scipy_sparse_to_torch_sparse(self.bi_graph).to(self.device)
        if self.v_feat is not None:
            iv_hyper = torch.mm(c_feature, self.v_hyper)
            bv_hyper = torch.mm(bi_graph_torch_sparse, iv_hyper)
            # gumbel softmax: augment 
            iv_hyper = F.gumbel_softmax(iv_hyper, self.tau, dim=1, hard=False) 
            bv_hyper = F.gumbel_softmax(bv_hyper, self.tau, dim=1, hard=False)
        if self.t_feat is not None:
            it_hyper = torch.mm(t_feature, self.t_hyper)
            bt_hyper = torch.mm(bi_graph_torch_sparse, it_hyper)
            it_hyper = F.gumbel_softmax(it_hyper, self.tau, dim=1, hard=False)
            bt_hyper = F.gumbel_softmax(bt_hyper, self.tau, dim=1, hard=False)
        
        bv_hyper_embs, iv_hyper_embs = self.hgnnLayer(self.drop(iv_hyper), self.drop(bv_hyper), cb_item_feat, cb_bundle_feat)
        bt_hyper_embs, it_hyper_embs = self.hgnnLayer(self.drop(it_hyper), self.drop(bt_hyper), cb_item_feat, cb_bundle_feat)
        av_hyper_embs = torch.concat([bv_hyper_embs, iv_hyper_embs], dim=0)     
        at_hyper_embs = torch.concat([bt_hyper_embs, it_hyper_embs], dim=0) 
        ghe_embs = av_hyper_embs + at_hyper_embs
        ghe_embs = F.normalize(ghe_embs) # important 
        
        fusion_hypergraph_bundle, fusion_hypergraph_item = torch.split(ghe_embs, [self.num_bundle, self.num_item], dim=0)

        if all is True:
            return self.forward_all(cb_item_feat, fusion_hypergraph_item) 


        # CLHE module 
        modify_mask = seq_modify == self.num_item
        seq_modify.masked_fill_(modify_mask, 0)

        mm_feature_cat = torch.cat([c_feature, t_feature], dim=1) 
        mm_feature_full = self.mlp_multimodal(mm_feature_cat) # MLP projector 

        mm_feature = mm_feature_full[seq_modify]  # [bs, n_token, d]
        bs,n_token,d = mm_feature.shape

        features = [mm_feature_full]

        bi_feature_full = self.item_embeddings
        features.append(bi_feature_full)
        features = torch.stack(features, dim=-2)  # [bs, n_token, #modality, d]        

        # modality fusion 
        N_modal = 2
        final_feature = self.selfAttention(
            F.normalize(features.view(-1, N_modal, d), dim=-1))
        # reshape: represent bundle embs 
        final_feature = final_feature[seq_modify] # convert shape -> [bs, n_token, d]
        final_feature = final_feature.view(bs, n_token, d)

        return final_feature, fusion_hypergraph_bundle, cb_bundle_feat, cb_item_feat, fusion_hypergraph_item

class RAMEN(nn.Module):
    def __init__(self, conf, raw_graph, features):
        super().__init__()
        self.conf = conf
        device = self.conf["device"]
        self.device = device
        self.num_user = self.conf["num_users"]
        self.num_bundle = self.conf["num_bundles"]
        self.num_item = self.conf["num_items"]
        self.embedding_size = 64
        self.ui_graph, self.bi_graph_train, self.bi_graph_seen = raw_graph

        self.encoder = HierachicalEncoder(conf, raw_graph, features)
        self.decoder = HierachicalEncoder(conf, raw_graph, features)

        self.bundle_encode = TransformerEncoder(conf={
            "n_layer": conf["trans_layer"],
            "dim": 64,
            "num_token": 100,
            "device": self.device,
        }, data={"sp_graph": self.bi_graph_seen})
        
        # fusion asymmetric 
        self.bundle_alpha = conf["bundle_alpha"]

        # contrastive learning 
        self.alpha_item_loss = conf["alpha_item_loss"]
        self.alpha_bundle_loss = conf["alpha_bundle_loss"]
        self.tau = 0.2 
                  
    def forward(self, batch):
        idx, full, seq_full, modify, seq_modify = batch  # x: [bs, #items]

        mask = seq_full == self.num_item
        feat_bundle_view, fusion_hypergraph_bundle, cb_bundle_feat, _, _ = self.encoder(seq_full)  # [bs, n_token, d]
        # bundle feature construction >>>
        
        bundle_feature = self.bundle_encode(feat_bundle_view, mask=mask)
        
        feat_retrival_view, fusion_hypergraph_item, _ = self.decoder(batch, all=True)
        bundle_feature = self.bundle_alpha * bundle_feature + (1 - self.bundle_alpha) * cb_bundle_feat[idx]
        feat_retrival_view = feat_retrival_view
        
        logits = bundle_feature @ feat_retrival_view.T + fusion_hypergraph_bundle[idx] @ fusion_hypergraph_item.T
        loss = recon_loss_function(logits, full)  # main_loss

        # item-level contrastive learning 
        items_in_batch = torch.argwhere(full.sum(dim=0)).squeeze()            
        a = feat_retrival_view[items_in_batch]
        b = fusion_hypergraph_item[items_in_batch]
        item_loss_contrastive = self.alpha_item_loss*cl_loss_function(
                    a.view(-1, self.embedding_size), b.view(-1, self.embedding_size), self.tau)

        # bundle-level contrastive learning 
        bundle_loss_contrastive = self.alpha_bundle_loss*cl_loss_function(
                bundle_feature.view(-1, self.embedding_size), fusion_hypergraph_bundle[idx].view(-1, self.embedding_size), self.tau)

        return {
            'loss': loss + item_loss_contrastive + bundle_loss_contrastive,
            'item_loss': item_loss_contrastive.detach(),
            'bundle_loss': bundle_loss_contrastive.detach()
        }
        
    def evaluate(self, _, batch):
        idx, x, seq_x = batch
        mask = seq_x == self.num_item

        feat_bundle_view, fusion_hypergraph_bundle, cb_bundle_feat, _, _ = self.encoder(seq_x)
        bundle_feature = self.bundle_encode(feat_bundle_view, mask=mask)        
        feat_retrival_view, fusion_hypergraph_item, _ = self.decoder(
            (idx, x, seq_x, None, None), all=True)
        bundle_feature = self.bundle_alpha * bundle_feature + (1 - self.bundle_alpha) * cb_bundle_feat[idx]

        logits = bundle_feature @ feat_retrival_view.T + fusion_hypergraph_bundle[idx] @ fusion_hypergraph_item.T

        return logits

    def propagate(self, test=False):
        return None
