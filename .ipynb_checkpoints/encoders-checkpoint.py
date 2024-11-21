import math

import torch
import torch.nn as nn
from tabpfn.utils import normalize_data
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import numpy as np
from scripts import sparsemax
from imblearn.over_sampling import SMOTE

sparsemax = Sparsemax(dim=1)

# GLU 
def glu(act, n_units):
    
    act[:, :n_units] = act[:, :n_units].clone() * torch.nn.Sigmoid()(act[:, n_units:].clone())     
    
    return act

class Linear(nn.Linear):
    def __init__(self, num_features, emsize, replace_nan_by_zero=False):
        super().__init__(num_features, emsize)
        self.num_features = num_features
        self.emsize = emsize
        self.replace_nan_by_zero = replace_nan_by_zero

    def forward(self, x):
        if self.replace_nan_by_zero:
            x = torch.nan_to_num(x, nan=0.0)
        return super().forward(x)

    def __setstate__(self, state):
        super().__setstate__(state)
        self.__dict__.setdefault('replace_nan_by_zero', True)

'''
#smote+顺序注意力
class TabNetModel(nn.Module):
    def __init__(
        self,
        columns=3,
        num_features=3,
        feature_dims=128,
        output_dim=64,
        num_decision_steps=6,
        relaxation_factor=0.5,
        batch_momentum=0.001,
        virtual_batch_size=2,
        num_classes=2,
        epsilon=0.00001,
        emsize=64,
        apply_smote=False,  # 新增参数
        use_sequential_attention=False  # 新增参数
    ):
        super().__init__()
        
        # 模型参数初始化
        self.columns = columns
        self.num_features = num_features
        self.feature_dims = feature_dims
        self.output_dim = output_dim
        self.num_decision_steps = num_decision_steps
        self.relaxation_factor = relaxation_factor
        self.batch_momentum = batch_momentum
        self.virtual_batch_size = virtual_batch_size
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.apply_smote = apply_smote  # 新增参数
        self.use_sequential_attention = use_sequential_attention  # 新增参数

        # 特征变换层
        self.feature_transform_linear1 = torch.nn.Linear(num_features, self.feature_dims * 2, bias=False)
        self.BN = torch.nn.BatchNorm1d(num_features, momentum=batch_momentum)
        self.BN1 = torch.nn.BatchNorm1d(self.feature_dims * 2, momentum=batch_momentum)

        self.feature_transform_linear2 = torch.nn.Linear(self.feature_dims * 2, self.feature_dims * 2, bias=False)
        self.feature_transform_linear3 = torch.nn.Linear(self.feature_dims * 2, self.feature_dims * 2, bias=False)
        self.feature_transform_linear4 = torch.nn.Linear(self.feature_dims * 2, self.feature_dims * 2, bias=False)

        self.mask_linear_layer = torch.nn.Linear(self.feature_dims * 2 - output_dim, self.num_features, bias=False)
        self.BN2 = torch.nn.BatchNorm1d(self.num_features, momentum=batch_momentum)

        self.final_classifier_layer = torch.nn.Linear(self.output_dim, self.num_classes, bias=False)

        # Linear 类实例
        self.embedding_layer = nn.Linear(num_features, emsize)
        
        # SMOTE 实例化
        if self.apply_smote:
            self.smote = SMOTE(sampling_strategy='minority')

        # 顺序注意力机制（如果启用）
        if self.use_sequential_attention:
            self.attention_layer = nn.MultiheadAttention(embed_dim=self.feature_dims * 2, num_heads=4, batch_first=True)

    def forward(self, data, targets=None):
        # 应用 SMOTE 平衡处理
        if self.apply_smote and targets is not None:
            # 将数据转换为 numpy 格式，SMOTE 需要二维数据
            data_np = data.cpu().numpy()
            targets_np = targets.cpu().numpy()
            
            # 使用 SMOTE 生成平衡后的数据
            data_resampled, targets_resampled = self.smote.fit_resample(data_np, targets_np)
            
            # 转换回 Tensor
            data = torch.tensor(data_resampled, dtype=torch.float32, device=data.device)
            targets = torch.tensor(targets_resampled, dtype=torch.long, device=data.device)
        
        batch_size = data.shape[0]
        features = self.BN(data)
        output_aggregated = torch.zeros([batch_size, self.output_dim], device=data.device)
        
        masked_features = features
        mask_values = torch.zeros([batch_size, self.num_features], device=data.device)
        
        aggregated_mask_values = torch.zeros([batch_size, self.num_features], device=data.device)
        complemantary_aggregated_mask_values = torch.ones([batch_size, self.num_features], device=data.device)
        
        total_entropy = 0

        for ni in range(self.num_decision_steps):
            if ni == 0:
                transform_f1 = self.feature_transform_linear1(masked_features)
                norm_transform_f1 = self.BN1(transform_f1)

                transform_f2 = self.feature_transform_linear2(norm_transform_f1)
                norm_transform_f2 = self.BN1(transform_f2)
            
            else:
                transform_f1 = self.feature_transform_linear1(masked_features)
                norm_transform_f1 = self.BN1(transform_f1)

                transform_f2 = self.feature_transform_linear2(norm_transform_f1)
                norm_transform_f2 = self.BN1(transform_f2)

                transform_f2 = (self.glu(norm_transform_f2, self.feature_dims) + transform_f1) * np.sqrt(0.5)

                transform_f3 = self.feature_transform_linear3(transform_f2)
                norm_transform_f3 = self.BN1(transform_f3)

                transform_f4 = self.feature_transform_linear4(norm_transform_f3)
                norm_transform_f4 = self.BN1(transform_f4)

                transform_f4 = (self.glu(norm_transform_f4, self.feature_dims) + transform_f3) * np.sqrt(0.5)
                
                # 使用顺序注意力机制
                if self.use_sequential_attention:
                    # 使用注意力层进行变换
                    norm_transform_f4, _ = self.attention_layer(norm_transform_f4.unsqueeze(0), norm_transform_f4.unsqueeze(0), norm_transform_f4.unsqueeze(0))
                    norm_transform_f4 = norm_transform_f4.squeeze(0)
                
                decision_out = torch.relu(transform_f4[:, :self.output_dim])
                output_aggregated = torch.add(decision_out, output_aggregated)
                scale_agg = torch.sum(decision_out, axis=1, keepdim=True) / (self.num_decision_steps - 1)
                aggregated_mask_values = torch.add(aggregated_mask_values, mask_values * scale_agg)

                features_for_coef = (transform_f4[:, self.output_dim:])
                
                if ni < (self.num_decision_steps - 1):
                    mask_linear_layer = self.mask_linear_layer(features_for_coef)
                    mask_linear_norm = self.BN2(mask_linear_layer)
                    mask_linear_norm = torch.mul(mask_linear_norm, complemantary_aggregated_mask_values)
                    mask_values = self.sparsemax(mask_linear_norm)
                    
                    complemantary_aggregated_mask_values = torch.mul(complemantary_aggregated_mask_values, self.relaxation_factor - mask_values)
                    total_entropy = torch.add(total_entropy, torch.mean(torch.sum(-mask_values * torch.log(mask_values + self.epsilon), axis=1)) / (self.num_decision_steps - 1))
                    masked_features = torch.mul(mask_values, features)

            embedded_features = self.embedding_layer(masked_features)
        
        return output_aggregated, total_entropy, masked_features, embedded_features
'''

#只有顺序注意力+启用顺序注意力参数设置
class TabNetModel(nn.Module):
    
    def __init__(
        self,
        columns = 3,
        num_features = 3,
        feature_dims = 128,
        output_dim  =64,
        num_decision_steps =6,
        relaxation_factor = 0.5,
        batch_momentum = 0.001,
        virtual_batch_size = 2,
        num_classes = 2,
        epsilon = 0.00001,
        emsize=64,
        use_sequential_attention=False  # 新增参数
    ):
        
        super().__init__()
        
        self.columns = columns
        self.num_features  = num_features
        self.feature_dims = feature_dims
        self.output_dim = output_dim
        self.num_decision_steps = num_decision_steps
        self.relaxation_factor = relaxation_factor
        self.batch_momentum = batch_momentum
        self.virtual_batch_size = virtual_batch_size
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_sequential_attention = use_sequential_attention  # 新增参数
        
        self.feature_transform_linear1 = torch.nn.Linear(num_features, self.feature_dims * 2, bias=False)
        self.BN = torch.nn.BatchNorm1d(num_features, momentum = batch_momentum)
        self.BN1 = torch.nn.BatchNorm1d(self.feature_dims * 2, momentum = batch_momentum)
        
        self.feature_transform_linear2 = torch.nn.Linear(self.feature_dims * 2, self.feature_dims * 2, bias=False)
        self.feature_transform_linear3 = torch.nn.Linear(self.feature_dims * 2, self.feature_dims * 2, bias=False)
        self.feature_transform_linear4 = torch.nn.Linear(self.feature_dims * 2, self.feature_dims * 2, bias=False)
        
        self.mask_linear_layer = torch.nn.Linear(self.feature_dims * 2-output_dim, self.num_features, bias=False)
        self.BN2 = torch.nn.BatchNorm1d(self.num_features, momentum = batch_momentum)
        
        self.final_classifier_layer = torch.nn.Linear(self.output_dim, self.num_classes, bias=False)
        
        # 创建 Linear 类的实例
        self.embedding_layer = Linear(num_features, emsize)
        
        # 顺序注意力机制（如果启用）
        if self.use_sequential_attention:
            self.attention_layer = nn.MultiheadAttention(embed_dim=self.feature_dims * 2, num_heads=4, batch_first=True)

    
    def encoder(self, data):
        
        batch_size = data.shape[0]
        features = self.BN(data)
        output_aggregated = torch.zeros([batch_size, self.output_dim])
        
        masked_features = features
        mask_values = torch.zeros([batch_size, self.num_features])
        
        aggregated_mask_values = torch.zeros([batch_size, self.num_features])
        complemantary_aggregated_mask_values =torch.ones([batch_size, self.num_features])
        
        total_entropy = 0

        for ni in range(self.num_decision_steps):
            
            if ni==0:
                
                transform_f1  = self.feature_transform_linear1(masked_features)
                norm_transform_f1 = self.BN1(transform_f1)

                transform_f2      = self.feature_transform_linear2(norm_transform_f1)
                norm_transform_f2 = self.BN1(transform_f2)
            
            else:

                transform_f1 = self.feature_transform_linear1(masked_features)
                norm_transform_f1 = self.BN1(transform_f1)

                transform_f2      = self.feature_transform_linear2(norm_transform_f1)
                norm_transform_f2 = self.BN1(transform_f2)

                # GLU 
                transform_f2 = (glu(norm_transform_f2, self.feature_dims) +transform_f1) * np.sqrt(0.5)

                transform_f3 = self.feature_transform_linear3(transform_f2)
                norm_transform_f3 = self.BN1(transform_f3)

                transform_f4 = self.feature_transform_linear4(norm_transform_f3)
                norm_transform_f4 = self.BN1(transform_f4)

                # GLU
                transform_f4 = (glu(norm_transform_f4, self.feature_dims) + transform_f3) * np.sqrt(0.5)
                
                # 使用顺序注意力机制
                if self.use_sequential_attention:
                    # 使用注意力层进行变换
                    norm_transform_f4, _ = self.attention_layer(norm_transform_f4.unsqueeze(0), norm_transform_f4.unsqueeze(0), norm_transform_f4.unsqueeze(0))
                    norm_transform_f4 = norm_transform_f4.squeeze(0)
                
                decision_out = torch.nn.ReLU(inplace=True)(transform_f4[:, :self.output_dim])
                # Decision aggregation
                output_aggregated  = torch.add(decision_out, output_aggregated)
                scale_agg = torch.sum(decision_out, axis=1, keepdim=True) / (self.num_decision_steps - 1)
                aggregated_mask_values  = torch.add( aggregated_mask_values, mask_values * scale_agg)

                features_for_coef = (transform_f4[:, self.output_dim:]
                               
                if ni<(self.num_decision_steps-1):

                    mask_linear_layer = self.mask_linear_layer(features_for_coef)
                    mask_linear_norm = self.BN2(mask_linear_layer)
                    mask_linear_norm  = torch.mul(mask_linear_norm, complemantary_aggregated_mask_values)
                    mask_values = sparsemax(mask_linear_norm)
                    
                    complemantary_aggregated_mask_values = torch.mul(complemantary_aggregated_mask_values,self.relaxation_factor - mask_values)
                    total_entropy = torch.add(total_entropy,torch.mean(torch.sum(-mask_values * torch.log(mask_values + self.epsilon),axis=1)) / (self.num_decision_steps - 1))
                    masked_features = torch.mul(mask_values , features)
                    
            # 在此处调用 Linear 类的实例进行嵌入
            embedded_features = self.embedding_layer(masked_features)
           
        return  output_aggregated, total_entropy, masked_features, embedded_features



class StyleEncoder(nn.Module):
    def __init__(self, num_hyperparameters, em_size):
        super().__init__()
        self.em_size = em_size
        self.embedding = nn.Linear(num_hyperparameters, self.em_size)

    def forward(self, hyperparameters):  # B x num_hps
        return self.embedding(hyperparameters)


class StyleEmbEncoder(nn.Module):
    def __init__(self, num_hyperparameters, em_size, num_embeddings=100):
        super().__init__()
        assert num_hyperparameters == 1
        self.em_size = em_size
        self.embedding = nn.Embedding(num_embeddings, self.em_size)

    def forward(self, hyperparameters):  # B x num_hps
        return self.embedding(hyperparameters.squeeze(1))



class EmbeddingEncoder(nn.Module):
    def __init__(self, num_features, em_size, num_embs=100):
        super().__init__()
        self.num_embs = num_embs
        self.embeddings = nn.Embedding(num_embs * num_features, em_size, max_norm=True)
        self.init_weights(.1)
        self.min_max = (-2,+2)

    @property
    def width(self):
        return self.min_max[1] - self.min_max[0]

    def init_weights(self, initrange):
        self.embeddings.weight.data.uniform_(-initrange, initrange)

    def discretize(self, x):
        split_size = self.width / self.num_embs
        return (x - self.min_max[0] // split_size).int().clamp(0, self.num_embs - 1)

    def forward(self, x):  # T x B x num_features
        x_idxs = self.discretize(x)
        x_idxs += torch.arange(x.shape[-1], device=x.device).view(1, 1, -1) * self.num_embs
        # print(x_idxs,self.embeddings.weight.shape)
        return self.embeddings(x_idxs).mean(-2)


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, x):
        return (x-self.mean)/self.std


def get_normalized_uniform_encoder(encoder_creator):
    """
    This can be used to wrap an encoder that is fed uniform samples in [0,1] and normalizes these to 0 mean and 1 std.
    For example, it can be used as `encoder_creator = get_normalized_uniform_encoder(encoders.Linear)`, now this can
    be initialized with `encoder_creator(feature_dim, in_dim)`.
    :param encoder:
    :return:
    """
    return lambda in_dim, out_dim: nn.Sequential(Normalize(.5, math.sqrt(1/12)), encoder_creator(in_dim, out_dim))


def get_normalized_encoder(encoder_creator, data_std):
    return lambda in_dim, out_dim: nn.Sequential(Normalize(0., data_std), encoder_creator(in_dim, out_dim))


class ZNormalize(nn.Module):
    def forward(self, x):
        return (x-x.mean(-1,keepdim=True))/x.std(-1,keepdim=True)


class AppendEmbeddingEncoder(nn.Module):
    def __init__(self, base_encoder, num_features, emsize):
        super().__init__()
        self.num_features = num_features
        self.base_encoder = base_encoder
        self.emb = nn.Parameter(torch.zeros(emsize))

    def forward(self, x):
        if (x[-1] == 1.).all():
            append_embedding = True
        else:
            assert (x[-1] == 0.).all(), "You need to specify as last position whether to append embedding. " \
                                        "If you don't want this behavior, please use the wrapped encoder instead."
            append_embedding = False
        x = x[:-1]
        encoded_x = self.base_encoder(x)
        if append_embedding:
            encoded_x = torch.cat([encoded_x, self.emb[None, None, :].repeat(1, encoded_x.shape[1], 1)], 0)
        return encoded_x

def get_append_embedding_encoder(encoder_creator):
    return lambda num_features, emsize: AppendEmbeddingEncoder(encoder_creator(num_features, emsize), num_features, emsize)


class VariableNumFeaturesEncoder(nn.Module):
    def __init__(self, base_encoder, num_features):
        super().__init__()
        self.base_encoder = base_encoder
        self.num_features = num_features

    def forward(self, x):
        x = x * (self.num_features/x.shape[-1])
        x = torch.cat((x, torch.zeros(*x.shape[:-1], self.num_features - x.shape[-1], device=x.device)), -1)
        return self.base_encoder(x)


def get_variable_num_features_encoder(encoder_creator):
    return lambda num_features, emsize: VariableNumFeaturesEncoder(encoder_creator(num_features, emsize), num_features)

class NoMeanEncoder(nn.Module):
    """
    This can be useful for any prior that is translation invariant in x or y.
    A standard GP for example is translation invariant in x.
    That is, GP(x_test+const,x_train+const,y_train) = GP(x_test,x_train,y_train).
    """
    def __init__(self, base_encoder):
        super().__init__()
        self.base_encoder = base_encoder

    def forward(self, x):
        return self.base_encoder(x - x.mean(0, keepdim=True))


def get_no_mean_encoder(encoder_creator):
    return lambda num_features, emsize: NoMeanEncoder(encoder_creator(num_features, emsize))

Linear = nn.Linear
MLP = lambda num_features, emsize: nn.Sequential(nn.Linear(num_features+1,emsize*2),
                                                 nn.ReLU(),
                                                 nn.Linear(emsize*2,emsize))

class NanHandlingEncoder(nn.Module):
    def __init__(self, num_features, emsize, keep_nans=True):
        super().__init__()
        self.num_features = 2 * num_features if keep_nans else num_features
        self.emsize = emsize
        self.keep_nans = keep_nans
        self.layer = nn.Linear(self.num_features, self.emsize)

    def forward(self, x):
        if self.keep_nans:
            x = torch.cat([torch.nan_to_num(x, nan=0.0), normalize_data(torch.isnan(x) * -1
                                                          + torch.logical_and(torch.isinf(x), torch.sign(x) == 1) * 1
                                                          + torch.logical_and(torch.isinf(x), torch.sign(x) == -1) * 2
                                                          )], -1)
        else:
            x = torch.nan_to_num(x, nan=0.0)
        return self.layer(x)



class Conv(nn.Module):
    def __init__(self, input_size, emsize):
        super().__init__()
        self.convs = torch.nn.ModuleList([nn.Conv2d(64 if i else 1, 64, 3) for i in range(5)])
        self.linear = nn.Linear(64,emsize)

    def forward(self, x):
        size = math.isqrt(x.shape[-1])
        assert size*size == x.shape[-1]
        x = x.reshape(*x.shape[:-1], 1, size, size)
        for conv in self.convs:
            if x.shape[-1] < 4:
                break
            x = conv(x)
            x.relu_()
        x = nn.AdaptiveAvgPool2d((1,1))(x).squeeze(-1).squeeze(-1)
        return self.linear(x)


class CanEmb(nn.Embedding):
    def __init__(self, num_features, num_embeddings: int, embedding_dim: int, *args, **kwargs):
        assert embedding_dim % num_features == 0
        embedding_dim = embedding_dim // num_features
        super().__init__(num_embeddings, embedding_dim, *args, **kwargs)

    def forward(self, x):
        lx = x.long()
        assert (lx == x).all(), "CanEmb only works with tensors of whole numbers"
        x = super().forward(lx)
        return x.view(*x.shape[:-2], -1)


def get_Canonical(num_classes):
    return lambda num_features, emsize: CanEmb(num_features, num_classes, emsize)


def get_Embedding(num_embs_per_feature=100):
    return lambda num_features, emsize: EmbeddingEncoder(num_features, emsize, num_embs=num_embs_per_feature)