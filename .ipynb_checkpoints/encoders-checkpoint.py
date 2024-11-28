import math

import torch
import torch.nn as nn
from tabpfn.utils import normalize_data
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer


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


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        self.device_test_tensor = nn.Parameter(torch.tensor(1.))

    def forward(self, x):# T x B x num_features
        assert self.d_model % x.shape[-1]*2 == 0
        d_per_feature = self.d_model // x.shape[-1]
        pe = torch.zeros(*x.shape, d_per_feature, device=self.device_test_tensor.device)
        #position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        interval_size = 10
        div_term = (1./interval_size) * 2*math.pi*torch.exp(torch.arange(0, d_per_feature, 2, device=self.device_test_tensor.device).float()*math.log(math.sqrt(2)))
        #print(div_term/2/math.pi)
        pe[..., 0::2] = torch.sin(x.unsqueeze(-1) * div_term)
        pe[..., 1::2] = torch.cos(x.unsqueeze(-1) * div_term)
        return self.dropout(pe).view(x.shape[0],x.shape[1],self.d_model)


Positional = lambda _, emsize: _PositionalEncoding(d_model=emsize)

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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
import math
class AttentiveTransformer(nn.Module):
    def __init__(
        self,
        input_dim,
        group_dim,
        virtual_batch_size=128,
        momentum=0.02,
    ):
        """
        Initialize an attentive transformer.

        Parameters
        ----------
        input_dim : int
            Input size
        group_dim : int
            Number of groups for features (hidden dimension)
        virtual_batch_size : int
            Batch size for Ghost Batch Normalization
        momentum : float
            Momentum for batch normalization
        """
        super(AttentiveTransformer, self).__init__()
        # 线性变换层
        self.fc = nn.Linear(input_dim, group_dim, bias=False)
        nn.init.kaiming_uniform_(self.fc.weight, a=math.sqrt(5))  # 初始化权重

        # Ghost Batch Normalization (GBN)
        self.bn = nn.BatchNorm1d(group_dim, momentum=momentum)
    def forward(self, priors, processed_feat):
        # 线性变换
        x = self.fc(processed_feat)  # [batch_size, seq_len, group_dim]

        # 保存原始形状
        original_shape = x.shape  # [1152, 16, 512]

        # 展平 x 以适配 BatchNorm1d
        x = x.reshape(-1, x.size(-1))  # [1152*16, 512]

        # 展平 priors 以匹配展平后的 x
        priors = priors.reshape(-1, priors.size(-1))  # 使用 reshape 而非 view

        # 批量归一化
        x = self.bn(x)

        # 确保形状匹配
        assert priors.shape == x.shape, f"Shape mismatch: priors {priors.shape}, x {x.shape}"

        # 应用注意力
        x = x * priors
        x = F.softmax(x, dim=-1)

        # 恢复原始形状
        x = x.view(original_shape)

        
        return x



class SequentialAttention(nn.Module):
    def __init__(self, input_dim, emsize, output_dim, n_steps=3, gamma=1.3, epsilon=1e-15):
        super(SequentialAttention, self).__init__()
        self.input_dim = input_dim
        self.emsize = emsize
        self.output_dim = output_dim
        self.n_steps = n_steps
        self.gamma = gamma
        self.epsilon = epsilon

        # 初始特征变换层
        self.initial_transform = nn.Linear(input_dim, emsize)

        # 替换为类似原作者实现的注意力变换器
        self.attention_transformers = nn.ModuleList(
            [
                AttentiveTransformer(
                    input_dim=emsize,
                    group_dim=emsize,  # 保持维度一致
                    virtual_batch_size=128,
                    momentum=0.02,
                )
                for _ in range(n_steps)
            ]
        )

        # 特征变换器
        self.feature_transformers = nn.ModuleList(
            [nn.Linear(emsize, emsize) for _ in range(n_steps)]
        )

        # 添加线性层，将 x 投影到 emsize
        self.input_projection = nn.Linear(input_dim, emsize)

    def forward(self, x):
        batch_size = x.shape[0]
        remaining_importance = torch.ones((batch_size, self.emsize), device=x.device)

        M_loss = 0
        attention_output = self.initial_transform(x)

        # 投影输入到 emsize
        projected_x = self.input_projection(x)

        # 用于存储注意力权重
        attention_weights_list = []

        for step in range(self.n_steps):
            # 确保 remaining_importance 和 attention_output 的维度匹配
            if remaining_importance.dim() < attention_output.dim():
                remaining_importance = remaining_importance.unsqueeze(1).expand(-1, attention_output.size(1), -1)

           
            # 计算注意力权重
            attention_weights = self.attention_transformers[step](remaining_importance, attention_output)
            attention_weights_list.append(attention_weights)

            # 更新 remaining_importance
            remaining_importance = self.gamma * remaining_importance - attention_weights

        M_loss /= self.n_steps
        return attention_output, M_loss, attention_weights_list

    
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
class ModelWithAttention(nn.Module):
    def __init__(self, input_dim, emsize, output_dim, n_steps=3, gamma=1.3, epsilon=1e-15):
        super(ModelWithAttention, self).__init__()
        self.sequential_attention = SequentialAttention(
            input_dim=input_dim,
            emsize=emsize,
            output_dim=output_dim,
            n_steps=n_steps,
            gamma=gamma,
            epsilon=epsilon,
        )
        self.linear = Linear(emsize, output_dim)

    def forward(self, x):
        
        
        # 顺序注意力机制处理，得到筛选特征、M_loss 和注意力权重
        filtered_features, M_loss, attention_weights_list = self.sequential_attention(x)

        # 应用线性变换
        output = self.linear(filtered_features)
        
        if not isinstance(filtered_features, torch.Tensor):
            raise TypeError(f"Filtered features is not a tensor, got {type(filtered_features)}")

        # 返回最终编码结果
        return output  # 只返回 output

    '''
class ModelWithAttention(nn.Module):
    def __init__(self, input_dim, emsize, output_dim, n_steps=3, gamma=1.3, epsilon=1e-15):
        super(ModelWithAttention, self).__init__()
        self.sequential_attention = SequentialAttention(
            input_dim=input_dim,
            emsize=emsize,
            output_dim=output_dim,
            n_steps=n_steps,
            gamma=gamma,
            epsilon=epsilon,
        )
        self.linear = Linear(emsize, output_dim)

        # 添加一个属性用于存储注意力权重
        self.attention_weights_list = None

    def forward(self, x):
        # 顺序注意力机制处理，得到筛选特征、M_loss 和注意力权重
        filtered_features, M_loss, attention_weights_list = self.sequential_attention(x)

        # 将注意力权重保存到模型的属性中
        self.attention_weights_list = attention_weights_list

        # 应用线性变换
        output = self.linear(filtered_features)

        return output
    

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
