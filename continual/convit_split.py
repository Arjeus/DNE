# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#

'''These modules are adapted from those of timm, see
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
'''

import copy
import math
from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

import continual.utils as cutils
import continual.split_blocks as split_blocks


class BatchEnsemble(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()

        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.out_features, self.in_features = out_features, in_features
        self.bias = bias

        self.r = nn.Parameter(torch.randn(self.out_features))
        self.s = nn.Parameter(torch.randn(self.in_features))

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls, self.in_features, self.out_features, self.bias)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo))

        result.linear.weight = self.linear.weight
        return result

    def reset_parameters(self):
        device = self.linear.weight.device
        self.r = nn.Parameter(torch.randn(self.out_features).to(device))
        self.s = nn.Parameter(torch.randn(self.in_features).to(device))

        if self.bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.linear.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.linear.bias, -bound, bound)

    def forward(self, x):
        w = torch.outer(self.r, self.s)
        w = w * self.linear.weight
        return F.linear(x, w, self.linear.bias)


class split_Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., base_head_dim=32,
                 fc=split_blocks.split_Linear, split_block_config={}, mlp_mode='All'):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        head_num = in_features // base_head_dim
        out_head_dim = out_features // head_num
        hidden_head_dim = hidden_features // head_num

        first_config = copy.deepcopy(split_block_config)
        if mlp_mode == 'Second':
            first_config['simple_proj'] = True
            first_config['stack'] = False
        self.fc1 = fc(in_features, hidden_features, head_dim=base_head_dim, out_head_dim=hidden_head_dim,
                      **first_config)
        self.act = act_layer()

        second_config = copy.deepcopy(split_block_config)
        if mlp_mode == 'First':
            second_config['simple_proj'] = True
            second_config['stack'] = False
        self.fc2 = fc(hidden_features, out_features, head_dim=hidden_head_dim, out_head_dim=out_head_dim,
                      **second_config)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def freeze_split_old(self):
        self.fc1.freeze_split_old()
        self.fc2.freeze_split_old()

    def expand(self, extra_in_features, extra_hidden_features=None, extra_out_features=None):
        extra_out_features = extra_out_features or extra_in_features
        extra_hidden_features = extra_hidden_features or extra_in_features
        self.fc1.expand(extra_in_features, extra_hidden_features)
        self.fc2.expand(extra_hidden_features, extra_out_features)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, BatchEnsemble):
            trunc_normal_(m.linear.weight, std=.02)
            if isinstance(m.linear, nn.Linear) and m.linear.bias is not None:
                nn.init.constant_(m.linear.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # print("Now in MLP")
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x, debug=True)
        x = self.drop(x)
        return x


class split_GPSA(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 locality_strength=1., use_local_init=True, fc=None,
                 split_block_config={}):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.split = split_block_config['split']

        self.q = split_blocks.split_Linear(dim, dim, bias=qkv_bias, **split_block_config)
        self.k = split_blocks.split_Linear(dim, dim, bias=qkv_bias, **split_block_config)
        self.v = split_blocks.split_Linear(dim, dim, bias=qkv_bias, **split_block_config)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = split_blocks.split_Linear(dim, dim, **split_block_config)
        self.pos_proj = split_blocks.split_Linear(3, num_heads, simple_proj=True)
        self.proj_drop = nn.Dropout(proj_drop)
        self.locality_strength = locality_strength
        self.gating_param_list = nn.ParameterList([nn.Parameter(torch.ones(self.num_heads))])
        self.apply(self._init_weights)
        if use_local_init:
            self.v.local_init_latest_v(dim)
            self.pos_proj.local_init_latest_proj(self.num_heads, locality_strength=locality_strength)
            #self.local_init(locality_strength=locality_strength)

    def freeze_split_old(self):
        self.q.freeze_split_old()
        self.k.freeze_split_old()
        self.v.freeze_split_old()
        self.proj.freeze_split_old()
        self.pos_proj.freeze_split_old()
        for p in self.gating_param_list[:-1]:
            p.requires_grad = False

    def expand(self, extra_dim, extra_heads):
        org_head_dim = self.dim // self.num_heads
        assert extra_heads * org_head_dim == extra_dim, "Extra head dim must be the same as original head dim!"
        self.num_heads += extra_heads
        self.dim += extra_dim
        self.q.expand(extra_dim, extra_dim)
        self.k.expand(extra_dim, extra_dim)
        self.v.expand(extra_dim, extra_dim)
        self.v.local_init_latest_v(extra_dim)

        self.proj.expand(extra_dim, extra_dim)
        #Not sure how to init extra parts of self.proj, just random init, haha.
        #self.proj.local_init_latest_proj(extra_heads, locality_strength=self.locality_strength)
        self.pos_proj.expand(0, extra_heads)
        if not self.split:
            assert len(self.gating_param_list) == 1, "If split is enabled, there can only be one block"
            new_gating_param = nn.Parameter(torch.ones(self.num_heads).to(self.q.device))
            new_gating_param.data[:-extra_heads] = self.gating_param_list[-1].data
            self.gating_param_list[-1] = new_gating_param
            return

        for p in self.gating_param_list.parameters():
            p.requires_grad = False
        self.gating_param_list.append(nn.Parameter(torch.ones(extra_heads).to(self.q.device)))


    @property
    def gating_param(self):
        return torch.cat(list(self.gating_param_list), dim=-1)

    def reset_parameters(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B, N, C = x.shape
        if not hasattr(self, 'rel_indices') or self.rel_indices.size(1)!=N:
            self.get_rel_indices(N)

        # print("Now in SAB")

        attn = self.get_attention(x)
        # print("Disp attn in SAB")
        # print(attn[0,0])
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn, v

    def get_attention(self, x):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        pos_score = self.rel_indices.expand(B, -1, -1,-1)
        pos_score = self.pos_proj(pos_score).permute(0,3,1,2)
        patch_score = (q @ k.transpose(-2, -1)) * self.scale
        patch_score = patch_score.softmax(dim=-1)
        pos_score = pos_score.softmax(dim=-1)

        gating = self.gating_param.view(1,-1,1,1)
        attn = (1.-torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
        attn /= attn.sum(dim=-1).unsqueeze(-1)
        attn = self.attn_drop(attn)
        return attn

    def get_attention_map(self, x, return_map = False):

        attn_map = self.get_attention(x).mean(0) # average over batch
        distances = self.rel_indices.squeeze()[:,:,-1]**.5
        dist = torch.einsum('nm,hnm->h', (distances, attn_map))
        dist /= distances.size(0)
        if return_map:
            return dist, attn_map
        else:
            return dist

    def local_init(self, locality_strength=1.):
        self.v.weight.data.copy_(torch.eye(self.dim))
        locality_distance = 1 #max(1,1/locality_strength**.5)

        kernel_size = int(self.num_heads**.5)
        center = (kernel_size-1)/2 if kernel_size%2==0 else kernel_size//2
        for h1 in range(kernel_size):
            for h2 in range(kernel_size):
                position = h1+kernel_size*h2
                self.pos_proj.weight.data[position,2] = -1
                self.pos_proj.weight.data[position,1] = 2*(h1-center)*locality_distance
                self.pos_proj.weight.data[position,0] = 2*(h2-center)*locality_distance
        self.pos_proj.weight.data *= locality_strength

    def get_rel_indices(self, num_patches):
        img_size = int(num_patches**.5)
        rel_indices   = torch.zeros(1, num_patches, num_patches, 3)
        ind = torch.arange(img_size).view(1,-1) - torch.arange(img_size).view(-1, 1)
        indx = ind.repeat(img_size,img_size)
        indy = ind.repeat_interleave(img_size,dim=0).repeat_interleave(img_size,dim=1)
        indd = indx**2 + indy**2
        rel_indices[:,:,:,2] = indd.unsqueeze(0)
        rel_indices[:,:,:,1] = indy.unsqueeze(0)
        rel_indices[:,:,:,0] = indx.unsqueeze(0)
        device = self.q.device
        self.rel_indices = rel_indices.to(device)


class split_MHSA(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., fc=None):
        raise NotImplementedError("split MHSA not implemented")
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.apply(self._init_weights)

    def reset_parameters(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_attention_map(self, x, return_map = False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn_map = (q @ k.transpose(-2, -1)) * self.scale
        attn_map = attn_map.softmax(dim=-1).mean(0)

        img_size = int(N**.5)
        ind = torch.arange(img_size).view(1,-1) - torch.arange(img_size).view(-1, 1)
        indx = ind.repeat(img_size,img_size)
        indy = ind.repeat_interleave(img_size,dim=0).repeat_interleave(img_size,dim=1)
        indd = indx**2 + indy**2
        distances = indd**.5
        distances = distances.to('cuda')

        dist = torch.einsum('nm,hnm->h', (distances, attn_map))
        dist /= N

        if return_map:
            return dist, attn_map
        else:
            return dist

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class ScaleNorm(nn.Module):
    """See
    https://github.com/lucidrains/reformer-pytorch/blob/a751fe2eb939dcdd81b736b2f67e745dc8472a09/reformer_pytorch/reformer_pytorch.py#L143
    """
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1))
        self.eps = eps

    def forward(self, x):
        n = torch.norm(x, dim=-1, keepdim=True).clamp(min=self.eps)
        return x / n * self.g


class split_Block(nn.Module):

    def __init__(self, dim, num_heads,  mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=split_blocks.split_LayerNorm, attention_type=split_GPSA,
                 fc=split_blocks.split_Linear, split_block_config={}, dense_mode=['attn', 'mlp'], **kwargs):
        super().__init__()
        self.norm1 = norm_layer(dim)
        attn_split_block_config = copy.deepcopy(split_block_config)
        if 'attn' not in dense_mode:
            attn_split_block_config['simple_proj'] = True
            attn_split_block_config['stack'] = False
        self.attn = attention_type(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                                   proj_drop=drop, fc=fc, split_block_config=attn_split_block_config, **kwargs)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp_ratio = mlp_ratio
        mlp_split_block_config = copy.deepcopy(split_block_config)
        mlp_mode = 'None'
        if 'mlp' in dense_mode:
            mlp_mode = 'All'
        elif 'mlp_first' in dense_mode:
            mlp_mode = 'First'
        elif 'mlp_second' in dense_mode:
            mlp_mode = 'Second'
        else:
            mlp_mode = 'None'
        if mlp_mode == 'None':
            mlp_split_block_config['simple_proj'] = True
            mlp_split_block_config['stack'] = False
        self.mlp = split_Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, fc=fc,
                             split_block_config=mlp_split_block_config, mlp_mode=mlp_mode)

    def freeze_split_old(self):
        self.norm1.freeze_split_old()
        self.norm2.freeze_split_old()
        self.attn.freeze_split_old()
        self.mlp.freeze_split_old()

    def expand(self, extra_dim, extra_heads):
        self.norm1.expand(extra_dim)
        self.norm2.expand(extra_dim)
        self.attn.expand(extra_dim, extra_heads)
        extra_mlp_hidden_dim = int(extra_dim * self.mlp_ratio)
        self.mlp.expand(extra_dim, extra_hidden_features=extra_mlp_hidden_dim)

    def reset_parameters(self):
        self.norm1.reset_parameters()
        self.norm2.reset_parameters()
        self.attn.reset_parameters()
        self.mlp.apply(self.mlp._init_weights)

    def forward(self, x, mask_heads=None, task_index=1, attn_mask=None):
        if isinstance(self.attn, split_ClassAttention) or isinstance(self.attn, split_JointCA):  # Like in CaiT
            cls_token = x[:, :task_index]

            xx = self.norm1(x)
            xx, attn, v = self.attn(
                xx,
                mask_heads=mask_heads,
                nb=task_index,
                attn_mask=attn_mask
            )

            cls_token = self.drop_path(xx[:, :task_index]) + cls_token
            cls_token = self.drop_path(self.mlp(self.norm2(cls_token))) + cls_token

            return cls_token, attn, v

        xx = self.norm1(x)
        xx, attn, v = self.attn(xx)

        x = self.drop_path(xx) + x
        x = self.drop_path(self.mlp(self.norm2(x))) + x

        return x, attn, v


class split_ClassAttention(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications to do CA
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 fc=split_blocks.split_Linear, split_block_config={}):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = fc(dim, dim, bias=qkv_bias, **split_block_config)
        self.k = fc(dim, dim, bias=qkv_bias, **split_block_config)
        self.v = fc(dim, dim, bias=qkv_bias, **split_block_config)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = fc(dim, dim, **split_block_config)
        self.proj_drop = nn.Dropout(proj_drop)

        self.apply(self._init_weights)

    def reset_parameters(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def freeze_split_old(self):
        self.q.freeze_split_old()
        self.k.freeze_split_old()
        self.v.freeze_split_old()
        self.proj.freeze_split_old()

    def expand(self, extra_dim, extra_heads):
        org_head_dim = self.dim // self.num_heads
        assert extra_heads * org_head_dim == extra_dim, "Extra head dim must be the same as original head dim!"
        self.num_heads += extra_heads
        self.dim += extra_dim
        self.q.expand(extra_dim, extra_dim)
        self.k.expand(extra_dim, extra_dim)
        self.v.expand(extra_dim, extra_dim)
        self.proj.expand(extra_dim, extra_dim)

    def forward(self, x, mask_heads=None, **kwargs):
        # print("Now in TAB")
        B, N, C = x.shape
        q = self.q(x[:,0]).unsqueeze(1).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        q = q * self.scale
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        # print("Disp attn in TAB")
        # print(attn[0, 0])
        attn = self.attn_drop(attn)

        if mask_heads is not None:
            mask_heads = mask_heads.expand(B, self.num_heads, -1, N)
            attn = attn * mask_heads

        x_cls = (attn @ v).transpose(1, 2).reshape(B, 1, C)
        x_cls = self.proj(x_cls)
        x_cls = self.proj_drop(x_cls)

        return x_cls, attn, v


class split_JointCA(nn.Module):
    """Forward all task tokens together.

    It uses a masked attention so that task tokens don't interact between them.
    It should have the same results as independent forward per task token but being
    much faster.

    HOWEVER, it works a bit worse (like ~2pts less in 'all top-1' CIFAR100 50 steps).
    So if anyone knows why, please tell me!
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., fc=nn.Linear):
        raise NotImplementedError("split JointCA not implemented")
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = fc(dim, dim, bias=qkv_bias)
        self.k = fc(dim, dim, bias=qkv_bias)
        self.v = fc(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = fc(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.apply(self._init_weights)

    def reset_parameters(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @lru_cache(maxsize=1)
    def get_attention_mask(self, attn_shape, nb_task_tokens):
        """Mask so that task tokens don't interact together.

        Given two task tokens (t1, t2) and three patch tokens (p1, p2, p3), the
        attention matrix is:

        t1-t1 t1-t2 t1-p1 t1-p2 t1-p3
        t2-t1 t2-t2 t2-p1 t2-p2 t2-p3

        So that the mask (True values are deleted) should be:

        False True False False False
        True False False False False
        """
        mask = torch.zeros(attn_shape, dtype=torch.bool)
        for i in range(nb_task_tokens):
            mask[:, i, :i] = True
            mask[:, i, i+1:nb_task_tokens] = True
        return mask

    def forward(self, x, attn_mask=False, nb_task_tokens=1, **kwargs):
        B, N, C = x.shape
        q = self.q(x[:,:nb_task_tokens]).reshape(B, nb_task_tokens, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        q = q * self.scale
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1))
        if attn_mask:
            mask = self.get_attention_mask(attn.shape, nb_task_tokens)
            attn[mask] = -float('inf')
        attn = attn.softmax(dim=-1)

        attn = self.attn_drop(attn)

        x_cls = (attn @ v).transpose(1, 2).reshape(B, nb_task_tokens, C)
        x_cls = self.proj(x_cls)
        x_cls = self.proj_drop(x_cls)

        return x_cls, attn, v


class split_PatchEmbed(nn.Module):
    """ Image to Patch Embedding, from timm
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = split_blocks.split_Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, simple_proj=True)
        self.apply(self._init_weights)

    def freeze_split_old(self):
        self.proj.freeze_split_old()

    def expand(self, extra_embd_size):
        self.proj.expand(in_channels=0, out_channels=extra_embd_size)

    def reset_parameters(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B, C, H, W = x.shape
        #assert H == self.img_size[0] and W == self.img_size[1], \
        #    f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x



class HybridEmbed(nn.Module):
    """ CNN Feature Map Embedding, from timm
    """
    def __init__(self, backbone, img_size=224, feature_size=None, in_chans=3, embed_dim=768):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))[-1]
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            feature_dim = self.backbone.feature_info.channels()[-1]
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.Linear(feature_dim, embed_dim)
        self.apply(self._init_weights)

    def reset_parameters(self):
        self.apply(self._init_weights)

    def forward(self, x):
        x = self.backbone(x)[-1]
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class ConVit_Split(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer='layer',
                 local_up_to_layer=3, locality_strength=1., use_pos_embed=True,
                 class_attention=False, ca_type='base', split_block_config={}, dense_mode=['attn', 'mlp']
        ):
        super().__init__()
        self.num_classes_list = [num_classes]
        self.num_heads_list = [num_heads]
        self.embed_dim_list = [embed_dim]
        self.local_up_to_layer = local_up_to_layer
        self.locality_strength = locality_strength
        self.use_pos_embed = use_pos_embed

        if norm_layer == 'layer':
            norm_layer = split_blocks.split_LayerNorm
            #norm_layer = nn.LayerNorm
        elif norm_layer == 'scale':
            raise NotImplementedError(f'Split scale normalization not implemented')
        else:
            raise NotImplementedError(f'Unknown normalization {norm_layer}')

        if hybrid_backbone is not None:
            raise NotImplementedError(f'Split HybrideEmbd not implemented')
        else:
            self.patch_embed = split_PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.num_patches = num_patches

        self.cls_token_list = nn.ParameterList([nn.Parameter(torch.zeros(1, 1, embed_dim))])
        self.pos_drop = nn.Dropout(p=drop_rate)

        if self.use_pos_embed:
            self.pos_embed_list = nn.ParameterList([nn.Parameter(torch.zeros(1, num_patches, embed_dim))])
            trunc_normal_(self.pos_embed_list[-1], std=.02)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        blocks = []

        if ca_type == 'base':
            ca_block = split_ClassAttention
        elif ca_type == 'jointca':
            ca_block = split_JointCA
        else:
            raise ValueError(f'Unknown CA type {ca_type}')

        for layer_index in range(depth):
            if layer_index < local_up_to_layer:
                # Convit
                block = split_Block(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[layer_index], norm_layer=norm_layer,
                    attention_type=split_GPSA, locality_strength=locality_strength,
                    split_block_config=split_block_config, dense_mode=dense_mode
                )
            elif not class_attention:
                # Convit
                block = split_Block(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[layer_index], norm_layer=norm_layer,
                    attention_type=split_MHSA,
                    split_block_config=split_block_config, dense_mode=dense_mode
                )
            else:
                # CaiT
                block = split_Block(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[layer_index], norm_layer=norm_layer,
                    attention_type=ca_block,
                    split_block_config=split_block_config, dense_mode=dense_mode
                )

            blocks.append(block)

        self.blocks = nn.ModuleList(blocks)
        self.norm = norm_layer(embed_dim)
        self.use_class_attention = class_attention

        # Classifier head
        self.feature_info = [dict(num_chs=embed_dim, reduction=0, module='head')]
        self.head = split_blocks.split_Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.cls_token_list[-1], std=.02)
        self.head.apply(self._init_weights)

    def expand(self, extra_classes, extra_dim, extra_heads):
        self.num_classes_list.append(extra_classes)
        self.num_heads_list.append(extra_heads)
        self.embed_dim_list.append(extra_dim)
        self.patch_embed.expand(extra_dim)

        for p in self.cls_token_list.parameters():
            p.requires_grad = False
        self.cls_token_list.append(nn.Parameter(torch.zeros(1, 1, extra_dim).to(self.patch_embed.proj.device)))
        trunc_normal_(self.cls_token_list[-1], std=.02)

        if self.use_pos_embed:
            for p in self.pos_embed_list.parameters():
                p.requires_grad = False
            self.pos_embed_list.append(nn.Parameter(torch.zeros(1, self.num_patches, extra_dim).to(self.patch_embed.proj.device)))
            trunc_normal_(self.pos_embed_list[-1], std=.02)
        for block in self.blocks:
            block.expand(extra_dim, extra_heads)

        self.norm.expand(extra_dim)
        if self.num_classes > 0:
            self.head.expand(extra_dim, extra_classes)


    @property
    def cls_token(self):
        return torch.cat(list(self.cls_token_list), dim=-1)

    @property
    def pos_embed(self):
        return torch.cat(list(self.pos_embed_list), dim=-1)

    @property
    def num_classes(self):
        return sum(self.num_classes_list)

    @property
    def num_heads(self):
        return sum(self.num_heads_list)

    @property
    def embed_dim(self):
        return sum(self.embed_dim_list)

    @property
    def num_features(self):
        return self.embed_dim

    @property
    def final_dim(self):
        return self.embed_dim # num_features for consistency with other models

    def freeze(self, names):
        for name in names:
            if name == 'all':
                return cutils.freeze_parameters(self)
            elif name == 'old_heads':
                self.head.freeze(name)
            elif name == 'backbone':
                cutils.freeze_parameters(self.blocks)
                cutils.freeze_parameters(self.patch_embed)
                cutils.freeze_parameters(self.pos_embed)
                cutils.freeze_parameters(self.norm)
            else:
                raise NotImplementedError(f'Unknown name={name}.')

    def reset_classifier(self):
        self.head.apply(self._init_weights)

    def reset_parameters(self):
        for b in self.blocks:
            b.reset_parameters()
        self.norm.reset_parameters()
        self.head.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_internal_losses(self, clf_loss):
        return {}

    def end_finetuning(self):
        pass

    def begin_finetuning(self):
        pass

    def epoch_log(self):
        return {}

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def forward_sa(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        if self.use_pos_embed:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks[:self.local_up_to_layer]:
            x, _ = blk(x)

        return x

    def forward_features(self, x, final_norm=True):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)

        if self.use_pos_embed:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks[:self.local_up_to_layer]:
            x, _, _ = blk(x)

        if self.use_class_attention:
            for blk in self.blocks[self.local_up_to_layer:]:
                cls_tokens, _, _ = blk(torch.cat((cls_tokens, x), dim=1))
        else:
            x = torch.cat((cls_tokens, x), dim=1)
            for blk in self.blocks[self.local_up_to_layer:]:
                x, _ , _ = blk(x)

        if final_norm:
            if self.use_class_attention:
                cls_tokens = self.norm(cls_tokens)
            else:
                x = self.norm(x)

        if self.use_class_attention:
            return cls_tokens[:, 0], None, None
        else:
            return x[:, 0], None, None

    def forward(self, x):
        x = self.forward_features(x)[0]
        x = self.head(x)
        return x