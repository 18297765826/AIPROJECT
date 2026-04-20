"""
input->embedding
->attention
->add &norm
->FFN
->add &norm
->输出
"""

import torch
import torch.nn.functional as F


Q = torch.randn(1, 3, 4)
print(Q.size(-1))

def attention(query, key, value):
    #获取维度
    d_k = query.size(-1)
    #计算注意力分数
    scores=torch.matmul(query,key.transpose(-2,-1))/torch.sqrt(torch.tensor(d_k))
    weights=F.softmax(scores,dim=-1)
    output=torch.matmul(weights,value)
    return output
