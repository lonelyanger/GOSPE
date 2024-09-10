from math import cos
from numpy import diag
from sympy import dict_merge
import torch
import torch.nn.functional as F
from copy import deepcopy



def node_cost_st(cost_s, cost_t, p_s=None, p_t=None, loss_type="L2"):
    """
    Args:
        cost_s (torch.Tensor): [K, T, N, N], where T is the length of the control trajectory & N is the number of the agents. Here data around dim-K is the same.
        cost_t (torch.Tensor): [K, B, M, M], where K is the number of the skills & B is the length of the sampled batch.
        p_s (torch.Tensor): [N, 1]
        p_t (torch.Tensor): [M, 1]
    
    Returns:
        cost_st (torch.Tensor): [K, T, B, N, M]
    """
    dim_n = cost_s.shape[-1]
    dim_k, dim_b, dim_m = cost_t.shape[:3]
    
    with torch.no_grad():
        if loss_type == "L2":
            f1_st = torch.matmul(cost_s ** 2, p_s).repeat(1, 1, dim_m)                            # [K, T, N, 1] -> [K, T, N, M]

            f2_st = torch.matmul(cost_t ** 2, p_t).permute(0, 2, 1).repeat(1, dim_n, 1)        # [K, B, 1, M] -> [K, B, N, M]
            cost_st = f1_st.unsqueeze(1) + f2_st.unsqueeze(0)                                       # [K, T, B, N, M]
        else:
            raise NotImplementedError
    
    return cost_st


def node_cost(cost_s, cost_t, trans,ot_hyperparams, p_s=None, p_t=None, loss_type="L2"):
    """
    Args:
        cost_s (torch.Tensor): [T, N, N]
        cost_t (torch.Tensor): [B, M, M]
        trans (torch.Tensor): [T, B, N, M].
        p_s (torch.Tensor): [N, 1]
        p_t (torch.Tensor): [M, 1]
    
    Returns:
        cost (torch.Tensor): [ T, B, N, M]
    """
    dim_t, dim_n = cost_s.shape[0:2]
    dim_b, dim_m = cost_t.shape[:2]


    cost_st = node_cost_st(cost_s, cost_t, p_s, p_t, loss_type)
    if loss_type == "L2":
        cost=cost_st - 2 * torch.matmul(torch.matmul(cost_s.unsqueeze(1).repeat(1, dim_b, 1, 1), trans),cost_t.unsqueeze(0).repeat(dim_t, 1, 1, 1))
    else:
        raise NotImplementedError
    # if ot_hyperparams["GetTrans"]:
    #     seqidx=ot_hyperparams["nagents"]
    #     cost[:,:,0:seqidx,seqidx:dim_n]=torch.inf
    #     cost[:,:,seqidx:dim_m,0:seqidx]=torch.inf
        # seqidx=ot_hyperparams["nagents"]
        # cost[:,:,0:seqidx,seqidx:dim_n]=cost[:,:,0:seqidx,seqidx:dim_n]*3
        # cost[:,:,seqidx:dim_m,0:seqidx]=cost[:,:,seqidx:dim_m,0:seqidx]*3
   
   
    return cost


def sinkhorn_knopp_iteration(cost,ot_hyperparams, trans0=None, p_s=None, p_t=None,
                             a: torch.Tensor = None, beta: float = 1e-1,
                             error_bound: float = 1e-3, max_iter: int = 30):
    """
    Sinkhorn-Knopp iteration algorithm

    When initial optimal transport "trans0" is not available, the function solves
        min_{trans in Pi(p_s, p_t)} <cost, trans> + beta * <log(trans), trans>

    When initial optimal transport "trans0" is given, the function solves:
        min_{trans in Pi(p_s, p_t)} <cost, trans> + beta * KL(trans || trans0)

    Args:
        cost (torch.Tensor): [T, B, N, M], representing batch of distance between nodes.
        trans0 (torch.Tensor): [T, B, N, M], representing the optimal transport over the episode.
        p_s (torch.Tensor): [N, 1]
        p_t (torch.Tensor): [M, 1]

        a: representing the dual variable
        beta: the weight of entropic regularizer
        error_bound: the error bound to check convergence
        max_iter: the maximum number of iterations

    Returns:
        trans: optimal transport
        a: updated dual variable

    """
    
    dim_t, dim_b,dim_m, dim_n = cost.shape

    if a is None:
        a = torch.ones((cost.shape[-2], 1), device=cost.device) / cost.shape[-2]

    if p_s is None:
        p_s = torch.ones((cost.shape[-2], 1), device=cost.device) / cost.shape[-2]

    if p_t is None:
        p_t = torch.ones((cost.shape[-1], 1), device=cost.device) / cost.shape[-1]

    if trans0 is not None:
        kernel = torch.exp(-cost / beta-1) * trans0
        if ot_hyperparams["GetTrans"]:
            seqidx1=ot_hyperparams["task1_nagt"]
            seqidx2=ot_hyperparams["task2_nagt"]
            kernel[:,:,0:seqidx1,seqidx2:dim_n]=torch.exp(-cost[:,:,0:seqidx1,seqidx2:dim_n] / beta-1000) * trans0[:,:,0:seqidx1,seqidx2:dim_n]
            kernel[:,:,seqidx1:dim_m,0:seqidx2]=torch.exp(-cost[:,:,seqidx1:dim_m,0:seqidx2] / beta-1000) * trans0[:,:,seqidx1:dim_m,0:seqidx2] 
    else:
        kernel = torch.exp(-cost / beta-1)
    
    relative_error = torch.ones(dim_t, dim_b) * float("inf")
    indicater_mat = relative_error > error_bound
    iter_i = 0
    while torch.sum(indicater_mat) >= 1. and iter_i < max_iter:
        # b = torch.div(p_t, torch.matmul(kernel.permute(0, 1, 2, 4, 3), a) + 1e-10)
        b = torch.div(p_t, torch.matmul(kernel.permute(0, 1, 3, 2), a) + 1e-10)
        a_new = torch.div(p_s, torch.matmul(kernel, b) + 1e-10)

        relative_error = torch.div(torch.sum(torch.abs(a_new - a), dim=(-2, -1)), torch.sum(torch.abs(a), dim=(-2, -1)) + 1e-10)
        indicater_mat = relative_error > error_bound
        a = a_new
        iter_i += 1
    # trans = torch.matmul(a, b.permute(0, 1, 2, 4, 3)) * kernel
    trans = torch.matmul(a, b.permute(0, 1, 3, 2)) * kernel

    return trans, a


def gromov_wasserstein_discrepancy(cost_s, cost_t, ot_hyperparams, trans0=None, p_s=None, p_t=None):
    """
    Args:
        cost_s (torch.Tensor): [T, N, N], where T is the length of the control trajectory & N is the number of the agents. 
        cost_t (torch.Tensor): [B, M, M], B is the length of the sampled batch.
        ot_hyperparams (dict): hyper-parameters for the optimal transport algorithm.
        trans0 (torch.Tensor): [T, B, N, M].
        p_s (torch.Tensor): [N, 1]
        p_t (torch.Tensor): [M, 1]

    Returns:
        trans (torch.Tensor): [T, B, N, M].
        d_gw (torch.Tensor): [T, B]. The gromov-wasserstein discrepancy between the episode of graphs & batch of target graphs.
    """
    if ot_hyperparams['opt_trans']:
        dim_t, dim_n = cost_s.shape[0:2]
        dim_b, dim_m = cost_t.shape[:2]

        if p_s is None:
            p_s = (1. / dim_n) * torch.ones(size=[dim_n, 1], device=cost_s.device)
        if p_t is None:
            p_t = (1. / dim_m) * torch.ones(size=[dim_m, 1], device=cost_s.device)
        
        if trans0 is None:
            trans0 = torch.matmul(p_s, p_t.T).unsqueeze(0).unsqueeze(0).repeat(dim_t, dim_b, 1, 1)
    
        relative_error = torch.ones( dim_t, dim_b) * float("inf")
        indicater_mat = relative_error > ot_hyperparams["iter_bound"]
        iter_t = 0
        while torch.sum(indicater_mat) >= 1. and iter_t < ot_hyperparams["outer_iteration"]:
            cost = node_cost(cost_s, cost_t, trans0,ot_hyperparams, p_s, p_t, ot_hyperparams["loss_type"])
            trans, a = sinkhorn_knopp_iteration(
                cost=cost,
                trans0=trans0 if ot_hyperparams["ot_method"] == 'proximal' else None,
                p_s=p_s,
                p_t=p_t,
                error_bound=ot_hyperparams["sk_bound"],
                max_iter=ot_hyperparams["inner_iteration"]
            )
            relative_error = torch.div(torch.sum(torch.abs(trans - trans0), dim=(-2, -1)), torch.sum(torch.abs(trans0), dim=(-2, -1)) + 1e-10)
            indicater_mat = relative_error > ot_hyperparams["iter_bound"]
            trans0 = trans
            iter_t += 1
        ot_hyperparams["GetTrans"]=False
        cost = node_cost(cost_s, cost_t, trans,ot_hyperparams, p_s, p_t, ot_hyperparams["loss_type"])
        d_gw = torch.sum(cost * trans, dim=(-2, -1))
        d_gw=d_gw.to('cuda')
    else:
        trans=0
        d_gw=torch.cdist(cost_s,cost_t)


    return trans, d_gw



def gromov_wasserstein_discrepancy2(cost_s, cost_t, ot_hyperparams, trans0=None, p_s=None, p_t=None):
    """
    Args:
        cost_s (torch.Tensor): [T, N, N], where T is the length of the control trajectory & N is the number of the agents. 
        cost_t (torch.Tensor): [B, M, M], B is the length of the sampled batch.
        ot_hyperparams (dict): hyper-parameters for the optimal transport algorithm.
        trans0 (torch.Tensor): [T, B, N, M].
        p_s (torch.Tensor): [N, 1]
        p_t (torch.Tensor): [M, 1]

    Returns:
        trans (torch.Tensor): [T, B, N, M].
        d_gw (torch.Tensor): [T, B]. The gromov-wasserstein discrepancy between the episode of graphs & batch of target graphs.
    """
    cost_s=cost_s.to('cuda').detach()
    cost_t=cost_t.to('cuda').detach()
    # cost_s=cost_s.detach()
    # cost_t=cost_t.detach()
    dim_t, dim_n = cost_s.shape[0:2]
    dim_b, dim_m = cost_t.shape[:2]


    if p_s is None:
        p_s = (1. / dim_n) * torch.ones(size=[dim_n, 1], device=cost_s.device)
    if p_t is None:
        p_t = (1. / dim_m) * torch.ones(size=[dim_m, 1], device=cost_s.device)
    
    if trans0 is None:
        trans0 = torch.matmul(p_s, p_t.T).unsqueeze(0).unsqueeze(0).repeat(dim_t, dim_b, 1, 1)

    relative_error = torch.ones( dim_t, dim_b) * float("inf")
    indicater_mat = relative_error > ot_hyperparams["iter_bound"]
    iter_t = 0
    while torch.sum(indicater_mat) >= 1. and iter_t < ot_hyperparams["outer_iteration"]:
        cost = node_cost(cost_s, cost_t, trans0,ot_hyperparams, p_s, p_t, ot_hyperparams["loss_type"])
        trans, a = sinkhorn_knopp_iteration(
            cost=cost,
            ot_hyperparams=ot_hyperparams,
            trans0=trans0 if ot_hyperparams["ot_method"] == 'proximal' else None,
            p_s=p_s,
            p_t=p_t,
            error_bound=ot_hyperparams["sk_bound"],
            max_iter=ot_hyperparams["inner_iteration"]
        )
        relative_error = torch.div(torch.sum(torch.abs(trans - trans0), dim=(-2, -1)), torch.sum(torch.abs(trans0), dim=(-2, -1)) + 1e-10)
        indicater_mat = relative_error > ot_hyperparams["iter_bound"]
        trans0 = trans
        iter_t += 1
    ot_hyperparams["GetTrans"]=False
    cost = node_cost(cost_s, cost_t, trans,ot_hyperparams, p_s, p_t, ot_hyperparams["loss_type"])
    d_gw = torch.sum(cost * trans, dim=(-2, -1))
    d_gw=d_gw.to('cuda')
    
    # trans=trans.reshape(-1,dim_n,dim_m)
    # seqidx=ot_hyperparams['nagents']
    # trans[:,0:seqidx,seqidx:dim_m]=0
    # trans[:,seqidx:dim_n,0:seqidx]=0
    # trans=torch.stack([mat/mat.sum() for mat in trans],dim=0)
    # trans=trans.reshape(dim_t, dim_b,dim_n,dim_m)

    # cost = node_cost(cost_s, cost_t, trans,ot_hyperparams, p_s, p_t, ot_hyperparams["loss_type"])
    # d_gw = torch.sum(cost * trans, dim=(-2, -1))


    

    return trans, d_gw


if __name__ == "__main__":
    T, B, N, M = 1, 1, 4,5
    
    cost_s = torch.rand(size=[T, N, N], device="cuda").float()
    # s=torch.rand(size=[N, N], device="cuda").float()
    # s=s+s.T
    # s=s-torch.diag(torch.diag(s))
    # cost_s[0]=s
    # cost_t=torch.rand(size=[T, N, N], device="cuda").float()

    cost_t = torch.rand(size=[B, M, M], device="cuda").float()
    # cost_s=torch.FloatTensor([[[0,   1, 0.5],
    #                            [1,   0,   1],
    #                            [0.5, 1,   0]]])
    # randp=torch.tensor([1,0,2])
    # s=s[randp]
    # s=s[:,randp]
    # cost_t[0]=s
    # print(cost_s[0])
    # print(cost_t[0])

    cost_s = torch.as_tensor(cost_s, device="cuda")
    cost_t = torch.as_tensor(cost_t, device="cuda")

    ot_hyperparams = {
        "ot_method": "proximal",
        "loss_type": "L2",
        "inner_iteration": 100,
        "outer_iteration": 100,
        "iter_bound": 1e-3,
        "sk_bound": 1e-3,
        "opt_trans":True,
        "GetTrans":True,
        "task1_nagt":2,
        "task2_nagt":3,
        "beta":0.01
    }

    # node_cost_st(cost_s, cost_t)
    # cost = node_cost(cost_s, cost_t, trans, mu, mu)
    # sinkhorn_knopp_iteration(cost, trans0=trans)
    trans,d_gw=gromov_wasserstein_discrepancy2(cost_s, cost_t, ot_hyperparams)
    print(d_gw)
    # print(torch.round(trans,decimals=2))
    print(trans.shape)
    print(trans)

