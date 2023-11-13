import torch 

def addCounterFactualIntervation(h,
                                 l,
                                 layers,
                                 i,
                                 k,
                                 alpha):
    """This is the counterfactual intervetion
    
    """

    if l in layers:

        h = torch.clone(h)
        h_l = torch.squeeze(h,dim=0)
        h_i = h_l[i]

        lambda_i = torch.tensor([sum(h_i.T * h_i[ix]) for ix in range(h_i.size(0))])

        h_p = torch.zeros(h.size(2))

        for j in range(*k): h_p[j] = (lambda_i[j] * h_i[j])

        h_i = h_i - (alpha * h_p)

        h[:,i,:] = h_i

    return h