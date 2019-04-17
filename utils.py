import cvxpy as cp
import numpy as np

def calc_mu_sigma(A,loss_list,mask=None):
    """
    documentation to come :/
    
    Solving: 
    See section 4.3 for what this is solving
    sigma is the sigma squared in that section
    
    
    I think this could be made into one loop if you just took the sigma loop and 
    used the output of the final objective to set the value of mu. leaving for now
    """
    
    mu = cp.Variable((1,A.shape[1]))
    
    mu.value = np.ones((1,A.shape[1]),dtype=np.float)

    #these need to be full arrays because of how cvxpy deals with broadcasting
    # this results in array mu_arr that is like many vstacks of mu
    # for A.shape = (m,n) we use 
    # mu.shape = (1,n) to do the outer product properly
    
    mu_arr = np.ones([A.shape[0],1]) @ mu
    
    #Calc mu
    #use constraints to prevent the solver from zeroing out already 
    #found mu values. This is a problem if there are multiple data types in
    #the matrix.
    
    constraints = [] 
    for columns, loss_fxn in loss_list:
        obj = cp.Minimize(loss_fxn(A,mu_arr,columns))
        prob =cp.Problem(obj,constraints=constraints)
        prob.solve()
        constraints.append(mu[:,columns]==mu[:,columns].value)

    #calc sigma
    sigma = np.zeros(A.shape[1])
    for i in range(len(loss_list)):
        for j in loss_list[i][0]:
            sigma[j] = loss_list[i][1](A,mu_arr,np.array([j])).value
    if mask is None:
        nj = A.shape[0]
    else:
        nj = mask.sum(axis=0)
    sigma /= (nj-1)
    
    sigma_arr = np.ones([A.shape[0],1]) @ sigma[None,:]
    return mu_arr.value,sigma_arr

def gen_random_missing_mask(A,n_missing,return_indices = False):
    """
    generates a mask + (optional) indices of the missing data
    """
    missing_idx =np.random.choice(np.arange(A.shape[0]),size=(n_missing,1),replace=False)
    missing_idx =np.hstack([missing_idx,np.random.randint(0,A.shape[1],size=(n_missing,1))])
    
    mask = np.ones_like(A,dtype=np.bool)
    mask[missing_idx[:,0],missing_idx[:,1]]=False
    if return_indices:
        return mask, missing_idx
    else:
        return mask
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())
    