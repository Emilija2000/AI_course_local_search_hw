import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)

def f(x):
    x1 = x[0,:]
    x2 = x[1,:]
    x3 = x[2,:]
    return 4/3*(x1**2 + x2**2 - x1*x2)**0.75 + x3

def neighbours(x,k):
    rand_delta = (np.random.rand(x.shape[0],x.shape[1]*k)-0.5)*0.2
    x = x.repeat(k,axis=1)
    x = x + rand_delta
    x[x>2] = 2
    x[x<0] = 0 
    return x

def init_population_beam(k):
    x = np.random.uniform(0,2,size=(3,1))
    #initial k children
    curr = neighbours(x,k)
    curr_f = f(curr)
    ind = np.argmin(curr_f)
    best = curr_f[ind]
    best_x = curr[:,ind]
    return curr, curr_f, best, best_x

def beam_iter(population,k):
    curr = neighbours(population, k)
    curr_f = f(curr)
    ind = np.argpartition(curr_f,k)
    ind = ind[:k]
    
    curr = curr[:,ind]
    curr_f = curr_f[ind]
    
    ind_best = np.argmin(curr_f)
    return curr,curr_f,ind_best
    
    
if __name__=='__main__':
    k_all = [3,2,30]
    avg_best_f = []
    avg_current_f = []
    num_iter_k = []
    
    stop_crit = 1e-6
    num_repeat=100
    
    
    for j in range(len(k_all)):
        k = k_all[j]
        
        best_f_all = []
        current_f_all = []
        for i in range(num_repeat):
            best_f = []
            current_f = []
            num_iter = []
            
            curr, curr_f, best, best_x = init_population_beam(k)
            
            best_f.append(best)
            current_f.append(np.average(curr_f))
            
            while(best > stop_crit):
                curr,curr_f,ind_best = beam_iter(curr, k)
                
                if(curr_f[ind_best]<best):
                    best = curr_f[ind_best]
                    best_x = curr[:,ind_best]
                    
                best_f.append(best)
                current_f.append(np.average(curr_f))
        
            best_f_all.append(best_f)
            current_f_all.append(current_f)
            
            
        # transform to the same shape
        length = max(map(len, best_f_all))
        best_f_all=np.array([xi+[0]*(length-len(xi)) for xi in best_f_all])
        
        length = max(map(len, current_f_all))
        current_f_all=np.array([xi+[0]*(length-len(xi)) for xi in current_f_all])
        
        avg_best_f.append(np.average(best_f_all,axis=0))
        avg_current_f.append(np.average(current_f_all,axis=0))
        
        num_iter = [k]
        for i in range(len(avg_best_f[-1])-1):
            num_iter.append(num_iter[-1]+k^2)
        num_iter_k.append(num_iter)
    
    
    # plots
    fig,ax = plt.subplots(1,3,figsize = (18,6), dpi=80)
    ax[0].plot(num_iter_k[0],avg_best_f[0],'b')
    ax[0].plot(num_iter_k[0],avg_current_f[0],'r')
    ax[0].set_title('Odabrani broj potomaka k=3')
    ax[0].legend(['najbolje f','trenutno srednje f'])
    ax[0].set_xlabel('broj generisanih kandidata')
    ax[0].set_ylabel('f(x)')
    ax[1].plot(num_iter_k[1],avg_best_f[1],'b')
    ax[1].plot(num_iter_k[1],avg_current_f[1],'r')
    ax[1].set_title('Premalo k=2')
    ax[1].legend(['najbolje f','trenutno srednje f'])
    ax[1].set_xlabel('broj generisanih kandidata')
    ax[1].set_ylabel('f(x)')
    ax[2].plot(num_iter_k[2],avg_best_f[2],'b')
    ax[2].plot(num_iter_k[2],avg_current_f[2],'r')
    ax[2].set_title('Preveliko k=30')
    ax[2].legend(['najbolje f','trenutno srednje f'])
    ax[2].set_xlabel('broj generisanih kandidata')
    ax[2].set_ylabel('f(x)')
    #plt.suptitle('Konvergencija kriterijumske funkcije')
    
