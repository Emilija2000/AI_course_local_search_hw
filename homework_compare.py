import matplotlib.pyplot as plt
import numpy as np

from homework_genetic_alg import init_population, ga_iter
from homework_simulated_annealing import init_an, an_iter
from homework_local_beam_search import init_population_beam, beam_iter

np.random.seed(42)

#parameters 
NUM_ITER = 100
#ga
POPULATION_NUM = 20
EPOCHS = 500
PROB_MUTATIONS = 0.005
#an
M = 30
T = np.linspace(0.3, 0, num=30,endpoint=False)
T = np.repeat(T,M)
#bm
K = 3

if __name__ == '__main__':
    #sim annealing
    iter_num_arr_an = np.zeros(NUM_ITER)
    best_f_arr_an = np.zeros(NUM_ITER)
    for j in range(NUM_ITER):
        #init
        x, best_f = init_an()
        iter_num = 0
        
        # begin cooling
        for k in range(len(T)):
            x,func = an_iter(x,T[k])
            
            # update best
            if (func < best_f):
                best_f = func
                iter_num = k
                
        best_f_arr_an[j] = best_f
        iter_num_arr_an[j] = iter_num
    
    
    #beam search
    iter_num_arr_bm = np.zeros(NUM_ITER)
    best_f_arr_bm = np.zeros(NUM_ITER)
    for j in range(NUM_ITER):
        
        curr, curr_f, best_f, best_x = init_population_beam(K)
        iter_num = 0
    
        for i in range(1,EPOCHS):
            curr,curr_f,ind_best = beam_iter(curr, K)
            
            if(curr_f[ind_best]<best_f):
                best_f = curr_f[ind_best]
                iter_num = i
                
        best_f_arr_bm[j] = best_f
        iter_num_arr_bm[j] = K+ iter_num * K**2
            
        
    #genetic alg
    iter_num_arr_ga = np.zeros(NUM_ITER)
    best_f_arr_ga = np.zeros(NUM_ITER)
    for j in range(NUM_ITER):
        
        population, fit_func = init_population(POPULATION_NUM)
        best_f = np.max(fit_func)
        iter_num = 0
        
        for k in range(0,EPOCHS):                        
            population,fit = ga_iter(population,PROB_MUTATIONS)
            
            best_it = np.max(fit)
            if(best_it>best_f):
                best_f = best_it
                iter_num = k
                
        best_f_arr_ga[j] = -best_f
        iter_num_arr_ga[j] = iter_num * POPULATION_NUM
      
              
    #plots
    fig,ax = plt.subplots(3,2,figsize=(12,18),dpi=80)
    ax[0,0].hist(iter_num_arr_an); ax[0,0].set_title('Broj generisanih kandidata - simulirano kaljenje')
    ax[0,1].hist(best_f_arr_an); ax[0,1].set_title('Najbolje pronadjeno resenje - simulirano kaljenje')
    ax[1,0].hist(iter_num_arr_bm); ax[1,0].set_title('Broj generisanih kandidata - pretraga po snopu')
    ax[1,1].hist(best_f_arr_bm); ax[1,1].set_title('Najbolje pronadjeno resenje - pretraga po snopu')
    ax[2,0].hist(iter_num_arr_ga); ax[2,0].set_title('Broj generisanih kandidata - genetski algoritam')
    ax[2,1].hist(best_f_arr_ga); ax[2,1].set_title('Najbolje pronadjeno resenje - genetski algoritam')
    
    
    columns = ['','Mean', 'STD']
    values = [
        ['Simulirano kaljenje', np.mean(iter_num_arr_an), np.std(iter_num_arr_an)],
        ['Pretraga po snopu', np.mean(iter_num_arr_bm), np.std(iter_num_arr_bm)],
        ['Genetski algoritam', np.mean(iter_num_arr_ga), np.std(iter_num_arr_ga)],
        ]
    plt.figure(figsize=(6,3))
    plt.table(cellText=values, colLabels=columns, loc='center')
    plt.tight_layout()
    plt.axis('off')