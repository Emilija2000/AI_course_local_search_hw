import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)

#parameters
MIN_RANGE=0
MAX_RANGE=2
BIT_POW_CODE = 20
NUM_ITER = 100

POPULATION_NUM = [4,20,200]
EPOCHS = 10000 #epoch*pop_num 
PROB_MUTATIONS = 0.005


def f(x):
    x1 = x[:,0]
    x2 = x[:,1]
    x3 = x[:,2]
    return 4/3*(x1**2 + x2**2 - x1*x2)**0.75 + x3

def fitness(x):
    x = decode_batch(x)
    return -f(x)

def code(x, b_num=BIT_POW_CODE, min_range=MIN_RANGE, max_range=MAX_RANGE):
    x_int = np.floor(2**b_num * (x-min_range)/(max_range-min_range)).astype('uint')
    
    x_bit_str = ""
    for j in range(3):
        x_bit_str += np.binary_repr(x_int[j], width = b_num+1)[1:]
    
    x_bit = np.fromstring(x_bit_str,dtype='S1').astype('int')
    return x_bit
 
def decode(x, b_num=BIT_POW_CODE, min_range=MIN_RANGE, max_range=MAX_RANGE):
    
    x = np.array([x[:b_num], x[b_num:2*b_num], x[2*b_num:]])
    x = x.dot(2**np.arange(b_num)[::-1])
    
    x = x/2**b_num * (max_range-min_range) + min_range
    return x

def code_batch(x):
    pop_coded = []
    for i in range(x.shape[0]):
        pop_coded.append(code(x[i,:]))
    return pop_coded
    
def decode_batch(x):
    decoded = np.zeros((len(x),3))
    for i in range(len(x)):
        decoded[i,:] = decode(x[i])
    return decoded

def init_population(num, min_range=MIN_RANGE, max_range=MAX_RANGE):
    x = np.random.uniform(min_range,max_range,size=(num,3))
    x_c = code_batch(x)
    fit_func = fitness(x_c)
    return x_c, fit_func

def select(population):
    #softmax as probabilities
    fit = fitness(population)
    prob = np.exp(fit)
    prob = prob/np.sum(prob)
    
    #random sampling
    prob = np.cumsum(prob)
    rnd = np.random.rand(len(fit))
    
    #choose parents
    parents = []
    for i in range(len(fit)):
        ind = np.where(rnd[i]<prob)[0][0]
        parents.append(population[ind])
    return parents

def crossover(parents):
    rnd = np.random.randint(low=0, high=len(parents[0]), size=(len(parents)//2,))
    
    children = []
    for i in range(len(rnd)):
        children.append(np.concatenate((parents[2*i][:rnd[i]],parents[2*i+1][rnd[i]:])))
        children.append(np.concatenate((parents[2*i+1][:rnd[i]],parents[2*i][rnd[i]:])))
    return children

def mutation(population, prob_mutation = PROB_MUTATIONS):
    l = len(population[0])
    for i in range(len(population)):
        rnd = np.random.rand(l)
        rnd = rnd<prob_mutation
        population[i][rnd] = 1-population[i][rnd]
        
    return population

def ga_iter(population,prob_mut=PROB_MUTATIONS):
    parents = select(population)
    population = crossover(parents)
    population = mutation(population,prob_mut)
    return population, fitness(population)

if __name__ == '__main__':
    
    avg_pop_fitness = np.zeros((len(POPULATION_NUM),NUM_ITER,EPOCHS//POPULATION_NUM[0]+1))
    generated_num = np.zeros((len(POPULATION_NUM),EPOCHS//POPULATION_NUM[0]+1))
    
    for i in range(len(POPULATION_NUM)):
        pop_num = POPULATION_NUM[i]
        generated_num[i,:] = np.arange(start=0,step=pop_num,stop=pop_num*EPOCHS//POPULATION_NUM[0]+1)
        
        for j in range(NUM_ITER):
            
            population, fit_func = init_population(pop_num)
            avg_pop_fitness[i,j,0] = np.average(fit_func)
            
            for k in range(1,EPOCHS//pop_num+1):#EPOCHS):
                population,fit = ga_iter(population)
                avg_pop_fitness[i,j,k] = np.average(fit)
                
    
    avg_fitness = np.average(avg_pop_fitness,axis=1)
    
    plt.figure()
    plt.plot(generated_num[0,:EPOCHS//POPULATION_NUM[0]+1],avg_fitness[0,:EPOCHS//POPULATION_NUM[0]+1])
    plt.plot(generated_num[1,:EPOCHS//POPULATION_NUM[1]+1],avg_fitness[1,:EPOCHS//POPULATION_NUM[1]+1])
    plt.plot(generated_num[2,:EPOCHS//POPULATION_NUM[2]+1],avg_fitness[2,:EPOCHS//POPULATION_NUM[2]+1])
    plt.legend(['premala populacija: 4 jedinke','izabrana populacija: 20 jedinki','prevelika populacija: 200 jedinki'])
    plt.xlabel('broj generisanih kandidata')
    plt.ylabel('prosecna vrednost fitness funkcije')
    
    
    '''
    fig,ax = plt.subplots(1,3,figsize=(18,6),dpi=80)
    ax[0].plot(generated_num[0,:],avg_fitness[0,:])
    ax[1].plot(generated_num[1,:],avg_fitness[1,:])
    ax[2].plot(generated_num[2,:],avg_fitness[2,:])
    '''