from math import exp
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)

def f(x):
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    return 4/3*(x1**2 + x2**2 - x1*x2)**0.75 + x3

def neighbour(x):
    rand_delta = (np.random.rand(len(x))-0.5)*0.2
    x = x + rand_delta
    x[x>2] = 2
    x[x<0] = 0 
    return x

def an_iter(x, Tk):
    x_new = neighbour(x)
    delta = f(x_new) - f(x)
    if delta<=0:
        x = x_new
    else:
        rnd = np.random.rand()
        if rnd < exp(-delta/Tk):
            x = x_new
    
    func = f(x)
    return x, func

def init_an():
    x = np.random.uniform(0,2,size=(3,))
    func = f(x)
    return x, func
    
    

if __name__ == '__main__':
    
    num_repeat = 100
    
    # cooling schedule
    # sch 1
    M1 = 30
    T1 = np.linspace(0.3, 0, num=50,endpoint=False)
    T1 = np.repeat(T1,M1)
    # sch 2 - too fast
    M2 = 5
    T2 = np.linspace(0.3, 0, num=30,endpoint=False)
    T2 = np.repeat(T2,M2)
    # sch 3 - too slow
    M3 = 200
    T3 = np.linspace(0.3, 0, num=30,endpoint=False)
    T3 = np.repeat(T3,M3)
    
    M = [M1,M2,M3]
    T = [T1,T2,T3]
    
    
    avg_best_f = []
    avg_current_f = []
    
    for sch in range(3):
        # for every cooling schedule
        best_f = np.zeros((num_repeat,len(T[sch])+1))
        current_f = np.zeros((num_repeat,len(T[sch])+1))
        
        for i in range(num_repeat):
            # init
            x, best = init_an()
            best_x = x
            k = 0
            best_f[i,k] = best
            current_f[i,k] = best
            
            # begin cooling
            for Tk in T[sch]:
                x,func = an_iter(x,Tk)
                
                # update best and current values
                k = k + 1
                if (func < best):
                    best = func
                    best_x = x
                current_f[i,k] = func
                best_f[i,k] = best
        
        avg_best_f.append(np.average(best_f,axis=0))
        avg_current_f.append(np.average(current_f,axis=0))
    
    # plots
    fig,ax = plt.subplots(1,3,figsize = (18,6), dpi=80)
    ax[0].plot(avg_best_f[0],'b')
    ax[0].plot(avg_current_f[0],'r')
    ax[0].set_title('Odabrani raspored hladjenja')
    ax[0].set_ylabel('f(x)')
    ax[0].set_xlabel('broj iteracija')
    ax[0].legend(['najbolje f','trenutno f'])
    ax[1].plot(avg_best_f[1],'b')
    ax[1].plot(avg_current_f[1],'r')
    ax[1].set_title('Prebrzo hladjenje')
    ax[1].legend(['najbolje f','trenutno f'])
    ax[1].set_ylabel('f(x)')
    ax[1].set_xlabel('broj iteracija')
    ax[2].plot(avg_best_f[2],'b')
    ax[2].plot(avg_current_f[2],'r')
    ax[2].set_title('Presporo hladjenje')
    ax[2].legend(['najbolje f','trenutno f'])
    ax[2].set_ylabel('f(x)')
    ax[2].set_xlabel('broj iteracija')
    #plt.suptitle('Konvergencija kriterijumske funkcije')
    
    plt.figure()
    plt.plot(T[0])
    plt.plot(T[1])
    plt.plot(T[2])
    plt.legend(['odabrano hladjenje','prebrzo hladjenje','presporo hladjenje'])
    plt.xlabel('broj iteracija')
    plt.ylabel('T')
    #plt.title('Rasporedi hladjenja')
    
    
    
    
    
    
