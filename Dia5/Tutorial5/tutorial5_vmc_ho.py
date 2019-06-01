######################################################################################
### Code taken from https://github.com/agdelma/qmc_ho
### modified by Estelle Inack 
###
###  Variational Monte Carlo for the harmonic oscillator
######################################################################################

import numpy as np
import matplotlib.pyplot as plt

red,blue,green = '#e85c47','#4173b2','#7dcca4'

#/// psi_al(x) = e^(-al x^2) => P(x) = e^(-2al x^2) 

def EL(x,α): # /// H = p^2/2 + x^2/2 applied o e^(-al x^2)
    return α + x**2*(0.5-2*α**2)

def transition_probability(x,x̄,α):   #/// P() / P(x) Tr(x -> x̄) = e^
    return np.exp(-2*α*(x̄**2-x**2))

def vmc(num_walkers,num_MC_steps,num_equil_steps,α,δ=1.0):
    
    # initilaize walkers   /// Positions that will be the result of the sampling
    walkers = -0.5 + np.random.rand(num_walkers)
    
    # initialize energy and number of accepted updates
    estimator = {'E':np.zeros(num_MC_steps-num_equil_steps)} #/// Global variables
    num_accepted = 0
    
    for step in range(num_MC_steps):
        
        # generate new walker positions 
        new_walkers = np.random.normal(loc=walkers, scale=δ, size=num_walkers)
        
        # test new walkers
        for i in range(num_walkers):
            if np.random.random() < transition_probability(walkers[i],new_walkers[i],α):
                num_accepted += 1
                walkers[i] = new_walkers[i]
                
            # measure energy
            if step >= num_equil_steps:
                measure = step-num_equil_steps  # /// Number of measurement
                estimator['E'][measure] = EL(walkers[i],α)
                
    # output the acceptance ratio
    print('accept: %4.2f' % (num_accepted/(num_MC_steps*num_walkers)))
    
    return estimator

α = 0.4
num_MC_steps = 30000 #// Number of points with which we will approximate the current distribution e^(-2 al x^2)
num_walkers = 400   #// Number of times we will do the aproximation, i.e. we are approximating the distribution, for each alpha, this number of times. This is required to get a standard deviation in the variational energy, which will help us use the minimum variance principle to determine if our minimum E_var function is also an eigenstate.
num_equil_steps = 3000

np.random.seed(1173)

estimator = vmc(num_walkers,num_MC_steps,num_equil_steps,α)

#from scipy.stats import sem
Ē,ΔĒ = np.average(estimator['E']),np.std(estimator['E'])/np.sqrt(estimator['E'].size-1)

print('Ē = %f ± %f' % (Ē,ΔĒ))

Ēmin = []
ΔĒmin = []
α = np.array([0.3, 0.45, 0.475, 0.5, 0.525, 0.55, 0.7])
for cα in α: 
    estimator = vmc(num_walkers,num_MC_steps,num_equil_steps,cα)
    Ē,ΔĒ = np.average(estimator['E']),np.std(estimator['E'])/np.sqrt(estimator['E'].size-1)
    Ēmin.append(Ē)
    ΔĒmin.append(ΔĒ)
    print('%5.3f \t %7.5f ± %f' % (cα,Ē,ΔĒ))


plt.figure(figsize=(8,16)) #///
cα = np.linspace(α[0],α[-1],1000)

plt.subplot(121) #/// Plot calculated variational energies
plt.plot(cα,0.5*cα + 1/(8*cα), '-', linewidth=1, color=green, zorder=-10, 
         label=r'$\frac{\alpha}{2} + \frac{1}{8\alpha}$')
plt.errorbar(α,Ēmin,yerr=ΔĒmin, linestyle='None', marker='o', elinewidth=1.0, 
             markersize=6, markerfacecolor=blue, markeredgecolor=blue, ecolor=blue, label='VMC')
plt.xlabel(r'$\alpha$')
plt.ylabel('$E_{var}$')
plt.xlim(0.29,0.71)
plt.legend(loc='upper center')

 #// Plot for standard deviation
plt.subplot(122)
plt.scatter(α, ΔĒmin)
plt.xlim(0.29,0.71)
plt.ylim(0, 0.001)
plt.ylabel('$\Delta E_{var}$')
plt.xlabel(r'$\alpha$')

plt.show() 
