########## Machine Learning for Quantum Matter and Technology  ######################
### Juan Carrasquilla, Estelle Inack, Giacomo Torlai, Roger Melko
### with code from Lauren Hayward Sierens/PSI
### Tutorial 1: Monte Carlo for the Ising model
#####################################################################################

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import os
import random
import time

_UNDEFINED = -11111111
CUR_ENERGIA = _UNDEFINED
CUR_MAGNETI = _UNDEFINED
n_sweeps = 0

def updateEnergia(flipped_site): # Asume que el spin del sitio ya ha sido cambiado
    global CUR_ENERGIA
    if CUR_ENERGIA == _UNDEFINED:
          CUR_ENERGIA = getEnergy()
    else:
          CUR_ENERGIA += 2*J*(spins[flipped_site]*spins[neighbours[flipped_site,0]] + spins[flipped_site]*spins[neighbours[flipped_site,1]] + spins[flipped_site]*spins[neighbours[flipped_site,2]] + spins[flipped_site]*spins[neighbours[flipped_site,3]])
            
def updateMagnetizacion(flipped_site): # Asume que el spin del sitio ya ha sido cambiado
    global CUR_MAGNETI
    if CUR_MAGNETI == _UNDEFINED:
          CUR_MAGNETI = getEnergy()
    else:
          CUR_MAGNETI += 2*spins[flipped_site]

### Input parameters: ###
T_list = np.linspace(5.0,0.5,int(input("Numero de pasos: "))) #temperature list
L = int(input("Tamanio cuadricula: "))                            #linear size of the lattice
N_spins = L**2                   #total number of spins
J = 1                            #coupling parameter

### Critical temperature: ###
Tc = 2.0/np.log(1.0 + np.sqrt(2))*J  #T/J ~ 2.269

### Monte Carlo parameters: ###
n_eqSweeps = 1000   #number of equilibration sweeps : "rule of thumb: ~10% of measurement sweeps. Think of it as the time it takes for the system to reach thermal equilibrium given the random initial state"
n_measSweeps = 10000  #number of measurement sweeps

### Parameters needed to show animation of spin configurations: ###
animate = False
bw_cmap = colors.ListedColormap(['black', 'white'])

### Create a directory where measured observables will be stored: ###
results_dir = 'Data'
if not(os.path.isdir(results_dir)):
    os.mkdir(results_dir)

### Initially, the spins are in a random state (a high-T phase): ###
spins = np.zeros(N_spins,dtype=np.int)
for i in range(N_spins):
    spins[i] = 2*random.randint(0,1) - 1 #either +1 or -1

### Store each spin's four nearest neighbours in a neighbours array (using periodic boundary conditions): ###
neighbours = np.zeros((N_spins,4),dtype=np.int)
for i in range(N_spins):
    #neighbour to the right:
    neighbours[i,0]=i+1
    if i%L==(L-1):
        neighbours[i,0]=i+1-L
    
    #upwards neighbour:
    neighbours[i,1]=i+L
    if i >= (N_spins-L):
        neighbours[i,1]=i+L-N_spins

    # *********************************************************************** #
    # **********          1a) FILL IN CODE TO CALCULATE           *********** #
    # **********  THE NEIGHBOUR TO THE LEFT (IN neighbours[i,2])  *********** #
    # ********** AND THE DOWNWARDS NEIGHBOUR (IN neighbours[i,3]) *********** #
    # *********************************************************************** #
    # DONE
    #neighbour to the left:
    neighbours[i,2]=i-1
    if i%L==0:
        neighbours[i,2] = i + L-1
    
    #downwards neighbour:
    neighbours[i,3]=i-L
    if i < L:
        neighbours[i,3]= N_spins-L + i
#end of for loop

### Function to calculate the total energy ###
def getEnergy():
    energy = 0
    for i in range(N_spins):
        energy += -J*( spins[i]*spins[neighbours[i,0]] + spins[i]*spins[neighbours[i,1]]) # This is correct
    return energy
#end of getEnergy() function

### Function to calculate the total magnetization ###
def getMag():
    return np.sum(spins)
#end of getMag() function

### Function to perform one Monte Carlo sweep ### #i.e. a Transition from one state to another; for us this si given by the Metropolis algorithm
def sweep(): # Modified to update by itself the energy
    #do one sweep (N_spins single-spin flips):
    for i in range(N_spins):
        #randomly choose which spin to consider flipping:
        site = random.randint(0,N_spins-1)
        #calculate the change in energy for the proposed move:
        deltaE = 2*J*(spins[site]*spins[neighbours[site,0]] + spins[site]*spins[neighbours[site,1]] + spins[site]*spins[neighbours[site,2]] + spins[site]*spins[neighbours[site,3]]) # Añadido en reemplazo de las anteriores
        # *********************************************************************** #
        # ************       1c) REPLACE THE ABOVE FIVE LINES.        *********** #
        # ************ FILL IN CODE TO CALCULATE THE CHANGE IN ENERGY *********** #
        # ************     USING ONLY THE FOUR NEAREST NEIGHBOURS     *********** #
        # *********************************************************************** #
    
        if (deltaE <= 0) or (random.random() < np.exp(-deltaE/T)):  #Metropolis algorithm realization of the transition probability
            #flip the spin:
            spins[site] = -spins[site]
            updateEnergia(site)
    #end loop over i
#end of sweep() function

#################################################################################
########## Loop over all temperatures and perform Monte Carlo updates: ##########
#################################################################################
t1 = time.clock() #for timing

            
for T in T_list: 
    CUR_ENERGIA =  _UNDEFINED
    CUR_MAGNETI =  _UNDEFINED
    
    print('\nT = %f' %T)
    
    #open a file where observables will be recorded:
    fileName         = '%s/ising2d_L%d_T%.4f.txt' %(results_dir,L,T)
    file_observables = open(fileName, 'w')
    
    #equilibration sweeps:
    for i in range(n_eqSweeps):
        sweep()
    
    #start doing measurements: #sweep and then measure, n_measSweeps times for each Temp: CREATE A SAMPLE FOR THIS TEMPERATURE and measure
    for i in range(n_measSweeps):
        sweep()

        #Write the observables to file:
        #energy = getEnergy() #Done by the sweep automatically
        #mag    = getMag()
        file_observables.write('%d \t %.8f \t %.8f \n' %(i, CUR_ENERGIA, CUR_MAGNETI))

        if animate:
            #Display the current spin configuration:
            plt.clf()
            plt.imshow( spins.reshape((L,L)), cmap=bw_cmap, norm=colors.BoundaryNorm([-1,0,1], bw_cmap.N), interpolation='nearest' )
            plt.xticks([])
            plt.yticks([])
            plt.title('%d x %d Ising model, T = %.3f' %(L,L,T))
            plt.pause(0.01)
        #end if

        if (i+1)%1000==0:
            print('  %d sweeps complete' %(i+1))
    #end loop over i

    file_observables.close()
#end loop over temperature

t2 = time.clock()
print('Elapsed time: %f seconds' %(t2-t1))
