"""
Quantum States Tomography

Authors: Giannina Zerr, Federico Holik and Marcelo Losada

"""
############# ENTRIES #############
# Choose here your options
# You must have the measurements over the selected state

state_type= 'GHZ' # options: 'GHZ' or 'Werner'
N= 2 #  Number of qubits, only for GHZ, if it's a Werner N=2
s=25 # Only for Werner states, is 0 to 49, and corresponds to p in ps
symmetry= 'Permutational' # options: 'Complete' or 'Permutational'

############# LIBRARIES #############

# Remember to set console working directory

import matplotlib.pyplot as plt
import matplotlib
import statistics
from qutip import *
import cvxpy as cp
from functools import reduce
from cvxopt import solvers, blas, matrix, spmatrix, spdiag, log, div
import numpy as np
import math as m
from scipy import linalg as la
from scipy.linalg import logm, expm
from scipy import log, log2
from scipy.stats import unitary_group
import SP_functions as f
import itertools
import sympy
from itertools import permutations
import itertools
from scipy.sparse import csr_matrix, lil_matrix
from scipy import sparse
import pandas as pd
from datetime import datetime
import statistics

# For folder saving:
import json
def save_pet(pet,filename):
    with open(filename, 'w') as f:
        f.write(json.dumps(pet))
def load_pet(filename):
    with open(filename) as f:
        pet = json.loads(f.read())
    return pet
import os
import shutil
import sys

############# GENERATORS AND MEASUREMENTS READING #############
# Do not change this unless you know what are you doing

print('ENTRIES')
if state_type=='GHZ':
    state= str(state_type)+'_'+str(N)+'qubits' # for name folders
    print('State:', state_type, N, 'qubits')
elif state_type=='Werner':
    N=2
    state= str(state_type)+'_s'+str(s)+'_'+str(N)+'qubits' # for name folders
    print('State:', state_type, 's=', s,  N, 'qubits')

BasisOrt_Load = np.load("Orthogonal_Basis/BasisOrt_" + symmetry + "_" + str(N) + ".npz", allow_pickle=True) # States will be write on this base, it deppends on the symmetry type
BasisOrt = [BasisOrt_Load[key] for key in BasisOrt_Load] 
Size = len(BasisOrt)
print('Symmetry:', symmetry)
print('Ortogonal base size (space dimension)=',Size)

# We define variables and specify chosen observables:
a = cp.Variable(len(BasisOrt)) # Linear combination coefficients ("weights") of each element of the base, which I am going to optimize
expr1 = sum(a[i] * BasisOrt[i] for i in range(len(BasisOrt))) # rho as a variable

# Paulis
I = np.array([[1, 0], [0, 1]])
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1.0j], [1.0j, 0]])
Z = np.array([[1, 0], [0, -1]])
Matrixes={'I': I, 'X': X, 'Y': Y, 'Z': Z}

print('')
# Measurements Folder
measurements_folder= 'Measurements/' +str(state)+ '_' + str(symmetry)
if not os.path.exists(measurements_folder):
    print(f"⚠️ The folder '{measurements_folder}' doesn't exists.")
    sys.exit(0)
    
# The observables measured are imported from the data

noise_types=load_pet(measurements_folder+ '/noise_types.txt')
print('Noise types:', noise_types)
noise_levels=load_pet(measurements_folder+ '/noise_levels.txt')
print('Noise levels:', noise_levels)
shots=load_pet(measurements_folder+ '/shots.txt')
print('Shots:', shots)
print('')
# Folder for saving results
if not os.path.exists('Results'):
    os.mkdir('Results')
results_folder = 'Results/'+str(state)+'_'+str(symmetry)
if os.path.exists(results_folder):
    response = input(f"⚠️ The folder '{results_folder}' already exists. Do you want to overwrite it? (y/N): ").strip().lower()
    if response == 'y':
        shutil.rmtree(results_folder)  # Deletes the folder and its contents
        os.mkdir(results_folder)  # Recreates the folder
        print(f"Folder '{results_folder}' overwritten.")
    else:
        print(f"The existing folder '{results_folder}' will be kept.")
        sys.exit(0)
else:
    os.mkdir(results_folder)  # Creates the folder if it doesn't exist
    print(f"Folder '{results_folder}' created.")

save_pet(noise_types, results_folder+ '/noise_types.txt')
save_pet(noise_levels, results_folder+ '/noise_levels.txt')
save_pet(shots, results_folder+ '/shots.txt')

response = input(f"❔ Everything is ready for tomography. This may take a while. Do you want to continue? (Y/n): ").strip().lower()
if response == 'n':
    print('Stopping execution.')
    sys.exit(0)
else:
    print('STARTING TOMOGRAPHY')
    
print('')
    
############# TOMOGRAPHY #############

for x in noise_types:
    noise_type=x
    #folder=noise_type
    print('Processing data for ',noise_type)

    # Folder for saving results for a noise type
    results_folder_noise_type=results_folder+'/'+str(noise_type)
    if not os.path.exists(results_folder_noise_type):
        os.mkdir(results_folder_noise_type)   
    
    # Data lecture for different noise levels
    measurement_folder_noise_type=measurements_folder+('/')+str(noise_type)    
    noise_level=load_pet(measurement_folder_noise_type+ '/noise_level.txt')
    DM_ideal=np.load(measurement_folder_noise_type+'/DM_ideal.npy')
    
    data_extraction=[]
    for k in range(len(shots)):
        data = load_pet(measurement_folder_noise_type+'/'+ 'shots'+ str(shots[k]) + '.txt')
        data_extraction.append(data)
    
    # Observables measured
    data_keys=list(data_extraction[0][0][0].keys())
    measured_observables=[]
    for i in range(len(data_keys)):
        product=1
        for caracter in data_keys[i]:
            product= np.kron(product, Matrixes[caracter])
        measured_observables.append(product)
    
    # 'Capital Delta' (see paper)
    d = cp.Variable(len(measured_observables))
    
    # Mean Values (size of list same as noise_level) 
    total_measurements=[]
    for k in range(len(shots)):
        measurements=[]
        for j in range(len(data_extraction[k])):
            set_mean_values=[]
            for i in range(len(noise_level)):
                data = data_extraction[k][j][i]
                mean_values=[]
                for h in data_keys:
                    value= data[h]
                    mean_values.append(value) # mean vlaues of one measurement
                set_mean_values.append(mean_values) # mean vlaues of a set with differents noise levels
            measurements.append(set_mean_values) # mean values of the total measurements
        print( round(((k+1)*100) / (len(data_extraction)),2) ,'%') 
        total_measurements.append(measurements)
      
    for key in noise_level[0]:
        values_for_key=[]
        for level in noise_level:
            value= level[key]
            values_for_key.append(value)
  
    counter = 0
    counter_err = 0
    
    print('Starting tomography for ',noise_type)
    Fid_shots=[]
    Fid_shots_R = []    
    Fid_shots_SD = []
    Fid_shots_R_SD = []
    
    for k in range(len(shots)):
        measurements=total_measurements[k]
        total_fidelities=[]
        total_fidelities_R = []      
        total_fidelities_SD = []
        total_fidelities_R_SD = []
        
        t_1 = datetime.now()
        
        for n in range(len(noise_level)): # para distintos niveles de ruido
            Fidelities = []
            Fidelities_R = []
            Trace_Distances = []
            Obtained = []
            StateComp = []
            
            for h in range(len(measurements)): # number of experiments
                mean_values=measurements[h][n]
                MeanValues = list(
                    d[j] * cp.abs(mean_values[j])
                    >= cp.abs(cp.trace(expr1 @ measured_observables[j]) - mean_values[j])
                    for j in range(len(measured_observables)))
                Ineqs = list(d[i] >= 0 for i in range(len(measured_observables)))
                constr1 = [expr1 >> 0, cp.trace(expr1) == 1] + MeanValues + Ineqs
                Variational = 100 * sum(d[i] for i in range(len(measured_observables))) - 0.1*cp.log_det(expr1) # functional
                
                obj = cp.Minimize(Variational)
                prob = cp.Problem(obj, constr1)
                prob.solve(
                    solver=cp.SCS,
                    verbose=False,
                    eps=0.0001e-01,
                    alpha=1,
                    max_iters=1000,
                    normalize=True,
                    scale=0.1,
                    acceleration_lookback=19,
                    rho_x=1.00e-08,
                    warm_start=True,
                    )
                R = a.value
                t_2 = datetime.now()
        
                try:
                    RM0 = sum(a.value[i] * BasisOrt[i] for i in range(len(BasisOrt)))
                    RM = RM0 / np.trace(RM0)
                    Fid = f.fidelity(DM_ideal, RM)
                    RHO_Real = np.load(measurement_folder_noise_type + '/DM_'+ str(n)+'.npy')
                    Fid_R = f.fidelity(RHO_Real, RM)
                    TD = tracedist(Qobj(RM), Qobj(DM_ideal))
                    Fidelities.append(Fid)
                    Fidelities_R.append(Fid_R)
                    Trace_Distances.append(TD)
                    Obtained.append(RM)
                    StateComp.append(DM_ideal)
                    counter = counter + 1
                except TypeError:
                    counter_err = counter_err + 1
                print(round((k*100)/len(shots)+(n*100)/(len(noise_level)*len(shots))+(h*100)/(len(noise_level)*len(shots)*len(measurements)),2),'%')
            Fidelity= sum(Fidelities[i] for i in range(len(Fidelities)))/len(Fidelities)
            Fidelity_SD = statistics.pstdev(Fidelities)    
            Fidelity_R= sum(Fidelities_R[i] for i in range(len(Fidelities_R)))/len(Fidelities_R)
            Fidelity_SD_R = statistics.pstdev(Fidelities_R)
            total_fidelities.append(Fidelity)
            total_fidelities_SD.append(Fidelity_SD)   
            total_fidelities_R.append(Fidelity_R)
            total_fidelities_R_SD.append(Fidelity_SD_R)
            if counter_err !=0:
                print('No convergence', counter_err)
            t_2 = datetime.now()
            np.save(results_folder_noise_type+'/DMtomo_noise'+str(n)+'_shots'+str(shots[k]),Obtained)       
        Fid_shots.append(total_fidelities)
        Fid_shots_R.append(total_fidelities_R)    
        Fid_shots_SD.append(total_fidelities_SD)
        Fid_shots_R_SD.append(total_fidelities_R_SD)
    
    t_2 = datetime.now()
    print("Processing time=: {}".format(t_2 - t_1))    
    
    print('Saving results and plotting')
  
    for k in range(len(shots)):
        Fidelities=[]
        Fidelities_R=[]
        Fidelities_SD=[]
        Fidelities_R_SD=[]
        for i in range(len(noise_level)):
            Fidelities.append(Fid_shots[k][i])
            Fidelities_R.append(Fid_shots_R[k][i])
            Fidelities_SD.append(Fid_shots_SD[k][i])
            Fidelities_R_SD.append(Fid_shots_R_SD[k][i])
    
        final_result={}
        final_result['Fidelities']=Fidelities
        final_result['Fidelities_SD']=Fidelities_SD
        final_result['Fidelities_R']=Fidelities_R
        final_result['Fidelities_R_SD']=Fidelities_R_SD
        save_pet(final_result,results_folder_noise_type+'/Fidelities_shots'+ str(shots[k]) + '.txt')
    
    
    #Plotting
    
    # Values of x axis
    noise=str(noise_type)
    gate = str(noise +'_gateCNot')
    noise_level_axis = [ x[gate] for x in noise_level ]
    
    for k in range(len(shots)):
        plt.errorbar(noise_level_axis, Fid_shots_R[k], Fid_shots_R_SD[k], marker='o', markersize=2,  linestyle='None',label=noise)
        plt.errorbar(noise_level_axis, Fid_shots[k], Fid_shots_SD[k], marker='o', markersize=2, linestyle='None',label=noise+ ' Ideal')
        plt.hlines(
            y=1,
            xmin=[noise_level_axis[0]],
            xmax=[noise_level_axis[-1]],
            colors="purple",
            linestyles="--",
            lw=2, 
        )
        plt.xlabel('Noise level ' + str(gate), fontsize=15)
        plt.ylabel("Fidelity", fontsize=15)
        plt.legend()
        plt.title(str(state)+ '   Shots '+str(shots[k])+'   '+ str(noise_type), fontsize=16)
        plt.savefig(results_folder_noise_type +'/tomo'+ str(state) +('_')+str(noise_type)+ '_shots'+ str(shots[k])  +'.png', dpi=300)
        plt.show()
    print('Ending process for',noise_type)
    print('---------------------------------------------')
print('Tomography is completed =)')
t_2 = datetime.now()
print("Total duration of tomography: {}".format(t_2 - t_1))
print('')
print('---------------------------------------------')

############# COMPARATIVE GRAPHS #############
print('Starting comparative graphs')
noise_types=load_pet(results_folder+ '/noise_types.txt')
noise_levels=load_pet(results_folder+ '/noise_levels.txt')
shots=load_pet(results_folder+ '/shots.txt')

# Noise comparison
noise_level_axis = noise_levels
for k in shots:    
    for noise in noise_types:
        fid_data = load_pet(results_folder + '/' + str(noise) + '/Fidelities_shots' + str(k) + '.txt')
        plt.errorbar(noise_levels, fid_data['Fidelities_R'], fid_data['Fidelities_R_SD'], 
                 marker='.', markersize=2, linestyle='None', label=f'{noise}')
        plt.errorbar(noise_levels, fid_data['Fidelities'], fid_data['Fidelities_SD'], 
                 marker='.', markersize=2, linestyle='None', label=f'{noise} Ideal')   
    plt.hlines(
        y=1,
        xmin=[noise_level_axis[0]],
        xmax=[noise_level_axis[-1]],
        colors="purple",
        linestyles="--",
        lw=2, 
    )
    plt.xlabel('Noise level', fontsize=15)
    plt.ylabel("Fidelity", fontsize=15)
    plt.legend(loc='lower left', fontsize=8)
    plt.title('Shots ='+str(k), fontsize=16)
    plt.savefig(results_folder+'/comparative_noise_for_fixed_shots='+str(k)+'.png', dpi=300)
    plt.show()
   
# Shots comparison
for x in range(len(noise_levels)):    
    fid_results = {noise: {'Fid': [], 'Fid_SD': []} for noise in noise_types}   
    for k in shots:
        for noise in noise_types:
            Fid = load_pet(results_folder + '/' + str(noise) + '/Fidelities_shots' + str(k) + '.txt')
            fid_results[noise]['Fid'].append(Fid['Fidelities_R'][x])
            fid_results[noise]['Fid_SD'].append(Fid['Fidelities_R_SD'][x])
    for i, noise in enumerate(noise_types):
        plt.errorbar(shots, fid_results[noise]['Fid'], fid_results[noise]['Fid_SD'], 
                     marker='.', markersize=2, linestyle='None', label=noise)  
    plt.hlines(y=1, xmin=shots[0], xmax=shots[-1], colors="purple", linestyles="--", lw=2)    
    plt.xlabel('Shots', fontsize=15)
    plt.ylabel('Fidelity', fontsize=15)
    plt.legend(loc='lower left', fontsize=8)
    plt.title(f'Noise = {round(noise_levels[x], 3)}', fontsize=16)
    plt.savefig(f'{results_folder}/comparative_shots_for_fixed_noise={round(noise_levels[x], 3)}.png', dpi=300)
    plt.show()
print('Comparative graphs are completed')
print('')
print('Work completed =D')
