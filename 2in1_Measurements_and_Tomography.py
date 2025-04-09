"""
Quantum States - Circuit and measurements

Authors: Giannina Zerr, Federico Holik and Marcelo Losada

"""
import numpy as np
from braket.devices import LocalSimulator

############# ENTRIES #############
# Choose here your options

# State #
state_type= 'GHZ' # options: 'GHZ' or 'Werner'
N= 2 # Number of qubits, choose any for GHZ, if it is a Werner N=2
s=25 # Only for Werner states, is 0 to 49, and corresponds to p in ps

# Tomography #
symmetry= 'Permutational' # options: 'Complete' or 'Permutational'
device = LocalSimulator("braket_dm")
c= 30 # number of measurements
shots = [100,300,500,700,1000,1500,2500]

# Noise #
noise_types=['Depolarizing','BitFlip','AmplitudeDamping']
noises_number=7 # Amount of differents levels of noise, without noise =1
noise_list= np.linspace(0, 0.15, noises_number)
#linspace(start, stop, num_samples)
noise_listzeros=np.zeros(noises_number)

############# LIBRARIES #############

import matplotlib
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline') for jupyter notebook
from qutip import *
from braket.aws import AwsDevice
from braket.circuits import Circuit, Gate, Noise, Observable, Instruction
from braket.circuits.noise_model import (GateCriteria, NoiseModel,
                                         ObservableCriteria)
from braket.circuits.noises import (AmplitudeDamping, BitFlip, Depolarizing,
                                    PauliChannel, PhaseDamping, PhaseFlip,
                                    TwoQubitDepolarizing)
from braket.devices import LocalSimulator
import statistics
import itertools
from itertools import permutations
import cvxpy as cp
from functools import reduce
from cvxopt import solvers, blas, matrix, spmatrix, spdiag, log, div
import math as m
from scipy import linalg as la
from scipy.linalg import logm, expm
from scipy import log, log2
from scipy.stats import unitary_group
import SP_functions as f
import sympy
from scipy.sparse import csr_matrix, lil_matrix
from scipy import sparse
import os
import shutil
import sys
from datetime import datetime

beggining_time = datetime.now()

############# FUNCTIONS #############

# Dictionaries saving
import json
def save_pet(pet,filename):
    with open(filename, 'w') as f:
        f.write(json.dumps(pet))
def load_pet(filename):
    with open(filename) as f:
        pet = json.loads(f.read())
    return pet

# Measurement projectors generator
def local_measurement(N):
    Px = [ ]
    for i in range(N):
        M = Instruction(Gate.H(), [i])
        C = Circuit([M])
        Px.append(C)

    Py = []
    for i in range(N):
        M1 = Instruction(Gate.Si(), [i]) 
        M2 = Instruction(Gate.H(), [i])      
        C1 = Circuit([M1,M2])
        Py.append(C1)

    Pz = []
    for i in range(N):
        M = Instruction(Gate.I(), [i])
        C = Circuit([M])
        Pz.append(C)
    return Px, Py, Pz


# Observables list for measurement
def observables(N, symmetry):
    if symmetry == 'Complete':
        L0 = []
        for comb in itertools.product('IXYZ', repeat=N):
            l = ''.join(list(comb))
            m = list(l.strip(" "))
            L0.append(m)
    if symmetry == 'Permutational':
        L0 = []
        for comb in itertools.combinations_with_replacement('IXYZ', N):
            l = ''.join(list(comb))
            m = list(l.strip(" "))
            L0.append(m) 
    return L0


############# NOISE MODEL #############

def noise(x):
    noise_model = NoiseModel()
    noise_model.add_noise(Depolarizing(x['Depolarizing_gateH']), GateCriteria(Gate.H))
    noise_model.add_noise(Depolarizing(x['Depolarizing_gateCNot']), GateCriteria(Gate.CNot))
    noise_model.add_noise(Depolarizing(x['Depolarizing_gateRy']), GateCriteria(Gate.Ry))
    
    noise_model.add_noise(BitFlip(x['BitFlip_gateH']), GateCriteria(Gate.H))
    noise_model.add_noise(BitFlip(x['BitFlip_gateCNot']), GateCriteria(Gate.CNot))
    noise_model.add_noise(BitFlip(x['BitFlip_gateRy']), GateCriteria(Gate.Ry))
    
    noise_model.add_noise(AmplitudeDamping(x['AmplitudeDamping_gateH']), GateCriteria(Gate.H))
    noise_model.add_noise(AmplitudeDamping(x['AmplitudeDamping_gateCNot']), GateCriteria(Gate.CNot))
    noise_model.add_noise(AmplitudeDamping(x['AmplitudeDamping_gateRy']), GateCriteria(Gate.Ry))
    
    return noise_model

def noisy(noise_type):
    noise_level=[]
    if noise_type == 'Depolarizing':
        for i in range(len(noise_list)):
            noise_dic={}
            noise_dic['Depolarizing_gateH']= noise_list[i]/10
            noise_dic['Depolarizing_gateCNot']= noise_list[i]
            noise_dic['Depolarizing_gateRy']= noise_list[i]/10
            noise_dic['BitFlip_gateH']=noise_listzeros[i]
            noise_dic['BitFlip_gateCNot']=noise_listzeros[i]
            noise_dic['BitFlip_gateRy']= noise_listzeros[i]
            noise_dic['AmplitudeDamping_gateH']=noise_listzeros[i]
            noise_dic['AmplitudeDamping_gateCNot']=noise_listzeros[i]
            noise_dic['AmplitudeDamping_gateRy']= noise_listzeros[i]
            noise_level.append(noise_dic)
    elif noise_type == 'BitFlip':
        for i in range(len(noise_list)):
            noise_dic={}
            noise_dic['Depolarizing_gateH']= noise_listzeros[i]
            noise_dic['Depolarizing_gateCNot']= noise_listzeros[i]
            noise_dic['Depolarizing_gateRy']= noise_listzeros[i]
            noise_dic['BitFlip_gateH']=noise_list[i]/10
            noise_dic['BitFlip_gateCNot']=noise_list[i]
            noise_dic['BitFlip_gateRy']=noise_list[i]/10
            noise_dic['AmplitudeDamping_gateH']=noise_listzeros[i]
            noise_dic['AmplitudeDamping_gateCNot']=noise_listzeros[i]
            noise_dic['AmplitudeDamping_gateRy']= noise_listzeros[i]
            noise_level.append(noise_dic)
    elif noise_type == 'AmplitudeDamping':
        for i in range(len(noise_list)):
            noise_dic={}
            noise_dic['Depolarizing_gateH']= noise_listzeros[i]
            noise_dic['Depolarizing_gateCNot']= noise_listzeros[i]
            noise_dic['Depolarizing_gateRy']= noise_listzeros[i]
            noise_dic['BitFlip_gateH']=noise_listzeros[i]
            noise_dic['BitFlip_gateCNot']=noise_listzeros[i]
            noise_dic['BitFlip_gateRy']= noise_listzeros[i]
            noise_dic['AmplitudeDamping_gateH']=noise_list[i]/10
            noise_dic['AmplitudeDamping_gateCNot']=noise_list[i]
            noise_dic['AmplitudeDamping_gateRy']=noise_list[i]/10
            noise_level.append(noise_dic)
    return noise_level,noise_dic


############# MEASUREMENTS FUNCTION #############

def measurements(circuit,L0,noise_level,shot,device,target):
    Tomos_Noises = []
    for x in range(len(noise_level)): # x is a dictionary of dictionaries
        noise_model=noise(noise_level[x])
        ### Real state with noise
        circ_0 = noise_model.apply(circuit)
        circ_0.density_matrix(target=target)
        task_0 = device.run(circ_0, shots=0)
        result_0 = task_0.result()
        DM = result_0.values[0]
        np.save(save_folder+'/DM_'+ str(x)+'.npy',DM)   
        Mean_Values = {}    
        for M in L0:
            A = noise_model.apply(circuit)
            obs=eval('Observable.' + M[0] + '()')
            m=M[0]
            for i in np.arange(1,len(M)):
                obs=obs@eval('Observable.' + M[i] + '()')
                m=m+M[i]
            # Measurements wanted
            A.expectation(observable = obs, target=list(target))
            result = device.run(A, shots = shot).result()
            Mean_Values[m] = result.values[0]
        Tomos_Noises.append(Mean_Values)
    return Tomos_Noises, DM
                          
############# CIRCUIT STATES #############

# GHZ circuit
def generation_circuit_GHZ (N):
    circuit = Circuit()
    circuit.h(0)
    for i in range(1,N):
        circuit.cnot(0, i)
    return circuit
                          
# Werner circuit
def generation_circuit_Werner(s, thetas): # s is 0 to 49, and corresponds to p in ps
    th1=thetas[s][0]
    th2=thetas[s][1]
    th3=thetas[s][2]
    circuit = Circuit()
    circuit.ry(0,th1)
    circuit.cnot(0,2)
    circuit.ry(0,th2)
    circuit.ry(2,th3)
    circuit.cnot(0,1)
    circuit.cnot(2,3)
    circuit.h(0)
    circuit.cnot(0,2)
    return circuit

# Possible values for p
ps=np.linspace(0,1,50)
# Corresponds to this thetas (th1=th2)
thetas=np.array([[-1.48410673e-06,  1.57079714e+00,  1.57079714e+00],
       [ 1.99995619e-02,  1.59121202e+00,  1.59121202e+00],
       [ 3.92216941e-02,  1.61165765e+00,  1.61165765e+00],
       [ 5.77088536e-02,  1.63216534e+00,  1.63216534e+00],
       [ 7.55078502e-02,  1.65275557e+00,  1.65275557e+00],
       [ 9.26426357e-02,  1.67344844e+00,  1.67344844e+00],
       [ 1.09138654e-01,  1.69428856e+00,  1.69428856e+00],
       [ 1.25036303e-01,  1.71527972e+00,  1.71527972e+00],
       [ 1.40339006e-01,  1.73643921e+00,  1.73643921e+00],
       [ 1.55068846e-01,  1.75778844e+00,  1.75778844e+00],
       [ 1.69239003e-01,  1.77934451e+00,  1.77934451e+00],
       [ 1.82859940e-01,  1.80112373e+00,  1.80112373e+00],
       [ 1.95939645e-01,  1.82314173e+00,  1.82314173e+00],
       [ 2.08483900e-01,  1.84541361e+00,  1.84541361e+00],
       [ 2.20514817e-01,  1.86795529e+00,  1.86795529e+00],
       [ 2.32005107e-01,  1.89077944e+00,  1.89077944e+00],
       [ 2.42967731e-01,  1.91390071e+00,  1.91390071e+00],
       [ 2.53401277e-01,  1.93733289e+00,  1.93733289e+00],
       [ 2.63302554e-01,  1.96108961e+00,  1.96108961e+00],
       [ 2.72652874e-01,  1.98518278e+00,  1.98518278e+00],
       [ 2.81485073e-01,  2.00963115e+00,  2.00963115e+00],
       [ 2.89754214e-01,  2.03444395e+00,  2.03444395e+00],
       [ 2.97454450e-01,  2.05963589e+00,  2.05963589e+00],
       [ 3.04581001e-01,  2.08522178e+00,  2.08522178e+00],
       [ 3.11117972e-01,  2.11121596e+00,  2.11121596e+00],
       [ 3.17066080e-01,  2.13763213e+00,  2.13763213e+00],
       [ 3.22358616e-01,  2.16448239e+00,  2.16448239e+00],
       [ 3.27010423e-01,  2.19179236e+00,  2.19179236e+00],
       [ 3.30999553e-01,  2.21959196e+00,  2.21959196e+00],
       [ 3.34290030e-01,  2.24787351e+00,  2.24787351e+00],
       [ 3.36852096e-01,  2.27667004e+00,  2.27667004e+00],
       [ 3.38649133e-01,  2.30600893e+00,  2.30600893e+00],
       [ 3.39642823e-01,  2.33592320e+00,  2.33592320e+00],
       [ 3.39786083e-01,  2.36643126e+00,  2.36643126e+00],
       [ 3.39035025e-01,  2.39758089e+00,  2.39758089e+00],
       [ 3.37313628e-01,  2.42941812e+00,  2.42941812e+00],
       [ 3.34548084e-01,  2.46198085e+00,  2.46198085e+00],
       [ 3.30665638e-01,  2.49534873e+00,  2.49534873e+00],
       [ 3.25555184e-01,  2.52959790e+00,  2.52959790e+00],
       [ 3.19091742e-01,  2.56482913e+00,  2.56482913e+00],
       [ 3.11127181e-01,  2.60117613e+00,  2.60117613e+00],
       [ 3.01430825e-01,  2.63880267e+00,  2.63880267e+00],
       [ 2.89750876e-01,  2.67794357e+00,  2.67794357e+00],
       [ 2.75726504e-01,  2.71893041e+00,  2.71893041e+00],
       [ 2.58820336e-01,  2.76222902e+00,  2.76222902e+00],
       [ 2.38263143e-01,  2.80859003e+00,  2.80859003e+00],
       [ 2.12664798e-01,  2.85928144e+00,  2.85928144e+00],
       [ 1.79436441e-01,  2.91679994e+00,  2.91679994e+00],
       [ 1.31933889e-01,  2.98765432e+00,  2.98765432e+00],
       [-2.87409174e-03,  3.13826865e+00,  3.13826865e+00]])
                          

############# ENTRIES #############
# You chose it at the begginning

if state_type=='GHZ':
    target=list(range(N)) # Measurements are over all qubits
    circuit=generation_circuit_GHZ(N)
    circuit_ideal=generation_circuit_GHZ(N) # for obtaining density matrix without noise
    state= str(state_type)+'_'+str(N)+'qubits' # for name folders
elif state_type=='Werner':
    N=2
    target=[2,3] # Measurements are going to be over qubits 2 and 3
    circuit=generation_circuit_Werner(s, thetas)
    circuit_ideal=generation_circuit_Werner(s, thetas) # for obtaining density matrix without noise
    p=ps[s]
    state= str(state_type)+'_s'+str(s)+'_'+str(N)+'qubits' # for name folders
    
print('ENTRIES:')
print('state: ', state)
print('Symmetry: ', symmetry)
print('Device: ', device)
print('Number of measurements: ', c)
print('Shots: ', shots)
print('Noise types: ', noise_types)
print('Noise levels: ', noise_list)
print('')

############# GENERATORS #############
# Do not change this unless you know what are you doing

Px,Py,Pz = local_measurement(N) # Observables
L0=observables(N,symmetry) # Measurement basis

# Folder for saving measurements
if not os.path.exists('Measurements'):
    os.mkdir('Measurements')
mother_folder='Measurements/' +state + "_" + symmetry

if os.path.exists(mother_folder):
    response = input(f"⚠️ The folder '{mother_folder}' already exists. Do you want to overwrite it? (y/N): ").strip().lower()
    if response == 'y':
        shutil.rmtree(mother_folder)  # Deletes the folder and its contents
        os.mkdir(mother_folder)  # Recreates the folder
        print(f"Folder '{mother_folder}' overwritten.")
    else:
        print(f"The existing folder '{mother_folder}' will be kept.")
        sys.exit(0)
else:
    os.mkdir(mother_folder)  # Creates the folder if it doesn't exist
    print(f"Folder '{mother_folder}' created.")

save_pet(noise_types,mother_folder+'/noise_types.txt')
list_noise_list=np.array(noise_list).tolist()
save_pet(list_noise_list,mother_folder+ '/noise_levels.txt')
save_pet(shots, mother_folder+ '/shots.txt')

doc= open(mother_folder+'/ENTRIES_'+str(state)+'.txt',"w+")
doc.write(' '+'\r\n')
doc.write('SELECTED ENTRIES'+'\r\n')
doc.write('State: '+ str(state)+'\r\n')
if state_type== 'Werner':
    doc.write('Werner p value: '+ str(p)+'\r\n')
    doc.write('Werner thetas values: th1='+ str(thetas[s][0])+' and th2=th3=' +str(thetas[s][1])+'\r\n')
doc.write('Number of measurements: '+str(c) +'\r\n')
doc.write('Set_shots = ' + str(shots)+'\r\n')
doc.write('Noise types = ' + str(noise_types)+'\r\n')
doc.write('Noise levels used: '+ str(noise_list) +'\r\n')
doc.write('Total amount of differents levels of noise: ' + str(noises_number) +'\r\n')
doc.write('Symmetry: '+ str(symmetry)+'\r\n')
doc.write('Measurement basis size= '+ str(len(L0))+'\r\n')    
doc.write(' '+'\r\n')
doc.write(' ----------------------------------------------- \r\n')
doc.write(' '+'\r\n')
doc.write('SPECIFICATIONS'+'\r\n')
doc.write(' '+'\r\n')
doc.write('Circuit:'+'\r\n')
doc.write(str(circuit)+'\r\n')
doc.write(' '+'\r\n')
doc.write('Measurement basis: '+ str(L0)+'\r\n')
doc.write(' '+'\r\n')
doc.close()

############# MEASUREMENTS #############
response = input(f"❔ Everything is ready for measurements. This may take a while. Do you want to continue? (Y/n): ").strip().lower()
if response == 'n':
    print('Stopping execution.')
    sys.exit(0)
else:
    print('STARTING MEASUREMENTS')
    
print('')
ti = datetime.now()

for x in range(len(noise_types)):
    noise_type=noise_types[x]
    print('Starting process for:',noise_type)
    noise_level,noise_dic=noisy(noise_type)

    save_folder=mother_folder+('/')+str(noise_type)
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    save_pet(noise_level,save_folder+ '/noise_level' + '.txt')

    doc= open(save_folder+'/ENTRIES_'+str(noise_type)+'.txt',"w+")
    doc.write(' '+'\r\n')
    doc.write('Noise type = ' + str(noise_type)+'\r\n')
    doc.write(' '+'\r\n')
    doc.write('Noise levels used: ' +'\r\n')
    for key in noise_dic:
        doc.write(str(key)+(': '))
        for level in noise_level:
            doc.write(str(level[key])+(' '))
        doc.write('\r\n')
    doc.close()
    
    # Obtain density matrix for ideal state (without noise)
    circuit_ideal.density_matrix(target=target)
    task_ideal = device.run(circuit_ideal, shots=0)
    result_ideal = task_ideal.result()
    DM_ideal = result_ideal.values[0]
    np.save(save_folder+'/DM_ideal.npy',DM_ideal)
    
    print('----- Starting measurements')
    for k in range(len(shots)):
        total_data = []
        for i in np.arange(1,c+1):
            data = measurements(circuit,L0,noise_level, shots[k],device, target)[0]
            total_data.append(data)
        save_pet(total_data, save_folder+ '/'+ 'shots'+ str(shots[k]) + '.txt')
        print(round(((k+1)*100)/(len(shots)*len(noise_types))+(x*100)/(len(noise_types)),2),'%')
    print('Ending process for',noise_type)
    print(' ')
    t = datetime.now()
    print('Processing time=', t-ti)
    print(' ')
print('Measurements are completed =)')
print('')


############# GENERATORS AND MEASUREMENTS READING #############
# Do not change this unless you know what are you doing

print('Loading Basis')
BasisOrt_Load = np.load("Orthogonal_Basis/BasisOrt_" + symmetry + "_" + str(N) + ".npz", allow_pickle=True) # States will be write on this base, it deppends on the symmetry type
BasisOrt = [BasisOrt_Load[key] for key in BasisOrt_Load] 
Size = len(BasisOrt)
print('Symmetry:', symmetry)
print('Ortogonal base size (space dimension)=',Size)
print('')

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

# The observables measured are imported from the data

noise_types=load_pet(measurements_folder+ '/noise_types.txt')
noise_levels=load_pet(measurements_folder+ '/noise_levels.txt')
shots=load_pet(measurements_folder+ '/shots.txt')

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

end_time = datetime.now()
print("Total duration of work: {}".format(end_time - beggining_time))