import time
import numpy as np
import scipy as sp
import scipy.stats
import math
import subprocess as prcs
from scipy.optimize import curve_fit
from lmfit import Model


''' User Inputs '''
# Thermo file names 
# File Structure:
# 	Step | Volume | Pressure
AA_Thermo_Name = "MD_AA_Thermo.out"
CG_Thermo_Name = "MD_CG_Thermo.out"
AA_TrajName    = "dump.coords.dat"
CG_Pressure_Name = "MD_CG_Pressure.dat"
NumberThreads = str(4)

def Analytic2TermDA(x,Psi_1,Psi_2,N,AAAvgVol):
	''' Two term Das-Anderson volume correction '''
	return -1*N/AAAvgVol*(Psi_1+2*Psi_2*((x/AAAvgVol)-1))
	
def FitCurve(func, xdata, ydata):
	''' Fit data to a function form '''
	fmodel = Model(func)
	params = fmodel.make_params(Psi_1=1, Psi_2=0, N=640, AAAvgVol=10648)
	params['N'].vary = False
	params['AAAvgVol'].vary = False
	results = fmodel.fit(ydata, params, x=xdata)
	y = fmodel.eval(x=xdata, params=results.params)
	#parameters_opt, parameter_covariance = curve_fit(func,xdata,ydata)
	return results, ydata

''' ********** The module *********** '''
Kb    = 1.3806485E-23    # Boltzmann constant in J-(K-molecule)^-1
Nav   = 6.022E23		 # Avagodro's number
T     = 298 			 # Temperature K-molecule
sigma = 0.1 			 # Length scale in nanometers

# Replace dummy trajectory in LAMMPS RERUN File
with open('LAMMPS_RUN.in', 'r') as g:
	filedata = g.read()

filedata = filedata.replace('DUMMY_CGTRAJ', str(AA_TrajName))

with open('LAMMPS_RUN_OUT.in', 'w') as g:
	g.write(filedata)

call_1 = "lmp_serial.exe -sf omp -pk omp "+ NumberThreads+ ' -i LAMMPS_RUN_OUT.in'

print (call_1)
p1 = prcs.Popen(call_1, stdout=prcs.PIPE, shell=True)	
(output, err) = p1.communicate()
p_status = p1.wait()
print("{}".format(err))

with open("LAMMPS_LOG.out",'w') as logout:
	#for i in str(output):
	logout.write(output.decode("utf-8"))

# load in thermo data
delete = range(0,0) # delete the first XX rows, if necessary
AA_Thermo = np.delete(np.loadtxt(AA_Thermo_Name), delete, axis=0)
CG_Thermo = np.delete(np.loadtxt(CG_Pressure_Name), delete, axis=0)


AA_Step      = AA_Thermo[:,0]
AA_Volume    = AA_Thermo[:,1]
AA_Pressure  = AA_Thermo[:,2]
CG_Step      = CG_Thermo[:,0]
CG_Pressure  = CG_Thermo[:,1]

DeltaPressure = np.subtract(AA_Pressure,CG_Pressure)
Delta_Pressure_Out = np.column_stack((AA_Volume,DeltaPressure))

np.savetxt("AA_DeltaPressure.dat",Delta_Pressure_Out)

results, ydata = FitCurve(Analytic2TermDA, (AA_Volume) ,DeltaPressure)
#perr = np.sqrt(np.diag(parameter_covariance))

print ("Optimized parameters are:")
print (results.params['Psi_1'].value, results.params['Psi_2'].value)
print (results.params['Psi_1'].stderr, results.params['Psi_2'].stderr)
with open('DA_Parameters.out','w') as pout: # write out the coefficients
	for index,param in enumerate(results.params):
		pout.write("Parameter {} is: {} +/- {} \n".format(index,param,results.params[param].stderr))
		
import matplotlib.pyplot as plt
plt.scatter(AA_Volume, DeltaPressure, s=40, facecolors='k')
plt.scatter(AA_Volume, ydata, s=80, facecolors='none', edgecolors='r')
#plt.show()
plt.savefig("PressureCorrection.png",bbox_inches='tight')
plt.close()


#[AA_Vol_hist,AA_Vol_bins] = np.histogram(AA_Volume, 100, density=False)
#[CG_Vol_hist,CG_Vol_bins] = np.histogram(CG_Volume, 100, density=False)
#AA_Vol_bins = AA_Vol_bins[:-1]
#CG_Vol_bins = CG_Vol_bins[:-1]

#CG_Vol_hist_out = np.column_stack((CG_bins,CG_hist))
#AA_Vol_hist_out = np.column_stack((AA_bins,AA_hist))

