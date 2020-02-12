#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#                    PHYS 730 FINAL PROJECT
#                        KHAGENDRA ADHIKARI
#                           SPRING 2016
#
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# This program generates the Hamiltonian matrix of "Frustrated Quantum Heisenburg
# Spin Chain " where the interactions are limited to next_nearest_neighbor sites
# and solves for the ground state energy over a range of tuning parameter,
# g=J_2/J_1 the ratio of next_nearest_neighbor to nearest_neighbor interaction.

import numpy as np
import scipy.sparse as ssp
import scipy.sparse.linalg as ssl
import matplotlib.pyplot as plt

MS = 2 # MS=2S+1=2*(1/2) + 1 = 2

# 2 by 2 Spin matrices stored in Compressed Sparse Row matrix format.  
Sigz = ssp.csr_matrix([[0.5,0.0],[0.0,-0.5]]) # S^z matrix
SigP = ssp.csr_matrix([[0.0,1.0],[0.0,0.0]])  # S^+ matrix
SigM = ssp.csr_matrix([[0.0,0.0],[1.0,0.0]])  # S^- matrix
Id = ssp.csr_matrix(np.identity(MS))          # I = Identity matrix

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
def spin(g):
    '''
        Input:     g = tuning parameter J_2/J_1
        Method:    Kroneckor product
        Output:    Hn=matrix due to nearest_neighbor interaction 
                   Hnn=matrix due to next_nearest_neighbor interaction
    '''
    Hn = (0.5*(ssp.kron(SigP,SigM,format='csr')+ssp.kron(SigM,SigP,format='csr'))
    +ssp.kron(Sigz,Sigz,format='csr'))
    
    Hnn = (0.5*g*(ssp.kron(ssp.kron(SigP,Id,format='csr'),SigM,format='csr')
    +ssp.kron(ssp.kron(SigM,Id,format='csr'),SigP,format='csr'))
    +g*ssp.kron(ssp.kron(Sigz,Id,format='csr'),Sigz,format='csr'))
    
    return Hn,Hnn
    
    
#*******************************************************************************    
def Hamiltonian(N,g):
    '''
        Inputs:    N = Numer of sites (Spins)
                   g = tuning parameter J_2/J_1
        Method:    Kronecker product
        Output:    H = Hamiltonian matrix that stores only non-zero matrix
                       elements in Compressed Sparse Row matrix format.   
    '''
    Hn,Hnn=spin(g)
    SpId = ssp.csr_matrix(np.identity(MS))
    Sz = MS**N
    H = ssp.csr_matrix((Sz,Sz))
    temp1 = dict()
    temp2 = dict()
    for i in range(N-1):
        for j in range(N-1):
            if i == j:
                temp1[(i,j)]=Hn
            else:
                temp1[(i,j)]=SpId
    Ha = ssp.csr_matrix((8,8))
    for i in range(N-2):
        for j in range(N-2):
            if i == j:
                temp2[(i,j)]=Hnn
            else:
                temp2[(i,j)]=SpId
    Hb = ssp.csr_matrix((16,16))
    try:
        for i in range(N-1):
            for j in range(N-2):
                if j < 1:
                    Ha = ssp.kron(temp1[(i,j)],temp1[(i,j+1)],format='csr')
                else:
                    Ha = ssp.kron(Ha,temp1[(i,j+1)],format='csr')
            H = H + Ha
        if N > 3:
            for i in range(N-2):
                for j in range(N-3):
                    if j < 1:
                        Hb = ssp.kron(temp2[(i,j)],temp2[(i,j+1)],format='csr')
                    else:
                        Hb = ssp.kron(Hb,temp2[(i,j+1)],format='csr')
                H = H + Hb
        else:
            Hb = ssp.csr_matrix(temp2[(0,0)])
            H = H + Hb
    except MemoryError:
        print('The matrix you tried to build requires too much memory space.')
        return
    return H
    
#*******************************************************************************
def Energy(N):
    '''
        Input:  N = Numer of sites (Spins)
        Output: List of ground state energy per site
    '''
    E0list=[]
    for g in glist:
        h=Hamiltonian(N,g)
        Evals,Evecs=ssl.eigsh(h)
        E0list.append([Evals[0]/N])
    return E0list
    
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@    
# List of coupling tuning parameter g.
glist=np.arange(0,1,0.05)
# List of colors
Color=['r-o','b-d','m-*','g-s','k-^']  
          
plt.close('all')
# Figure # 1 
# Sparse matrix for N=6 for non-frustrated model i.e g = 0.
plt.figure(1,figsize=(3.5,3.5))
h=h=Hamiltonian(6,0.0)
plt.spy(h,marker='s',ms=2)
plt.tick_params(labelsize=10)
plt.savefig('Adhikari_Khagen_Final_Project_Fig1.pdf') 

#*******************************************************************************
# Figure # 2
# Sparse matrix for N=6 for frustrated model. 
plt.figure(2,figsize=(3.5,3.5))
H=Hamiltonian(6,0.5)
plt.spy(H,marker='s',ms=2) 
plt.tick_params(labelsize=10)
plt.savefig('Adhikari_Khagen_Final_Project_Fig2.pdf') 

#*******************************************************************************
# Figure # 3
# Ground state energy per site versus tuning parameter g for various N.
plt.figure(3,figsize=(6,5))
Nmax=14
Nmin=6
Nlist=np.arange(Nmin,Nmax+1,4)
k=0
for N in Nlist:
    E0=Energy(N)
    plt.plot(glist,E0,''+Color[k],label='N=%d'%N,lw=1.3,ms=4)
    plt.xlabel('Relative coupling $J_2/J_1$',fontsize=12)
    plt.ylabel('Ground state energy per site',fontsize=12)
    plt.tick_params(labelsize=10)
    plt.tight_layout()
    plt.legend(loc=8)
    k+=1
plt.savefig('Adhikari_Khagen_Final_Project_Fig3.pdf')

#*******************************************************************************
# Figure # 4
# plot for the extrapolation of ground state energy for g=0.
plt.figure(4,figsize=(6,5))
Nstart=10
# Maximum value of Nstop depends on memory of the computer.Here I assign small 
# value in orther to get result in short time.In my report,I assigned Nstop = 22.
# Bigger the system-size (Nstop) better the result.
Nstop=16
# List of ground state energy for g=0.  
E0g0byN=[]
# List of 1/N^2.
OnebyNsqr=[]
for N in range(Nstart,Nstop+1,2):
    h=Hamiltonian(N,0.0)
    Evals,Evecs=ssl.eigsh(h)
    E0g0byN.append(Evals[0]/N)
    OnebyNsqr.append(1.0/N**2)
    
x=OnebyNsqr
y=E0g0byN
# Polynomail fit of degree 1 is a linear fit.
m,c=np.polyfit(x,y,deg=1)
# m and c are slope and intercept of equation y=m*x+c.
poly=m*np.array(x)+c
plt.plot(x,poly,color='red',lw=1.3,label='$E_0/N$= %1.6f'%c)
plt.plot(x,y,'mo',ms=4)
plt.xlabel('Inverse of square chain length,$1/N^2$',fontsize=12)
plt.ylabel('Ground state energy per site',fontsize=12)
plt.tick_params(labelsize=10)
plt.tight_layout()
plt.legend(loc=2)
plt.savefig('Adhikari_Khagen_Final_Project_Fig4.pdf')
plt.show()

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
