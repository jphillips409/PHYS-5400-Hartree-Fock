"""
Author: Johnathan Phillips
Email: j.s.phillips@wustl.edu
Date: Oct 21 2025

School: Washington University in St. Louis
Class: PHYS 5400 - Many Body Quantum Mechanics

Assignment:
    Calculate the energies of occupied states in He, Ne, and Ar using the Hartree-Fock method.
"""
import copy

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
import scipy.special
import scipy.interpolate
from scipy import sparse
from scipy.sparse import linalg as sla
from scipy.linalg import eigh
from fractions import Fraction
import sys

#Plot style
plt.rcParams['xtick.direction'] = 'in'  # Ticks pointing inwards on x-axis
plt.rcParams['ytick.direction'] = 'in'  # Ticks pointing inwards on y-axis

plt.rcParams['xtick.top'] = 'True'  # Ticks pointing inwards on x-axis
plt.rcParams['ytick.right'] = 'True'  # Ticks pointing inwards on x-axis

plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

plt.rcParams['figure.constrained_layout.use'] = True

plt.rcParams['legend.fontsize'] = 14

plt.rcParams['lines.linewidth'] = 2

#Defining constants for this problem
#H for checking calculation
#Z = 1 # Number of protons
#Ratm = 10 # angstrom
#screen = 0 # Electron screening parameter, 0 for hydrogen

#He
#Z = 2 # Number of protons
#Ratm = 10 # Bohr radii
#screen = 4.05#2.97 # Electron screening parameter, tuned for He
#screen = 1#1.2

#Be
#Z = 4
#Ratm = 10
#screen = 0

#Ne
Z = 10
Ratm = 10 # angstrom
#screen = 4.385
screen = 1

#Ar
#Z = 18
#Ratm = 15 # angstrom
#screen = 3.977
#screen = 0

CoulConst = 1 # N^2*m^2/C^2

#General constants
hbar = 1 # keV*A/c
me = 1  # electron mass in keV/c^2
#Starting parameters
rstart = 0.00000001 # Bohr radii
rstop = Ratm # Bohr radii
riter = 2000 # Number of steps. 2000 for He
#riter = 3000 # 3k for Ne
lorbit = 0 #orbital angular momentum

#Gives the constant for the kinetic energy
def KEconst(m):
    #return -(hbar**2)/(2*m)
    return -1/2

#Regular Coulomb potential, calcs in j and returns in Hartrees
def Coul(r,Z):
    #return -CoulConst*((Z*1.602176634*(10**-19))*(1.602176634*(10**-19)))/(r*(10**(-10)))*6.242*(10**15)
    return -Z/r

#Auxiliary Coulomb potential including a screening term, calcs in j and returns in keV
def AuxCoul(r,Z,a):
    Zeff = 1 + (Z-1)*np.exp(-a*r)
    #return -CoulConst*((Zeff*1.602176634*(10**-19))*(1.602176634*(10**-19)))/(r*(10**(-10)))*6.242*(10**15)
    return -Zeff/r

# Centrifugal component
def Centf(r, l, m):
    #return l * (l + 1) * ((hbar ** 2) / (2 * m * (r ** 2)))
    return l * (l + 1)/ (2 *  (r ** 2))

# Total potential
def PotTot(r, Z, l, m):
    return Coul(r, Z) + Centf(r, l, m)

# Total auxiliary potential
def AuxPotTot(r, Z, l, m, a):
    return AuxCoul(r, Z, a) + Centf(r, l, m)

# Returns values of 3 J notation after coupling two orbital angular momenta
def ThreeJ(l,lprime,lcoup):
    value = (2*lprime + 1)

    # Calculate sum of ls
    sumL = int(l + lprime + lcoup)
    # Because all m = 0, return 0 if odd
    if sumL%2 != 0: return 0

    # Only so many values because we go up to l = 1
    # Pre-calculate and access them
    # Need 3J ^ 2
    if l == 0 and lprime == 0 and lcoup == 0: return 1*value

    if l == 1 and lprime == 1 and lcoup == 0: return (1/3)*value

    if l == 1 and lprime == 0 and lcoup == 1: return (1/3)*value

    if l == 0 and lprime == 1 and lcoup == 1: return (1/3)*value

    if l == 0 and lprime == 1 and lcoup == 0: return (1/3)*value

    if l == 1 and lprime == 1 and lcoup == 2: return (2/15)*value

    # If here but no value, print failure and abort
    print('No valid 3J value for : ', l, ' ', lcoup, ' ', lprime)
    sys.exit()

# Calculates the orbitals from the auxiliary potential
# Returns the wavefunctions, their eigenvalues, l values, and n values
def Auxiliary(Z,lorbit,screenparameter,plotb):
    # Plots the potential
    x = np.linspace(rstart, rstop, riter)

    plt.plot(x, AuxCoul(x, Z, screenparameter))
    plt.plot(x, Centf(x, lorbit, me))
    plt.plot(x, AuxPotTot(x, Z, lorbit, me, screenparameter))
    plt.legend(['Coulomb Potential', 'Centrifugal Component', 'Total Potential'])

    plt.xlim(0, 10)
    plt.ylim(-30, 5)

    plt.xlabel('Radius ($a_{0}$)')
    plt.ylabel('Potential Energy (Hartree)')
    plt.axhline(0, color='black')
    if plotb == True: plt.savefig('auxPotential.pdf')
    if plotb == True: plt.show()
    # Sets up Hamiltonian matrix

    rstep = (rstop - rstart) / riter

    # sets up value arrays for radius and potential
    r = x
    potvaltot = np.zeros(riter)

    # fills radius and potential value arrays
    for i in range(riter):
        if i == 0:
            r[i] = (1 / 2) * rstep
        else:
            r[i] = r[i - 1] + rstep
        potvaltot[i] = AuxPotTot(r[i], Z, lorbit, me, screenparameter)

    # Defines Hamiltonian matrix for the Woods-Saxon potential
    Hamil = np.zeros((riter,riter))

    Hamil[0, 1] = KEconst(me) * (1.0 / (rstep ** 2))
    if lorbit % 2 == 0: Hamil[0, 0] = KEconst(me) * (-3.0 / (rstep ** 2)) + potvaltot[0]  # Different values if l even or odd
    if lorbit % 2 != 0: Hamil[0, 0] = KEconst(me) * (-1.0 / (rstep ** 2)) + potvaltot[0]

    # Fills rest of tridiagonal
    for i in range(1, riter - 1):
        Hamil[i, i - 1] = KEconst(me) * (1.0 / (rstep ** 2))
        Hamil[i, i] = KEconst(me) * (-2.0 / (rstep ** 2)) + potvaltot[i]
        Hamil[i, i + 1] = KEconst(me) * (1.0 / (rstep ** 2))

    Hamil[riter - 1, riter - 2] = KEconst(me) * (1.0 / (rstep ** 2))
    Hamil[riter - 1, riter - 1] = KEconst(me) * (-2.0 / (rstep ** 2)) + potvaltot[riter - 1]

    # Solves for the eigenvalues and eigenstates of the Woods-Saxon Hamiltonian
    EigValues, EigVectors = np.linalg.eig(Hamil)

    for i in range(0,riter):
        normalization_const = np.sqrt(np.trapz(EigVectors.T[i]**2, r))
        EigVectors[:,i] = EigVectors.T[i] / normalization_const

    # These are unsorted so they need to be sorted by energy
    permute = EigValues.argsort()
    EigValues = EigValues[permute]
    EigVectors = EigVectors[:, permute]

    # Check normalization for i = 0
    sum = 0
    vector = EigVectors[:,0]
    for i in range(riter):
        sum += rstep*vector[i]*vector[i]
    print('Vector sum u**2: ', sum)

    # Prints out the negative eigenvalues and counts them
    boundEig = 0
    Earr=[]
    larr=[]
    narr=[]
    #print('Bound states for l = ', lorbit)
    for i in range(0, len(r)):
        #Find bound states
        #if EigValues[i] < 0 and EigValues[i] > -1000: print(EigValues[i])
        if EigValues[i] < 0: boundEig += 1
        if EigValues[i] >= 0 and EigValues[i] > -1000: break  # don't waste time once you get into the continuum

    # Solves for the radial wavefunction R = u/r, only takes negative eigenvalues

    Rfunc=[]
    Rdistfunc=[]
    BoundVectors=[]

    # Assigns the eigenvectors into a radial distribution function array
    wavecount = 0
    for i in range(0, boundEig):
        if EigValues[i] < 0 and EigValues[i] > -1000:
            BoundVectors.append(EigVectors[:,i])
            Rfunc.append(EigVectors[:, i])
            Rfunc[wavecount] = Rfunc[wavecount] / r
            Rdistfunc.append(EigVectors[:, i])

            Earr.append(EigValues[i])
            larr.append(lorbit)
            narr.append(wavecount + 1 + lorbit)

            wavecount += 1
        elif EigValues[i] >= 0: break

    BoundVectors = np.array(BoundVectors)
    BoundVectors = np.squeeze(BoundVectors)
    Rfunc = np.array(Rfunc)
    Rfunc = np.squeeze(Rfunc)
    Rdistfunc = np.array(Rdistfunc)
    Rdistfunc = np.squeeze(Rdistfunc)

    #print('First Auxiliary radial functions: ', Rfunc[0])
    # Plot radial function
    plt.axhline(0, color='black', label='_nolegend_')
    plt.xlabel('Radius (a$_{0}$)')
    plt.ylabel('rR(r)')
    plt.xlim(0,10)

    for i in range(0, wavecount):
        if lorbit == 0: Rlabel = str(i+1) + 's'
        if lorbit == 1: Rlabel = str(i+2) + 'p'
        if lorbit == 2: Rlabel = str(i+3) + 'd'
        if lorbit == 3: Rlabel = str(i+4) + 'f'
        if lorbit == 4: Rlabel = str(i+5) + 'g'
        if lorbit == 5: Rlabel = str(i+6) + 'h'
        if lorbit == 6: Rlabel = str(i+7) + 'i'
        if lorbit == 7: Rlabel = str(i+8) + 'j'

        if i == 2:
            plt.plot(r, r*Rfunc[i], label=Rlabel + '*10')
        elif i == 3:
            plt.plot(r, r*Rfunc[i], label=Rlabel + '*20')
        elif i == 4:
            plt.plot(r, r*Rfunc[i], label=Rlabel + '*20')
        else:
            plt.plot(r, r*Rfunc[i], label=Rlabel)
        #plt.plot(r, Rfunc[i], label=Rlabel)


    plt.legend()
    if plotb == True: plt.savefig('auxradial.pdf')
    if plotb == True: plt.show()

    # Plots each radial distribution function squared
    plt.axhline(0, color='black', label='_nolegend_')
    plt.xlabel('Radius (a$_{0}$)')
    plt.ylabel('|rR(r)|$^2$')

    for i in range(0, wavecount):
        if lorbit == 0: Rlabel = str(i+1) + 's'
        if lorbit == 1: Rlabel = str(i+2) + 'p'
        if lorbit == 2: Rlabel = str(i+3) + 'd'
        if lorbit == 3: Rlabel = str(i+4) + 'f'
        if lorbit == 4: Rlabel = str(i+5) + 'g'
        if lorbit == 5: Rlabel = str(i+6) + 'h'
        if lorbit == 6: Rlabel = str(i+7) + 'i'
        if lorbit == 7: Rlabel = str(i+8) + 'j'
        if i == 2:
            plt.plot(r, (Rdistfunc[i] ** 2)*5, label=Rlabel + '*5')
        elif i == 3:
            plt.plot(r, (Rdistfunc[i] ** 2)*5, label=Rlabel + '*5')
        elif i == 4:
            plt.plot(r, (Rdistfunc[i] ** 2)*5, label=Rlabel + '*5')
        else:
            plt.plot(r, Rdistfunc[i] ** 2, label=Rlabel)
        #plt.plot(r, Rdistfunc[i] ** 2, label=Rlabel)

    plt.legend(loc="upper right")
    if plotb == True: plt.savefig('auxradialprob.pdf')
    if plotb == True: plt.show()
    #print('First Aux radial functions at end of Aux: ', Rfunc[0])
    return Earr, larr, narr, BoundVectors, potvaltot

# Direct component.
# Accepts radial, reduced wave function, and orbital angular momentum arrays.
# The wave array is a 2D array that holds each wave function
# Needs the quantum number and correct l to access the right wave function in modified hartree
def Direct(radarr,larr,narr, wavearr, n, l):

    # Switch for modified
    modified = False
    print('Calc for n = ', n, ' and l = ',l)
    print('Direct wave')
    print(wavearr)
    directpot = np.zeros(len(radarr))
    ndirect = np.zeros(len(radarr))
    integrand = np.zeros(len(radarr))
    # Loop over radius
    for i in range(0,len(radarr)):
        # Calculate electron density using each wave function
        for j in range(0,len(wavearr)):
            wavepicked = wavearr[j]
            lpicked = larr[j]
            npicked = narr[j]
            density = 2*(2*lpicked + 1) * wavepicked[i]*wavepicked[i]
            if modified == True and lpicked == l and npicked == n:
                if l == 0: density = (0.5)*density # s-wave
                if l == 1: density = (5/6)*density # p-wave
            ndirect[i] += density

        integrand[i] = ndirect[i] # Don't need to multiply by r^2 because I'm using the reduced
    # Calculate the integrand for each radial point. Only need to be done once
    # Loops over the radius again
    for i in range(0,len(radarr)):

        # Find max value array
        maxarr = np.zeros(len(radarr))
        for j in range(0,len(radarr)):
            if radarr[j] <= radarr[i]: maxarr[j] = radarr[i]
            else: maxarr[j] = radarr[j]

        # Perform a trapezoidal integral
        directpot[i] = np.trapz(integrand/maxarr,x=radarr)
    #print('direct potential ', directpot)
    return directpot

# Exchange component
# Need to do this for each occupied wave function
# Based on eq. 10.96 in the book, but in r-space
# Need to provide the l for the HF matrix
def Exchange(radarr,larr,wavearr,lHF):

    # Form a matrix of r x r
    epot = np.zeros((len(radarr),len(radarr)))
    DeltR = radarr[2]-radarr[1]
    # Iterate over r
    for i in range(0,len(radarr)):

        # Iterate over r'
        for j in range(0,len(radarr)):
            matrixel = 0

            # Iterate over the wave functions n' l'
            for w in range(len(wavearr)):
                wavepicked = wavearr[w]
                lpicked = larr[w]

                # Find the smallest and largest L (coupled l), iterate to this
                lmin = abs(lpicked - lHF)
                lmax = abs(lpicked + lHF)

                # Iterate over L
                for lcoup in range(lmin,lmax+1):

                    # Calculate 3J
                    C = ThreeJ(lHF, lpicked, lcoup)
                    if C == 0:
                        # Do not contribute to term if 0
                        continue
                    #Need a DeltR?
                    epot[i,j] += DeltR*wavepicked[i]*((min(radarr[i],radarr[j])**lcoup)/(max(radarr[i],radarr[j])**(lcoup+1)))*C*wavepicked[j]

    return epot

def main(Z,lorbit,screenparameter,plotb):

    # First calculate the orbitals from an auxiliary potential
    # Loop over desired orbital angular momenta
    lmax = 0
    if Z == 10 or Z == 18: lmax = 1

    auxE = []
    auxl = []
    auxn = []
    auxstates = []
    potentials = []
    for i in range(0,lmax+1):
        lorbit = i
        tE, tl, tn, tstates, tpot = Auxiliary(Z,lorbit,screenparameter,plotb)
        auxE.extend(tE)
        auxl.extend(tl)
        auxn.extend(tn)
        for sub in tstates:
            auxstates.append(sub)
        for j in range(len(tn)):
            potentials.append(tpot)

    # Probably very inefficient but I'm not very familiar with numpy
    auxstates = np.array(auxstates)
    auxstates = np.squeeze(auxstates)
    #print('auxstates ', auxstates)
    potentials = np.array(potentials)
    potentials = np.squeeze(potentials)
    lookupl = ['s', 'p', 'd', 'f', 'g', 'h', 'i', 'j']

    #print('Auxiliary energy array: ', auxE)
    #print('Auxiliary orb ang mom array: ', auxl)
    #print('Auxiliary quant num array: ', auxn)

    ############################################################
    # Begin HF part of the calculation
    # Change to u = ln(r) for the grid points
    rstep = (rstop - rstart) / riter
    r = np.linspace(rstart, rstop, riter)
    for i in range(riter):
        if i == 0:
            r[i] = (1 / 2) * rstep
        else:
            r[i] = r[i - 1] + rstep

    # Iterate until the change in energy is of 0.01 eV
    Esig = 0.01

    # The kinetic energy and external potential components only need to be calculated once for each l
    # The Hartree and Fock terms require previously solved wave functions

    # Define shape of Hamiltonian
    # The l value for each wavefunction is saved. Use this to access the correct Hamiltonian

    # Create flat array to store matrices
    Hamil = np.empty(lmax+1,dtype=object)

    for i in range(0,lmax+1):

        # Calculate tridiagonal kinetic energy matrix
        tHamil = np.zeros((riter, riter))
        tHamil[0, 1] = KEconst(me) * (1/ ((rstep**2)))
        if i % 2 == 0: tHamil[0, 0] = KEconst(me) * (-3.0 /((rstep**2))) # Different values if l even or odd
        if i % 2 != 0: tHamil[0, 0] = KEconst(me) * (-1.0/((rstep**2)))

        # Fills rest of tridiagonal
        for j in range(1, riter - 1):
            tHamil[j, j - 1] = KEconst(me) * (1/((rstep**2)))
            tHamil[j, j] = KEconst(me) * (-2.0/((rstep**2)))
            tHamil[j, j + 1] = KEconst(me) * (1/((rstep**2)))

        tHamil[riter - 1, riter - 2] = KEconst(me) * (1/((rstep**2)))
        tHamil[riter - 1, riter - 1] = KEconst(me) * (-2.0/((rstep**2)))

        Hamil[i] = tHamil

    # Counter for recursive loop
    counter = 0
    maxcount = 15 # set some max number of iterations

    # Temp arrays
    TE = []
    Tl = []
    Tn = []
    Tstates = []
    Tpots = []

    # Make arrays that will hold each set of values
    # Multidimensional, each hold an object which contains every array
    HFE = np.empty(maxcount+1,dtype=object)
    HFl = np.empty(maxcount+1,dtype=object)
    HFn = np.empty(maxcount+1,dtype=object)
    HFstates = np.empty(maxcount+1,dtype=object)
    HFpots = np.empty(maxcount+1,dtype=object)


    # Set first value equal to auxiliary arrays
    # Only use the occupied states! This is very important!
    for i in range(len(auxstates)):

        # Only 1s occupied
        if Z == 1:
            if auxl[i] == 0 and auxn[i] == 1:
                TE.append(auxE[i])
                Tl.append(auxl[i])
                Tn.append(auxn[i])
                Tstates.append(auxstates[i])
                Tpots.append(potentials[i])
            else: continue

        # Only 1s occupied
        if Z == 2:
            if auxl[i] == 0 and auxn[i] == 1:
                TE.append(auxE[i])
                Tl.append(auxl[i])
                Tn.append(auxn[i])
                Tstates.append(auxstates[i])
                Tpots.append(potentials[i])
            else: continue

            # Only 1s and 2s occupied
        if Z == 4:
            if auxl[i] == 0 and auxn[i] > 2: continue
            else:
                TE.append(auxE[i])
                Tl.append(auxl[i])
                Tn.append(auxn[i])
                Tstates.append(auxstates[i])
                Tpots.append(potentials[i])


        # Only 1s, 2s, and 2p occupied
        if Z == 10:
            if auxl[i] == 0 and auxn[i] > 2: continue
            elif auxl[i] == 1 and auxn[i] > 2: continue
            else:
                TE.append(auxE[i])
                Tl.append(auxl[i])
                Tn.append(auxn[i])
                Tstates.append(auxstates[i])
                Tpots.append(potentials[i])

        # Only 1s, 2s, 2p, 3s, 3p are occupied
        if Z == 18:
            if auxl[i] == 0 and auxn[i] > 3: continue
            elif auxl[i] == 1 and auxn[i] > 3: continue
            else:
                TE.append(auxE[i])
                Tl.append(auxl[i])
                Tn.append(auxn[i])
                Tstates.append(auxstates[i])
                Tpots.append(potentials[i])

    # Make a diagonal matrix for the potential
    Tpots = np.array(Tpots)
    ogpot = np.zeros((riter,riter))
    Tmatrix = []
    for inner in Tpots:
        for i in range(riter):
            for j in range(riter):
                if i == j: ogpot[i,j] = inner[i]
        Tmatrix.append(ogpot)
    Tmatrix = np.array(Tmatrix)
    #Tmatrix = np.squeeze(Tmatrix)

    HFE[0] = TE
    HFl[0] = Tl
    HFn[0] = Tn
    Tstates = np.array(Tstates)
    #Tstates = np.squeeze(Tstates) # Not sure if I need this
    HFstates[0] = Tstates
    HFpots[0] = Tmatrix

    # Reorder arrays to follow n,l scheme. Only for Z > 10
    # Manually do it for argon
    if Z == 18:
        rindx = np.array([0,1,3,2,4])
        temp = HFE[0]
        temp = np.array(temp)
        HFE[0] = temp[rindx]
        temp = HFn[0]
        temp = np.array(temp)
        HFn[0] = temp[rindx]
        temp = HFl[0]
        temp = np.array(temp)
        HFl[0] = temp[rindx]
        temp = HFstates[0]
        temp = np.array(temp)
        HFstates[0] = temp[rindx]
        temp = HFpots[0]
        temp = np.array(temp)
        HFpots[0] = temp[rindx]

    print('HF Input Values')
    print(HFE[0])
    print(HFl[0])
    print(HFn[0])
    #print(HFpots[0])

    #print('First input wave function for HF: ',HFstates[0])

    # need to access the n for the calculation
    maxn = 0
    if Z == 1 or Z == 2: maxn = 1
    if Z == 4: maxn = 2
    if Z == 10: maxn = 2
    if Z == 18: maxn = 3

    # Enter recursive loop for HF calculation
    while counter < maxcount:

        # Access objects used to calculation
        energies = HFE[counter]
        lvalues = HFl[counter]
        nvalues = HFn[counter]
        states = HFstates[counter]
        #pots = HFpots[counter]

        print('input states: ', states)

        counter += 1

        # Need these for l > 0
        tempstates = []
        tempE = []
        tempn = []
        templ = []
        temppots = []

        for n in range(1,2):

            # Form matrix to diagonalize
            maxl = 0
            if Z > 4: maxl = 1

            for i in range(0,maxl+1):

                # Select potential based on n,l
                potindx = 0
                for j in range(len(nvalues)):
                    if lvalues[j] == i: potindx = j

                #prvpot = copy.deepcopy(pots[potindx])
                potvaltot = np.zeros(riter)
                # fills radius and potential value arrays
                for j in range(riter):
                    potvaltot[j] = PotTot(r[j], Z, i, me)

                # Calculate the direct and exchange terms, n, l dependent for modified hartree
                Dpot = Direct(r, lvalues, nvalues, states, n, i)
                #potvaltot = potvaltot + Dpot
                addedpot = potvaltot + Dpot

                # Mix with counter - 1 potential, give mixing parameter
                #avgpot = (1 - mix)*prvpot + (mix)*potvaltot
                #print(avgpot)
                # Plot the Hartree potential
                plt.plot(r, potvaltot,label='Original Pot')
                plt.plot(r, Dpot,label='Direct Pot')
                plt.plot(r, addedpot,label='New Potential')

                #plt.plot(r, avgpot,label='Averaged Potential')
                plt.xlim(0, 10)
                plt.ylim(-30, 30)

                plt.xlabel('Radius ($a_{0}$)')
                plt.ylabel('Potential Energy (Hartree)')
                plt.axhline(0, color='black')
                plt.legend()
                plt.show()

                Epot = Exchange(r, lvalues, states, i)

                # Generate new r x r matrix with pot, direct, and exchange
                # Mix this matrix with the [i-1] total potential matrix
                #avgpot = np.zeros((riter, riter))
                mix = 1
                #for j in range(0, riter):
                    #for m in range(0, riter):
                        #if j == m: avgpot[j,m] = (1-mix)*prvpot[j,m] + mix*(potvaltot[j] - Epot[j,m])
                        #else: avgpot[j,m] = (1-mix)*prvpot[j,m] - mix*Epot[j,m]

                # Access the correct kinetic and external Hamiltonian using the l value
                # MUST BE A DEEP COPY
                # Otherwise the original Hamil will be updated in each iteration
                HFHamil = copy.deepcopy(Hamil[i])

                # Add direct and exchange terms to Hamiltonian
                for j in range(0, riter):
                    for m in range(0, riter):
                        if j == m: HFHamil[j,m] += addedpot[j]
                        HFHamil[j,m] -= Epot[j,m]

                # Diagonalize the Hamiltonian, might not be symmetric so don't use eigh
                HFValues, HFVectors = np.linalg.eig(HFHamil)

                permute = HFValues.argsort()
                HFValues = HFValues[permute]
                HFVectors = HFVectors[:, permute]

                for norm in range(0, riter):
                    normalization_const = np.sqrt(np.trapz(HFVectors.T[norm] ** 2, r))
                    HFVectors[:, norm] = HFVectors.T[norm] / normalization_const

                sum = 0
                vector = HFVectors[:, 0]
                for inorm in range(riter):
                    sum += rstep * vector[inorm] * vector[inorm]
                print('Vector sum u**2: ', sum)

                boundEig = 0
                # Find number of bound states
                for j in range(0, len(r)):
                    if HFValues[j] < 0: boundEig += 1
                    if HFValues[j] >= 0: break  # don't waste time once you get into the continuum

                # Solves for the radial wavefunction Rr, only takes negative eigenvalues
                boundvals = []
                boundvec = []

                # Assigns the eigenvectors into a radial distribution function array
                wavecount = 0
                for j in range(0, boundEig):
                    if HFValues[j] < 0 and HFValues[j] > -1000:
                        boundvec.append(HFVectors[:, j])
                        boundvals.append(HFValues[j])
                        wavecount += 1
                    elif HFValues[j] >= 0:
                        break
                #boundvec = np.squeeze(boundvec)

                if wavecount == 0:
                    if (i < 1):
                        print('No bound values, something is wrong')
                        sys.exit()
                    else:
                        boundvec.append(HFVectors[:, 0])
                        boundvals.append(0)
                        boundvec.append(HFVectors[:, 1])
                        boundvals.append(0)
                        boundvec.append(HFVectors[:, 2])
                        boundvals.append(0)
                        boundvec.append(HFVectors[:, 3])
                        boundvals.append(0)

                boundvec = np.array(boundvec)

                print("Bound HF eigenvalues: ", boundvals)
                #print("Bound HF eigenvectors: ", boundvec)

                plt.axhline(0, color='black', label='_nolegend_')
                plt.xlabel('Radius (a$_{0}$)')
                plt.ylabel('R(r)')
                for w in range(len(boundvec)):
                    plt.plot(r, boundvec[w],label=str(w))
                #plt.xlim(0, 2)
                plt.legend()
                plt.show()

                #plt.axhline(0, color='black', label='_nolegend_')
                #plt.xlabel('Radius (a$_{0}$)')
                #plt.ylabel('|rR(r)|$^2$')
                #plotvect = boundvec[0]
                #plt.plot(r, plotvect * plotvect)
                #plotvect = (HFVectors[1])
                #plt.plot(r, plotvect * plotvect)
                #plt.xlim(0, 2)
                #plt.legend()
                #plt.show()

                # Print the first few eigenvalues
                #"HF E 0: ", boundvals[0])
                #print("HF E 1: ", boundvals[1])
                #print("HF E 2: ", boundvals[2])
                #print("HF E 3: ", boundvals[3])

                # Only save the valence orbitals as radial wavefunctions
                if Z == 2 or Z == 1:
                    Radfunc = boundvec[0]
                    HFE[counter] = [boundvals[0]]
                    HFstates[counter] = np.array([Radfunc])
                    HFl[counter] = [i]
                    HFn[counter] = [n]
                    #HFpots[counter] = np.array([avgpot])

                    print('Iteration ', counter)
                    print('States: E ', HFE[counter])
                    print('States: l ', HFl[counter])
                    print('States: n ', HFn[counter])
                    print('Pots: ', HFpots[counter])
                    #print('Radial funcs: ', HFstates[counter])

                # Only save the valence orbitals as radial wavefunctions
                if Z == 4:
                    tempstates.append(boundvec[0])
                    tempstates.append(boundvec[1])
                    tempE.append(boundvals[0])
                    tempE.append(boundvals[1])
                    templ.append(i)
                    templ.append(i)
                    tempn.append(n)
                    tempn.append(n+1)
                    #temppots.append(avgpot)

                    #print('Iteration ', counter)
                    #print('States: E ', HFE[counter])
                    #print('States: l ', HFl[counter])
                    #print('States: n ', HFn[counter])
                    #print('Radial funcs: ', HFstates[counter])

                if Z == 10 or Z == 18:
                    # Save s-wave
                    if i == 0:
                        tempstates.append(boundvec[0])
                        tempstates.append(boundvec[1])
                        tempE.append(boundvals[0])
                        tempE.append(boundvals[1])
                        templ.append(i)
                        templ.append(i)
                        tempn.append(n)
                        tempn.append(n + 1)
                        #temppots.append(avgpot)

                        print('Temp Stuff')
                        print('states: ', tempstates)
                        print('E: ', tempE)
                        print('l: ', templ)
                        print('n: ', tempn)
                    # Save p-wave
                    if i == 1:
                        tempstates.append(boundvec[0])
                        tempE.append(boundvals[0])
                        #tempstates.append(states[2])
                        #tempE.append(energies[2])
                        templ.append(i)
                        tempn.append(n+1)
                        #temppots.append(avgpot)

                        print('Temp Stuff')
                        print('states: ', tempstates)
                        print('E: ', tempE)
                        print('l: ', templ)
                        print('n: ', tempn)

        if Z == 10 or Z == 18 or Z == 4:
            tempstates = np.array(tempstates)
            tempstates = np.squeeze(tempstates)

            temppots = np.array(temppots)
            temppots = np.squeeze(temppots)

            HFE[counter] = tempE
            HFstates[counter] = tempstates
            HFl[counter] = templ
            HFn[counter] = tempn
            HFpots[counter] = temppots

            print('Iteration ', counter)
            print('States: E ', HFE[counter])
            print('States: l ', HFl[counter])
            print('States: n ', HFn[counter])
            print('Pots ', HFpots[counter])
            # print('Radial funcs: ', HFstates[counter])

    print('Final Arrays')
    energylist = []
    count = []
    c = 0
    energylist2s = []
    energylist2p = []
    energylist3s = []
    energylist3p = []


    for inner in HFE:
        print('iteration ', inner, ' E (H): ', inner[0])
        energylist.append(inner[0])
        if maxn == 2:
            energylist2s.append(inner[1])
            if Z > 4: energylist2p.append(inner[2])
        if maxn == 3 and Z == 18:
            energylist2s.append(inner[1])
            energylist2p.append(inner[2])
            energylist3s.append(inner[3])
            energylist3p.append(inner[4])
        count.append(c)
        c+=1
    energylist = np.array(energylist)
    energylist2s = np.array(energylist2s)
    energylist2p = np.array(energylist2p)
    energylist3s = np.array(energylist3s)
    energylist3p = np.array(energylist3p)


    if maxn == 1: plt.plot(count, energylist,label = '1s')
    if maxn == 2:
        plt.plot(count, energylist2s,label = '2s')
        if Z > 4: plt.plot(count, energylist2p,label = '2p')
    if maxn == 3 and Z == 18:
        plt.plot(count, energylist/10,label = '1s x0.1')
        plt.plot(count, energylist2s,label = '2s')
        plt.plot(count, energylist2p,label = '2p')
        plt.plot(count, energylist3s,label = '3s')
        plt.plot(count, energylist3p,label = '3p')

    plt.xlabel('Iteration')
    plt.ylabel('Energy (Hartree)')
    plt.axhline(0, color='black')
    plt.legend()
    plt.show()

    # Plot the starting states and the final states
    statecount = 0
    valstate = []
    collist = ['b','g','r','c','m','y','k']
    for inner in HFstates:
        if statecount == 0 or statecount == len(HFstates) - 1:
            for i in range(len(inner)):
                if statecount == 0:
                    slabel = str(HFn[0][i]) + lookupl[HFl[0][i]]
                    plt.plot(r,inner[i], linestyle='dashed',color=collist[i],label=slabel)
                    valstate.append(inner[i,3])
                if statecount == len(HFstates) - 1:
                    flip = 1
                    if (inner[i,3]/valstate[i]) < 0: flip = -1
                    plt.plot(r, flip*inner[i],color=collist[i])

        statecount +=1

    plt.xlabel('Radius $a_{0}$')
    plt.ylabel('rR(r)')
    plt.axhline(0, color='black')
    plt.xlim(0, 1)
    plt.ylim(-3.5, 3.5)
    plt.legend()
    plt.show()

main(Z, 0, screen, True)

