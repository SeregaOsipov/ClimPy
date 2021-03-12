__author__ = 'Sergey Osipov <Serega.Osipov@gmail.com>'

import numpy as np
from scipy.sparse import spdiags, linalg

#this is the code based on the idea of http://onlinelibrary.wiley.com/doi/10.1002/2013JD020742/full
#Local partitioning of the overturning circulation in the tropics and the connection to the Hadley and Walker circulations
#Juliane Schwendike, Pallavi Govekar, Michael J. Reeder, Richard Wardle, Gareth J. Berry, Christian Jakob

class HadleyWalkerSeparationController:

    def five_pt_laplacian_sparse(self, Nx, Ny, x, y, dx, dy, r, isXBcsArePeriodic, isYBcsAreNeuman):
        e = np.ones(Nx)

        mainDiag = np.zeros(Nx*Ny)
        belowMainDiag = np.zeros(Nx*Ny)
        aboveMainDiag = np.zeros(Nx*Ny)
        aboveMainIdentity = np.zeros(Nx*Ny)
        belowMainIdentity = np.zeros(Nx*Ny)

        periodicAboveDiag = np.zeros(Nx*Ny)
        periodicBelowDiag = np.zeros(Nx*Ny)

        for yIndex in range(0, y.size):
            mainDiagItem = e * (-2*dx**-2 / np.sin(y[yIndex])**2 - 2*dy**-2)
            mainDiag[yIndex*Nx:(yIndex+1)*Nx] = mainDiagItem

            belowMainDiagItem = e * ( dx**-2 / np.sin(y[yIndex])**2 )
            belowMainDiagItem[-1] = 0
            belowMainDiag[yIndex*Nx:(yIndex+1)*Nx] = belowMainDiagItem

            aboveMainDiagItem = e * ( dx**-2 / np.sin(y[yIndex])**2 )
            aboveMainDiagItem[0] = 0
            aboveMainDiag[yIndex*Nx:(yIndex+1)*Nx] = aboveMainDiagItem

            aboveMainIdentityItem = e * (dy**-2 + np.cos(y[yIndex])/np.sin(y[yIndex])/2/dy)
            aboveMainIdentity[yIndex*Nx:(yIndex+1)*Nx] = aboveMainIdentityItem

            belowMainIdentityItem = e * (dy**-2 - np.cos(y[yIndex])/np.sin(y[yIndex])/2/dy)
            belowMainIdentity[yIndex*Nx:(yIndex+1)*Nx] = belowMainIdentityItem

            #this will impose periodic BC in x direction
            periodicAboveDiagItem = np.zeros(Nx)
            periodicAboveDiagItem[-1] = dx**-2 / np.sin(y[yIndex])**2
            periodicAboveDiag[yIndex*Nx:(yIndex+1)*Nx] = periodicAboveDiagItem

            periodicBelowDiagItem = np.zeros(Nx)
            periodicBelowDiagItem[0] = dx**-2 / np.sin(y[yIndex])**2
            periodicBelowDiag[yIndex*Nx:(yIndex+1)*Nx] = periodicBelowDiagItem

        A = spdiags([mainDiag, belowMainDiag, aboveMainDiag, belowMainIdentity, aboveMainIdentity], [0, -1, 1, -Nx, Nx], Nx*Ny, Nx*Ny)

        if ( isXBcsArePeriodic ):
            xA = spdiags([periodicBelowDiag, periodicAboveDiag], [-Nx+1, Nx-1], Nx*Ny, Nx*Ny)
            A += xA

        if ( isYBcsAreNeuman ):
            #this will impose Neumann BC for lower y boundary
            neumanTopMainDiag = np.zeros(Ny*Nx)
            neumanTopMainDiag[:Nx] = 4/3 * ( dy**-2 - np.cos(y[0])/np.sin(y[0])/2/dy )

            neumanTopAboveMainDiag = np.zeros(Ny*Nx)
            neumanTopAboveMainDiag[Nx:Nx+Nx] = -1/3 * ( dy**- 2 + np.cos(y[0])/np.sin(y[0])/2/dy)

            B = spdiags([neumanTopMainDiag, neumanTopAboveMainDiag], [0, Nx], Nx*Ny, Nx*Ny)

            #this will impose Neumann BC for upper y boundary
            neumanBottomMainDiag = np.zeros(Ny*Nx)
            neumanBottomMainDiag[-Nx:] = 4/3 * ( dy**-2 + np.cos(y[-1])/np.sin(y[-1])/2/dy )

            neumanBottomBelowMainDiag = np.zeros(Ny*Nx)
            neumanBottomBelowMainDiag[-Nx-Nx:-Nx] = -1/3 * ( dy**- 2 - np.cos(y[-1])/np.sin(y[-1])/2/dy)

            C = spdiags([neumanBottomMainDiag, neumanBottomBelowMainDiag], [0, -Nx], Nx*Ny, Nx*Ny)

            A += B+C

        A /= r**2

        return A

    #lat convention from 0 to pi (era has from -pi/2 to pi/2)
    def getLocalHadleyWalkerMassFluxes(self, fieldData, lonData, latData, isXBcsArePeriodic, isYBcsAreNeuman):

        #fix the lat convention (era has from -pi/2 to pi/2 and I need from 0 to pi)
        #make sure to copy array
        latData = latData[:] + 90

        Nx = lonData.size
        Ny = latData.size

        domainMinX = np.min(np.deg2rad(lonData))
        domainMaxX = np.max(np.deg2rad(lonData))

        domainMinY = np.min(np.deg2rad(latData))
        domainMaxY = np.max(np.deg2rad(latData))

        x = np.deg2rad(lonData)#; x = x[1:-1]
        y = np.deg2rad(latData)#; y = y[1:-1]
        X, Y = np.meshgrid(x, y)

        dx = x[1]-x[0]
        dy = y[1]-y[0]

        r = 6371*10**3 #m
        g = 9.8 #m s^-2
        #Set up and solve the linear system
        A = self.five_pt_laplacian_sparse(Nx, Ny, x, y, dx, dy, r, isXBcsArePeriodic, isYBcsAreNeuman).tocsr()
        #remember that we are solveing laplacian psi equals minus f
        F = -1*fieldData

        #BCs
        # F[:, 0] -= (np.sin(domainMinX)**2) * y * (dx**-2 / np.sin(y)**2)
        # F[:, -1] -= (np.sin(domainMaxX)**2) * y * (dx**-2 / np.sin(y)**2)
        # F[0, :] -= np.sin(x)**2 * domainMinY * (dy**-2 - np.cos(y[0])/np.sin(y[0])/2/dy)
        # F[-1, :] -= np.sin(x)**2 * domainMaxY * (dy**-2 + np.cos(y[-1])/np.sin(y[-1])/2/dy)

        F = F.reshape([Nx * Ny])

        U = linalg.spsolve(A, F)
        U = U.reshape([Ny, Nx])


        #compute the gradient of the solution in the spherical coordinates system
        psi_lambda, psi_teta = np.gradient(U, dx, dy, edge_order=2)
        psi_lambda /= -r*np.sin(Y)
        psi_teta /= -r

        #compute the derivatives
        term1, dummy = np.gradient(psi_lambda, dx, dy, edge_order=2)
        #even though paper says cos(Y), but they use -pi/2 to pi/2 angle convention
        dummy, term2 = np.gradient(psi_teta*np.sin(Y), dx, dy, edge_order=2)
        massFlux_lambda = -1*term1 / r/g
        massFlux_teta = -1*term2 / r/g

        #compute the vertical velocity for each of the circulations
        omega_lambda = -massFlux_lambda/np.sin(Y)*g
        omega_teta = -massFlux_teta/np.sin(Y)*g

        return U, massFlux_lambda, massFlux_teta, omega_lambda, omega_teta, psi_lambda, psi_teta

    #this routine CAN NOT have time dependece, only level, lon, lat dimensions
    def getLocalHadleyWalkerFluxesProfile(self, omega_data, lon_data, lat_data, isXBcsArePeriodic, isYBcsAreNeuman):
        U = np.zeros(omega_data.shape)
        massFlux_lambda = np.zeros(omega_data.shape)
        massFlux_teta = np.zeros(omega_data.shape)

        omega_lambda = np.zeros(omega_data.shape)
        omega_teta = np.zeros(omega_data.shape)

        psi_lambda = np.zeros(omega_data.shape)
        psi_teta = np.zeros(omega_data.shape)

        for levelIndex in range(0, omega_data.shape[0]):
            fieldData = omega_data[levelIndex,:,:]
            UItem, massFlux_lambdaItem, massFlux_tetaItem, omega_lambdaItem, omega_tetaItem, psi_lambdaItem, psi_tetaItem = self.getLocalHadleyWalkerMassFluxes(fieldData, lon_data, lat_data, isXBcsArePeriodic, isYBcsAreNeuman)
            U[levelIndex,:,:] = UItem
            massFlux_lambda[levelIndex,:,:] = massFlux_lambdaItem
            massFlux_teta[levelIndex,:,:] = massFlux_tetaItem
            omega_lambda[levelIndex,:,:] = omega_lambdaItem
            omega_teta[levelIndex,:,:] = omega_tetaItem
            psi_lambda[levelIndex,:,:] = psi_lambdaItem
            psi_teta[levelIndex,:,:] = psi_tetaItem

        output_vo = {}
        output_vo['U'] = U
        output_vo['massFlux_lambda'] = massFlux_lambda
        output_vo['massFlux_teta'] = massFlux_teta
        output_vo['omega_lambda'] = omega_lambda
        output_vo['omega_teta'] = omega_teta
        output_vo['psi_lambda'] = psi_lambda
        output_vo['psi_teta'] = psi_teta

        return output_vo

