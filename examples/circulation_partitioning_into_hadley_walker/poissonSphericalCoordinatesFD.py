import numpy as np
import matplotlib.pyplot as pl
from scipy.sparse import spdiags, linalg

# pl.ion()

def five_pt_laplacian_sparse(Nx, Ny, x, y, dx, dy):
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

    xA = spdiags([periodicBelowDiag, periodicAboveDiag], [-Nx+1, Nx-1], Nx*Ny, Nx*Ny)
    A += xA

    #this will impose Neumann BC for lower y boundary
    neumanTopMainDiag = np.zeros(Ny*Nx)
    neumanTopMainDiag[0:Nx] = 4/3 * ( dy**-2 - np.cos(y[0])/np.sin(y[0])/2/dy )

    neumanTopAboveMainDiag = np.zeros(Ny*Nx)
    neumanTopAboveMainDiag[Nx:Nx+Nx] = -1/3 * ( dy**- 2 - np.cos(y[0])/np.sin(y[0])/2/dy)

    B = spdiags([neumanTopMainDiag, neumanTopAboveMainDiag], [0, Nx], Nx*Ny, Nx*Ny)

    #this will impose Neumann BC for upper y boundary
    neumanBottomMainDiag = np.zeros(Ny*Nx)
    neumanBottomMainDiag[-Nx:] = 4/3 * ( dy**-2 + np.cos(y[-1])/np.sin(y[-1])/2/dy )

    neumanBottomBelowMainDiag = np.zeros(Ny*Nx)
    neumanBottomBelowMainDiag[-Nx-Nx:-Nx] = -1/3 * ( dy**-2 + np.cos(y[-1])/np.sin(y[-1])/2/dy)

    C = spdiags([neumanBottomMainDiag, neumanBottomBelowMainDiag], [0, -Nx], Nx*Ny, Nx*Ny)

    # A += B+C

    return A

# Define the RHS function:
# f = lambda x, y: y*(1+3*np.cos(2*x))
# f = lambda x, y: 4*y*(1+2*np.cos(2*x)+2*np.cos(4*x)+2*np.cos(6*x)+9*np.cos(8*x))
f = lambda x, y: 32*y*np.cos(4*x)**2*np.sin(y)**-2+(np.tan(y)**-1-32*y*np.sin(y)**-2)*np.sin(4*x)**2
Ue = lambda x, y: np.sin(4*x)**2 * y

#let calc rate of convergence and all the solutions and error

normMaxValues = np.zeros(7)
hValues = np.zeros(len(normMaxValues))
for i in range(0, len(normMaxValues)):
    Ny = 9 + 2 ** i
    Nx = Ny*2

    domainMinX = 0
    domainMaxX = 2*np.pi

    # domainMinX = np.pi/2
    # domainMaxX = np.pi

    domainMinY = np.pi/6
    domainMaxY = np.pi*4/6

    #do not include last point in x do keep problem periodic in X
    x = np.linspace(domainMinX, domainMaxX, Nx + 1); x = x[0:-1]
    #this is the Dirichlet BC case
    # x = np.linspace(domainMinX, domainMaxX, Nx + 2); x = x[1:-1]
    #this is the y Dirichlet BC case
    y = np.linspace(domainMinY, domainMaxY, Ny + 2); y = y[1:-1]
    #this is the y Neuman BC case
    # y = np.linspace(domainMinY, domainMaxY, Ny)
    # y = np.linspace(domainMinY, domainMaxY, Ny)
    X, Y = np.meshgrid(x, y)

    dx = x[1]-x[0]
    dy = y[1]-y[0]
    hValues[i] = dx
    
    #Set up and solve the linear system
    A = five_pt_laplacian_sparse(Nx, Ny, x, y, dx, dy).tocsr()
    F = f(X, Y)

    #this is x Dirichlet BC
    # F[:, 0] -= Ue(domainMinX, y) * dx**-2 / np.sin(y)**2
    # F[:, -1] -= Ue(domainMaxX, y) * dx**-2 / np.sin(y)**2
    #x BC are periodic and setup through the FD scheme

    #y index is inversed!
    #this is Dirichlet BC
    F[0, :] -= Ue(x, domainMinY) * (dy**-2 - np.cos(y[0])/np.sin(y[0])/2/dy)
    F[-1, :] -= Ue(x, domainMaxY) * (dy**-2 + np.cos(y[-1])/np.sin(y[-1])/2/dy)
    #this is Neumann BC
    # F[0, :] -= -np.sin(4*x)**2 * 2/3 * dy * (dy**-2 - np.cos(y[0])/np.sin(y[0])/2/dy)
    # F[-1, :] -= np.sin(4*x)**2 * 2/3 * dy * (dy**-2 + np.cos(y[-1])/np.sin(y[-1])/2/dy)

    F = F.reshape([Nx * Ny])
    U = linalg.spsolve(A, F)
    U = U.reshape([Ny, Nx])
    U_exact = Ue(X,Y)
    error = (U_exact - U)
    #don't forget to reshape, otherwise norm will be wrong
    normMaxValues[i] = np.linalg.norm(error.reshape([Nx*Ny]), np.inf)

maxU = np.max(U_exact)
minU = np.min(U_exact)

pl.figure(figsize=(20,10))
pl.subplots_adjust(hspace=1)

ax = pl.subplot(2, 3, 1)
ax.set_aspect('equal')
pl.pcolor(X, Y, U, cmap='Reds')
pl.clim(minU, maxU)
pl.colorbar();
pl.xlabel('x');
pl.ylabel('y');
pl.title('Figure 7a. Numerical solution');

ax = pl.subplot(2, 3, 2);
ax.set_aspect('equal')
pl.pcolor(X, Y, U_exact, cmap='Reds');
pl.clim(minU, maxU)
pl.xlabel('x');
pl.ylabel('y');
pl.title('Figure 7b. Exact solution');
pl.colorbar();

ax = pl.subplot(2, 3, 3);
ax.set_aspect('equal')
pl.pcolor(X, Y, np.abs(error), cmap='Reds')
pl.clim(minU, maxU)
pl.xlabel('x');
pl.ylabel('y');
pl.title('Figure 7c. Error');
pl.colorbar()

pl.subplot(2, 3, 4);
pl.semilogx(hValues, normMaxValues, '-o', label='Max norm');
pl.xlabel('h');
pl.ylabel('Max norm');
pl.legend(loc='best');
pl.title('Figure 7d. Convergence');

ax = pl.subplot(2, 3, 5);
ax.set_aspect('equal')
pl.pcolor(X, Y, f(X,Y));
# pl.clim(minU, maxU)
pl.xlabel('x');
pl.ylabel('y');
pl.title('f');
pl.colorbar();

pl.show()
