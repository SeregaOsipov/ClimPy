import numpy as np
import matplotlib.pyplot as pl
from scipy.sparse import spdiags, linalg

def five_pt_laplacian_sparse(m):
    e = np.ones(m ** 2)
    e2 = ([1] * (m - 1) + [0]) * m
    e3 = ([0] + [1] * (m - 1)) * m
    h = 1. / (m + 1)
    A = spdiags([-4 * e, e2, e3, e, e], [0, -1, 1, -m, m], m ** 2, m ** 2)
    A /= h ** 2
    return A

# Define the RHS function:
f = lambda x, y: (2 - (np.pi ** 2) * (x ** 2)) * np.cos(np.pi * y);

#let calc rate of convergence and all the solutions and error

normMaxValues = np.zeros(8);
hValues = np.zeros(len(normMaxValues));
for i in range(0, len(normMaxValues)):
    m = 9 + 2 ** i;
    h = 1. / (m + 1);
    x = np.linspace(0, 1, m + 2); x = x[1:-1];
    y = np.linspace(0, 1, m + 2); y = y[1:-1];
    hValues[i] = x[1] - x[0];
    X, Y = np.meshgrid(x, y);
    
    #Set up and solve the linear system
    A = five_pt_laplacian_sparse(m).tocsr();
    
    F = f(X, Y);#.reshape([m**2]);
    #y index is inversed! and don't forget to divide by h^2 
    F[:, 0] -= 0;
    F[:, -1] -= np.cos(np.pi * y) / h ** 2;
    F[-1, :] -= -x ** 2 / h ** 2;
    F[0, :] -= x ** 2 / h ** 2;
    F = F.reshape([m ** 2]);
    
    U = linalg.spsolve(A, F)
    U = U.reshape([m, m]);
    
    U_exact = X ** 2 * np.cos(np.pi * Y)
    #U_exact = U_exact.reshape([m**2]);
    error = U - U_exact
    #don't forget to reshape, otherwise norm will be wrong
    normMaxValues[i] = np.linalg.norm(error.reshape([m ** 2]), np.inf);

pl.clf();
pl.subplots_adjust(hspace=1);

pl.subplot(4, 1, 1);
pl.pcolor(X, Y, U)
pl.colorbar();
pl.xlabel('x');
pl.ylabel('y');
pl.title('Figure 7a. Numerical solution');

pl.subplot(4, 1, 2);
pl.pcolor(X, Y, U_exact);
pl.xlabel('x');
pl.ylabel('y');
pl.title('Figure 7b. Exact solution');
pl.colorbar();

pl.subplot(4, 1, 3);
pl.pcolor(X, Y, error);
pl.xlabel('x');
pl.ylabel('y');
pl.title('Figure 7c. Error');
pl.colorbar();

# pl.subplot(4, 1, 4);
# pl.pcolor(X, Y, F.reshape([m,m]));
# pl.xlabel('x');
# pl.ylabel('y');
# pl.title('Figure 7c. forcing');
# pl.colorbar();

pl.subplot(4, 1, 4);
pl.loglog(hValues, normMaxValues, label='Max norm');
pl.xlabel('h');
pl.ylabel('Max norm');
pl.legend(loc='best');
pl.title('Figure 7d. Convergence');

pl.show();
