import matplotlib
matplotlib.use('Qt4Agg')
from sfepy.mesh.mesh_generators import gen_block_mesh

__author__ = 'Sergey Osipov <Serega.Osipov@gmail.com>'

from optparse import OptionParser
import numpy as np

import sys
sys.path.append('.')

from sfepy.base.base import IndexedStruct
from sfepy.discrete import (FieldVariable, Material, Integral, Function,
                            Equation, Equations, Problem)
from sfepy.discrete.fem import Mesh, FEDomain, Field
from sfepy.terms import Term
from sfepy.discrete.conditions import Conditions, EssentialBC
from sfepy.solvers.ls import ScipyDirect
from sfepy.solvers.nls import Newton
from sfepy.postprocess.viewer import Viewer

import matplotlib.pyplot as plt

print "inside main"
from sfepy import data_dir

#mesh = Mesh.from_file(data_dir + '/meshes/2d/rectangle_tri.mesh')
mesh = gen_block_mesh([2],[100,100],[0,0], coors=[-2, 2])
domain = FEDomain('domain', mesh)

min_x, max_x = domain.get_mesh_bounding_box()[:,0]
eps = 1e-8 * (max_x - min_x)
omega = domain.create_region('Omega', 'all')
gammaL = domain.create_region('Gamma_Left', 'vertices in x < %.9f' % (min_x + eps),'facet')
gammaR = domain.create_region('Gamma_Right','vertices in x > %.9f' % (max_x - eps),'facet')

min_y, max_y = domain.get_mesh_bounding_box()[:,1]
eps = 1e-8 * (max_y - min_y)
gammaT = domain.create_region('Gamma_Top', 'vertices in y < %.9f' % (min_y + eps),'facet')
gammaB = domain.create_region('Gamma_Bottom','vertices in y > %.9f' % (max_y - eps),'facet')


# def get_forcing_term(coors, domain=None):
#     coors
#     x = coors[:, 0]
#     val = 55.0 * (x - 0.05)
#     val = 10.0 * (x)
#
#     val.shape = (coors.shape[0], 1, 1)
#     return val#{'f' : val}


# field = Field.from_args('temperature', nm.float64, 'vector', omega, approx_order=2)
field = Field.from_args('temperature', np.float64, 1, omega, 1)

t = FieldVariable('t', 'unknown', field, 0)
s = FieldVariable('s', 'test', field, primary_var_name='t')

# f = FieldVariable('f', 'parameter', field, {'setter' : get_forcing_term}, primary_var_name='set-to-None')
# self, name, kind, field, order=None, primary_var_name=None,special=None, flags=None, **kwargs):

def get_forcing_term(ts, coors, mode=None, **kwargs):
    if mode == 'qp':
        x = coors[:, 0]
        y = coors[:, 1]
        val = -2*np.sin(x/np.pi)*np.sin(y/np.pi)
        # val = np.exp(-(x**2+10*y**2))

        val.shape = (coors.shape[0], 1, 1)
        return {'val' : val}

c = Material('c', val=1.0)
# bc_fun = Function('shift_u_fun', shift_u_fun, extra_args={'shift' : 0.01})
f = Material('f', function=get_forcing_term)
# f = Material('f', val = 10.0)
# f = Material('f', val=[[10.0],[10.0]])
# f = Material('f', val=[[0],[0]])

integral = Integral('i', order=2)

# t1 = Term.new('dw_lin_elastic_iso(m.lam, m.mu, v, u)',
#      integral, omega, m=m, v=v, u=u)
# t2 = Term.new('dw_volume_lvf(f.val, v)', integral, omega, f=f, v=v)
# eq = Equation('balance', t1 + t2)
# eqs = Equations([eq])

t1 = Term.new('dw_laplace( c.val, s, t )', integral, omega, c=c, t=t, s=s)
# t2 = Term.new('dw_volume_dot( f, s, t )', integral, omega, f=f, t=t, s=s)
t2 = Term.new('dw_volume_dot( f.val, s, t )', integral, omega, f=f, t=t, s=s)
# t2 = Term.new('dw_volume_dot( f.val, s )', integral, omega, f=f, s=s)


eq = Equation('balance', t1 - t2)
eqs = Equations([eq])

# def set_right_bc_impl(ts, coors, bc=None, problem=None, **kwargs):
#     x = coors[:,0]
#     y = coors[:,1]
#     val = np.sin(x)+np.sin(y)
#     return val

def set_bc_impl(ts, coors, bc=None, problem=None, **kwargs):
    x = coors[:,0]
    y = coors[:,1]
    val = np.sin(x/np.pi)*np.sin(y/np.pi)
    # val = 0*x
    # val = 0.0001*np.ones(x.size)
    return val

# fix_u = EssentialBC('fix_u', omega, {'u.all' : 0.0})
# bc1 = EssentialBC('Gamma_Left', gammaL, {'t.0' : -20.0})
# bc2 = EssentialBC('Gamma_Right', gammaR, {'t.0' : 20.0})

set_bc_fun = Function('set_bc_impl', set_bc_impl)
bc1 = EssentialBC('Gamma_Left', gammaL, {'t.0' : set_bc_fun})
bc2 = EssentialBC('Gamma_Right', gammaR, {'t.0' : set_bc_fun})

bc3 = EssentialBC('Gamma_Top', gammaT, {'t.0' : set_bc_fun})
bc4 = EssentialBC('Gamma_Bottom', gammaB, {'t.0' : set_bc_fun})

ls = ScipyDirect({})

nls_status = IndexedStruct()
newtonConfig = {'i_max':10, 'eps_a':1e-10, 'eps_r':1}
nls = Newton(newtonConfig, lin_solver=ls, status=nls_status)

pb = Problem('Poisson', equations=eqs, nls=nls, ls=ls)
pb.save_regions_as_groups('regions')

# pb.time_update(ebcs=Conditions([fix_u, t1, t2]))
pb.time_update(ebcs=Conditions([bc1, bc2, bc3, bc4]))

vec = pb.solve()
print nls_status

pb.save_state('customCylinder.vtk', vec)

# if options.show:
# view = Viewer('customCylinder.vtk')
# view(vector_mode='warp_norm', rel_scaling=2,
#      is_scalar_bar=True, is_wireframe=True)


solutionData = vec.vec.reshape(100, 100)
xGrid = mesh.coors[:,0].reshape(100,100)
yGrid = mesh.coors[:,1].reshape(100,100)
analyticSolution = np.sin(xGrid/np.pi)*np.sin(yGrid/np.pi)

errorData = 100*(solutionData-analyticSolution)/analyticSolution

plt.subplot(221)
plt.title('numerics')
plt.contourf(xGrid, yGrid, solutionData, 100, cmap='seismic')
plt.colorbar()
plt.subplot(222)
plt.title('exact')
plt.contourf(xGrid, yGrid, analyticSolution, 100, cmap='seismic')
plt.colorbar()
plt.subplot(223)
plt.title('relative error')
levels = np.linspace(-100,100, 21, endpoint=True)
plt.contourf(xGrid, yGrid, errorData, 100, levels=levels, cmap='seismic')
# plt.clim(-100,100)
cbar = plt.colorbar()
# cbar.set_clim(-100, 100)
plt.show()