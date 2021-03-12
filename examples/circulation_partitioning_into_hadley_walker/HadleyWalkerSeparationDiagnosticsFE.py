import calendar
from scipy.interpolate.fitpack2 import RectBivariateSpline
from scipy.interpolate.interpolate import interp2d
from scipy.interpolate.ndgriddata import griddata
from libs.readers.DataSetDescriptionVO import DataSetDescriptionVO
from libs.readers.DataSetReadersFactory import DataSetReadersFactory

__author__ = 'Sergey Osipov <Serega.Osipov@gmail.com>'
import matplotlib
matplotlib.use('Qt4Agg')
from sfepy.mesh.mesh_generators import gen_block_mesh

from optparse import OptionParser
import numpy as np
import scipy as sp
import datetime as dt
import sfepy.discrete.fem.periodic as per

import sys
sys.path.append('.')

from sfepy.base.base import IndexedStruct
from sfepy.discrete import (FieldVariable, Material, Integral, Function,
                            Equation, Equations, Problem)
from sfepy.discrete.fem import Mesh, FEDomain, Field
from sfepy.terms import Term
from sfepy.discrete.conditions import Conditions, EssentialBC, PeriodicBC
from sfepy.solvers.ls import ScipyDirect
from sfepy.solvers.nls import Newton
from sfepy.postprocess.viewer import Viewer

import matplotlib.pyplot as plt

print "inside main"
from sfepy import data_dir


descVO = DataSetDescriptionVO(DataSetDescriptionVO.ERA)
# descVO.levelsName = ''
descVO.variableFilePath = '/home/osipovs/Downloads/netcdf-atls16-a562cefde8a29a7288fa0b8b7f9413f7-SblhfN.nc'
# descVO.variableFilePath = '/home/osipovs/Downloads/netcdf-atls07-a562cefde8a29a7288fa0b8b7f9413f7-0xPUGr.nc'
descVO.variableName = 'w'
# descVO.variableName = 'z'
readersFactory = DataSetReadersFactory()
reader = readersFactory.getReader(descVO)
eraDataSetVO = reader.readDataSet(descVO)

#mesh = Mesh.from_file(data_dir + '/meshes/2d/rectangle_tri.mesh')
mesh = gen_block_mesh([2],[100,100],[0,0])
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


# field = Field.from_args('temperature', nm.float64, 'vector', omega, approx_order=2)
field = Field.from_args('temperature', np.float64, 1, omega, 1)

t = FieldVariable('t', 'unknown', field, 0)
s = FieldVariable('s', 'test', field, primary_var_name='t')

# f = FieldVariable('f', 'parameter', field, {'setter' : get_forcing_term}, primary_var_name='set-to-None')
# self, name, kind, field, order=None, primary_var_name=None,special=None, flags=None, **kwargs):

#keep only first time index and values at some single level
eraDataSetVO.fieldData = eraDataSetVO.fieldData[0,23,:,:]

def get_forcing_term(ts, coors, mode=None, eraDataSetVO=None, **kwargs):
    if mode == 'qp':
        #don't forget that we reversed the order along the lat dimension so that lat is strictly increasing for interpolation
        latData = eraDataSetVO.latData[::-1]
        lonData = eraDataSetVO.lonData
        fieldData = eraDataSetVO.fieldData[::-1,:]
        #remap lat lon to x y from -1 to 1 matching the original domain
        xData = np.linspace(-1,1, lonData.size)
        yData = np.linspace(-1,1, latData.size)
        interp_fun = RectBivariateSpline(yData, xData, fieldData)
        # interp_fun = interp2d(eraDataSetVO.lonData, eraDataSetVO.latData, eraDataSetVO.fieldData[:,:,0])

        x = coors[:, 0]
        y = coors[:, 1]
        # latDataI = np.linspace(np.min(latData), np.max(latData), 100)
        # lonDataI = np.linspace(np.min(lonData), np.max(lonData), 100)
        # val = interp_fun(latDataI, lonDataI)
        # val = griddata((xData,yData),np.swapaxes(fieldData,0,1), (x,y))
        val = np.empty((coors.shape[0],1))
        for i in range(coors.shape[0]):
            val[i] = interp_fun(x[i], y[i])

        # plt.subplot(221)
        # plt.title('latlon')
        # plt.contourf(lonData, latData, fieldData, 100)
        # plt.colorbar()
        # plt.subplot(222)
        # plt.title('remap')
        # plt.contourf(xData, yData, fieldData, 100)
        # plt.colorbar()
        # plt.subplot(223)
        # plt.title('interp')
        # # temp = griddata((x,y),val, (xData[None,:],yData[:,None]))
        # # temp.shape = (temp.shape[0], temp.shape[1])
        # plt.scatter(y,x,val)
        # # plt.contourf(x,y,temp, 100)
        # # # plt.colorbar()
        # plt.show()

        val.shape = (coors.shape[0], 1, 1)
        return {'val' : val}

c = Material('c', val=1.0)

get_forcing_term_fun = Function('get_forcing_term', get_forcing_term, extra_args={'eraDataSetVO' : eraDataSetVO})

f = Material('f', function=get_forcing_term_fun)
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
    val = np.ones(x.shape)
    return val

# fix_u = EssentialBC('fix_u', omega, {'u.all' : 0.0})
# bc1 = EssentialBC('Gamma_Left', gammaL, {'t.0' : -20.0})
# bc2 = EssentialBC('Gamma_Right', gammaR, {'t.0' : 20.0})

set_bc_fun = Function('set_bc_impl', set_bc_impl)
# bc1 = PeriodicBC('Gamma_Top_Bottom', (gammaL, gammaR), {'t.0' : 't.0'}, 'match_x_plane')
set_periodic_bc_fun = Function('match_x_plane', per.match_x_plane)
bc1 = PeriodicBC('Gamma_Left_Right', (gammaL, gammaR), {'t.all' : 't.all'}, set_periodic_bc_fun)
# bc1 = EssentialBC('Gamma_Left', gammaL, {'t.0' : set_bc_fun})
# bc2 = EssentialBC('Gamma_Right', gammaR, {'t.0' : set_bc_fun})

bc3 = EssentialBC('Gamma_Top', gammaT, {'t.0' : set_bc_fun})
bc4 = EssentialBC('Gamma_Bottom', gammaB, {'t.0' : set_bc_fun})

ls = ScipyDirect({})

nls_status = IndexedStruct()
nls = Newton({}, lin_solver=ls, status=nls_status)

pb = Problem('Poisson', equations=eqs, nls=nls, ls=ls)
pb.save_regions_as_groups('regions')

# pb.time_update(ebcs=Conditions([fix_u, t1, t2]))
pb.time_update(ebcs=Conditions([bc1, bc3, bc4]))
# pb.time_update(ebcs=Conditions([bc1, bc2, bc3, bc4]))



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
analyticSolution = np.sin(xGrid)*np.sin(yGrid)

errorData = analyticSolution-solutionData

plt.figure()
plt.subplot(221)
plt.title('numerics')
plt.contourf(solutionData, 100)
plt.colorbar()
plt.subplot(222)
plt.title('exact')
plt.contourf(analyticSolution, 100)
plt.colorbar()
plt.subplot(223)
plt.title('absolute error')
plt.contourf(errorData, 100)
plt.colorbar()
plt.show()