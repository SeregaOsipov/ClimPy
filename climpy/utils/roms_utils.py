import time
import netCDF4
import numpy as np
# import seawater as csr
from scipy.signal._peak_finding import _boolrelextrema

__author__ = 'Sergey Osipov <Serega.Osipov@gmail.com>'


def get_roms_nc_parameters(nc_file_path):
    netcdf_dataset_impl = netCDF4.Dataset
    if (type(nc_file_path) is list):
        netcdf_dataset_impl = netCDF4.MFDataset
    nc = netcdf_dataset_impl(nc_file_path)
    Vtransform = nc['Vtransform'][:]
    Vstretching = nc['Vstretching'][:]
    sc_r = nc['s_rho'][:]
    Cs_r = nc['Cs_r'][:]
    sc_w = nc['s_w'][:]
    Cs_w = nc['Cs_w'][:]
    return Vtransform, Vstretching, sc_r, Cs_r, sc_w, Cs_w


def compute_depths(nc_file_path, tyxSlices, igrid=1, idims=0):
    Vtransform, Vstretching, sc_r, Cs_r, sc_w, Cs_w = get_roms_nc_parameters(nc_file_path)
    netcdf_dataset_impl = netCDF4.Dataset
    if (type(nc_file_path) is list):
        netcdf_dataset_impl = netCDF4.MFDataset
    nc = netcdf_dataset_impl(nc_file_path)
    # Read in S-coordinate parameters.
    N = len(sc_r)
    Np = N + 1

    if (len(sc_w) == N):
        sc_w = np.cat(-1, sc_w.transpose())
        Cs_w = np.cat(-1, Cs_w.transpose())

    # Get bottom topography.
    yxSlice = [tyxSlices[1], tyxSlices[2]]
    h = nc.variables['h'][yxSlice]
    [Mp, Lp] = h.shape
    L = Lp - 1
    M = Mp - 1

    # Get free-surface
    zeta = nc.variables['zeta'][tyxSlices]
    # zeta=np.zeros([Lp, Mp])

    if igrid == 1:
        if idims == 1:
            h = h.transpose()
            zeta = zeta.transpose()
    elif igrid == 2:
        hp = 0.25 * (h[1:L, 1:M] + h[2:Lp, 1:M] + h[1:L, 2:Mp] + h[2:Lp, 2:Mp])
        zetap = 0.25 * (zeta[1:L, 1:M] + zeta[2:Lp, 1:M] + zeta[1:L, 2:Mp] + zeta[2:Lp, 2:Mp])
        if idims:
            hp = hp.transpose()
            zetap = zetap.transpose()
    elif igrid == 3:
        hu = 0.5 * (h[1:L, 1:Mp] + h[2:Lp, 1:Mp])
        zetau = 0.5 * (zeta[1:L, 1:Mp] + zeta[2:Lp, 1:Mp])
        if idims:
            hu = hu.transpose()
            zetau = zetau.transpose()
    elif igrid == 4:
        hv = 0.5 * (h[1:Lp, 1:M] + h[1:Lp, 2:Mp])
        zetav = 0.5 * (zeta[1:Lp, 1:M] + zeta[1:Lp, 2:Mp])
        if idims:
            hv = hv.transpose()
            zetav = zetav.transpose()
    elif igrid == 5:
        if idims:
            h = h.transpose()
            zeta = zeta.transpose()

    # Set critical depth parameter.
    hc = np.min(h[:])
    if (nc.variables.has_key('hc')):
        hc = nc['hc'][:]
    # Compute depths, for a different variables size will not match

    if (Vtransform == 1):
        if igrid == 1:
            for k in range(N):
                z0 = (sc_r - Cs_r) * hc + Cs_r(k) * h
                z = z0 + zeta * (1.0 + z0 / h)
                #     end
                #   case 2
                #     for k=1:N,
                #       z0=(sc_r(k)-Cs_r(k))*hc + Cs_r(k).*hp;
                #       z(:,:,k)=z0 + zetap.*(1.0 + z0./hp);
                #     end
                #   case 3
                #     for k=1:N,
                #       z0=(sc_r(k)-Cs_r(k))*hc + Cs_r(k).*hu;
                #       z(:,:,k)=z0 + zetau.*(1.0 + z0./hu);
                #     end
                #   case 4
                #     for k=1:N,
                #       z0=(sc_r(k)-Cs_r(k))*hc + Cs_r(k).*hv;
                #       z(:,:,k)=z0 + zetav.*(1.0 + z0./hv);
                #     end
                #   case 5
                #     z(:,:,1)=-h;
                #     for k=2:Np,
                #       z0=(sc_w(k)-Cs_w(k))*hc + Cs_w(k).*h;
                #       z(:,:,k)=z0 + zeta.*(1.0 + z0./h);
                #     end
                # end
    elif Vtransform == 2:
        if igrid == 1:
            # we add time as 0 dimension to S(sigma, y, x) to get S(time, sigma, y, x)
            S = (hc * sc_r[np.newaxis, :, np.newaxis, np.newaxis] + h[np.newaxis, np.newaxis, :, :] * Cs_r[np.newaxis,
                                                                                                      :, np.newaxis,
                                                                                                      np.newaxis]) / (
                            hc + h[np.newaxis, np.newaxis, :, :])
            z = zeta[:, np.newaxis, :, :] + (zeta[:, np.newaxis, :, :] + h[np.newaxis, np.newaxis, :, :]) * S
        elif igrid == 4:
            S = (hc * sc_r[np.newaxis, :, np.newaxis, np.newaxis] + h[np.newaxis, np.newaxis, :, :] * Cs_r[np.newaxis,
                                                                                                      :, np.newaxis,
                                                                                                      np.newaxis]) / (
                            hc + h[np.newaxis, np.newaxis, :, :])
            z = zeta[:, np.newaxis, :, :] + (zeta[:, np.newaxis, :, :] + h[np.newaxis, np.newaxis, :, :]) * S
        elif igrid == 5:
            S = (hc * sc_w[np.newaxis, :, np.newaxis, np.newaxis] + h[np.newaxis, np.newaxis, :, :] * Cs_w[np.newaxis,
                                                                                                      :, np.newaxis,
                                                                                                      np.newaxis]) / (
                            hc + h[np.newaxis, np.newaxis, :, :])
            z = zeta[:, np.newaxis, :, :] + (zeta[:, np.newaxis, :, :] + h[np.newaxis, np.newaxis, :, :]) * S
            #   case 2
            #     for k=1:N,
            #       z0=(hc.*sc_r(k)+Cs_r(k).*hp)./(hc+hp);
            #       z(:,:,k)=zetap+(zetap+hp).*z0;
            #     end,
            #   case 3
            #     for k=1:N,
            #       z0=(hc.*sc_r(k)+Cs_r(k).*hu)./(hc+hu);
            #       z(:,:,k)=zetau+(zetau+hu).*z0;
            #     end,
            #   case 4
            #     for k=1:N,
            #       z0=(hc.*sc_r(k)+Cs_r(k).*hv)./(hc+hv);
            #       z(:,:,k)=zetav+(zetav+hv).*z0;
            #     end,
            #   case 5
            #     for k=1:Np,
            #       z0=(hc.*sc_w(k)+Cs_w(k).*h)./(hc+h);
            #       z(:,:,k)=zeta+(zeta+h).*z0;
            #     end
            # end

    return z


# DEPTHS:  Compute the ROMS depths associated with a 3D variable
#
#  [z]=depths(fname, gname, igrid, idims, tindex)
#
#  This function computes the depths at the requested staggered C-grid.
#  If the time record (tindex) is not provided, a zero free-surface is
#  assumed and the unperturbed depths are returned.
#
#  On Input:
#
#     fname       NetCDF data file name (character string)
#     gname       NetCDF grid file name (character string)
#     igrid       Staggered grid C-type (integer):
#                   igrid=1  => density points
#                   igrid=2  => streamfunction points
#                   igrid=3  => u-velocity points
#                   igrid=4  => v-velocity points
#                   igrid=5  => w-velocity points
#     idims       Depths dimension order switch (integer):
#                   idims=0  => (i,j,k)  column-major order (Fortran)
#                   idims=1  => (j,i,k)  row-major order (C-language)
#     tindex      Time index (integer)
#
#  On Output:
#
#     z           Depths (3D array; meters, negative)
def depths(fname, gname, igrid=1, idims=0, tindex=np.NAN):
    # Initialize vertical tranformation and stretching function to original formulation values.

    Vtransform = 1
    Vstretching = 1
    isHcPresent = False

    # Read in S-coordinate parameters.

    historyNc = netCDF4.Dataset(fname)
    Vtransform = historyNc['Vtransform'][:]
    Vstretching = historyNc['Vstretching'][:]
    isHcPresent = True
    name_r = 'sc_r'
    name_r = 's_rho'
    name_w = 'sc_w'
    name_w = 's_w'

    sc_r = historyNc[name_r][:]
    Cs_r = historyNc['Cs_r'][:]

    sc_w = historyNc[name_w][:]
    Cs_w = historyNc['Cs_w'][:]

    N = len(sc_r)
    Np = N + 1

    if (len(sc_w) == N):
        sc_w = np.cat(-1, sc_w.transpose())
        Cs_w = np.cat(-1, Cs_w.transpose())

    # Get bottom topography.
    h = historyNc['h'][:]
    [Mp, Lp] = h.shape
    L = Lp - 1
    M = Mp - 1

    # Get free-surface
    if (np.isnan(tindex)):
        zeta = np.zeros([Lp, Mp])
    else:
        zeta = historyNc['zeta'][tindex]

    if igrid == 1:
        if idims == 1:
            h = h.transpose()
            zeta = zeta.transpose()
    elif igrid == 2:
        hp = 0.25 * (h[1:L, 1:M] + h[2:Lp, 1:M] + h[1:L, 2:Mp] + h[2:Lp, 2:Mp])
        zetap = 0.25 * (zeta[1:L, 1:M] + zeta[2:Lp, 1:M] + zeta[1:L, 2:Mp] + zeta[2:Lp, 2:Mp])
        if idims:
            hp = hp.transpose()
            zetap = zetap.transpose()
    elif igrid == 3:
        hu = 0.5 * (h[1:L, 1:Mp] + h[2:Lp, 1:Mp])
        zetau = 0.5 * (zeta[1:L, 1:Mp] + zeta[2:Lp, 1:Mp])
        if idims:
            hu = hu.transpose()
            zetau = zetau.transpose()
    elif igrid == 4:
        hv = 0.5 * (h[1:Lp, 1:M] + h[1:Lp, 2:Mp])
        zetav = 0.5 * (zeta[1:Lp, 1:M] + zeta[1:Lp, 2:Mp])
        if idims:
            hv = hv.transpose()
            zetav = zetav.transpose()
    elif igrid == 5:
        if idims:
            h = h.transpose()
            zeta = zeta.transpose()

    # Set critical depth parameter.

    if (isHcPresent):
        hc = historyNc['hc'][:]
    else:
        hc = np.min(h[:])

    # Compute depths.
    # for a different variables size will not match
    z = np.zeros((N, Mp, Lp))

    if (Vtransform == 1):
        if igrid == 1:
            for k in range(N):
                z0 = (sc_r - Cs_r) * hc + Cs_r * h
                z = z0 + zeta * (1.0 + z0 / h)
    #     end
    #   case 2
    #     for k=1:N,
    #       z0=(sc_r(k)-Cs_r(k))*hc + Cs_r(k).*hp;
    #       z(:,:,k)=z0 + zetap.*(1.0 + z0./hp);
    #     end
    #   case 3
    #     for k=1:N,
    #       z0=(sc_r(k)-Cs_r(k))*hc + Cs_r(k).*hu;
    #       z(:,:,k)=z0 + zetau.*(1.0 + z0./hu);
    #     end
    #   case 4
    #     for k=1:N,
    #       z0=(sc_r(k)-Cs_r(k))*hc + Cs_r(k).*hv;
    #       z(:,:,k)=z0 + zetav.*(1.0 + z0./hv);
    #     end
    #   case 5
    #     z(:,:,1)=-h;
    #     for k=2:Np,
    #       z0=(sc_w(k)-Cs_w(k))*hc + Cs_w(k).*h;
    #       z(:,:,k)=z0 + zeta.*(1.0 + z0./h);
    #     end
    # end
    elif Vtransform == 2:
        if igrid == 1:
            for k in range(N):
                z0 = (hc * sc_r[k] + Cs_r[k] * h) / (hc + h)
                z[k, :, :] = zeta + (zeta + h) * z0
    #   case 2
    #     for k=1:N,
    #       z0=(hc.*sc_r(k)+Cs_r(k).*hp)./(hc+hp);
    #       z(:,:,k)=zetap+(zetap+hp).*z0;
    #     end,
    #   case 3
    #     for k=1:N,
    #       z0=(hc.*sc_r(k)+Cs_r(k).*hu)./(hc+hu);
    #       z(:,:,k)=zetau+(zetau+hu).*z0;
    #     end,
    #   case 4
    #     for k=1:N,
    #       z0=(hc.*sc_r(k)+Cs_r(k).*hv)./(hc+hv);
    #       z(:,:,k)=zetav+(zetav+hv).*z0;
    #     end,
    #   case 5
    #     for k=1:Np,
    #       z0=(hc.*sc_w(k)+Cs_w(k).*h)./(hc+h);
    #       z(:,:,k)=zeta+(zeta+h).*z0;
    #     end
    # end

    return z


# compute_mld_based_on_density_threshold Compute a mixed layer depth
# mld.pdvar - computed based on variable potential density criterion pd(0)-pd(mld)=var(T(0),S(0)), where var is a variable potential density difference which corresponds to constant temperature difference of 0.5 (in this case 0.8) degree C
# For more datail see Monterey, G., and S. Levitus, 1997: Seasonal variability of mixed layer depth for the World Ocean. NOAA Atlas, NESDIS 14, 100 pp., Washington, D.C.
def compute_mld_based_on_density_threshold(S, potentialTemperature, Z):
    # Compute a mixed layer depth (MLD) from vertical profiles of salinity and temperature, given depth of samples.
    # There are many criteria out there to compute a MLD, the one used here defined MLD as the first depth for which:
    #		ST(T(z),S(z)) > ST(T(z=0)-0.8,S(z=0))
    # Honestly, I don't remember the reference for this definition.
    # Inputs:
    #	S: salinity (psu)
    #	potentialTemperature: potential temperature (degC)
    #	Z: depth (m)
    # Output:
    #	H: mixed layer depth (m)
    # Rq:
    #	- We assume the vertical axis to be sorted from top (surface) to bottom. DEPTH ie Z IS NEGATIVE

    # [zmin iz0] = min(abs(Z));
    # if isnan(T(iz0))
    #     iz0 = iz0+1;
    # end
    #

    # mld = np.empty((1, ) + potentialTemperature.shape[1:])
    # mld[:] = np.NaN

    SST = potentialTemperature[-1]
    SSS = S[-1]

    # SST08 = SST - 0.8;        # #SSS   = SSS + 35
    # to compute potential density, compute regular density using potential temperature
    potentialDensitySurface08 = csr.dens(SSS, SST - 0.8, 0)
    potentialDensity = csr.dens(S, potentialTemperature, 0)

    # we reverse array in z to find first level
    ind = potentialDensity[::-1] > potentialDensitySurface08
    # this is where mld has been found
    validPointsMask = np.nansum(ind.data, axis=0)
    # these are the indicies of the vertical level
    zInd = np.argmax(ind == 1, axis=0)
    # generate indicies for the rest of the dimensions
    a1, a2 = np.indices(SST.shape)
    # dont forget to reverse array vertically
    mld = Z[::-1][zInd, a1, a2]
    # mask points where mld was not found
    mld[np.logical_not(validPointsMask)] = np.NaN
    return mld


def compute_mld_based_on_density_curvature(sigma, z, qi_treshold=0.55):
    # z is assumed to follow roms convention, negative, from bottom to top
    # modified from the original idea by
    # Ocean mixed layer depth: A subsurface proxy of ocean-atmosphere variability, K. Lorbacher, D. Dommenget, P. P. Niiler, A. Kohl, 12 July 2006, DOI: 10.1029/2003JC002157

    sigma_sorted = sigma[::-1]
    z_sorted = -1 * z[::-1]
    dz = np.diff(z_sorted, axis=0)
    # forward difference
    sigma_gradient = np.diff(sigma_sorted, axis=0) / np.diff(z_sorted, axis=0)
    sigma_curvature = np.empty(sigma_gradient.shape)
    # backward difference
    sigma_curvature[1:] = np.diff(sigma_gradient, axis=0) / np.diff(z_sorted[:-1], axis=0)
    # at the edge have to use forward difference
    sigma_curvature[0] = np.diff(sigma_gradient[0:2], axis=0) / np.diff(z_sorted[0:2], axis=0)

    # find curvature local maxima
    curvature_extrema_max_ind = _boolrelextrema(sigma_curvature, np.greater, order=1, axis=0)
    # find positive first derivative and convex down
    ind_positive_derivative_positive_curvature = np.logical_and(sigma_gradient > 0, sigma_curvature > 0)
    # keep only correct maxima
    ind_max_extremum = np.logical_and(curvature_extrema_max_ind, ind_positive_derivative_positive_curvature)

    qi_data = np.empty(ind_max_extremum.shape)
    qi_data[:] = np.NaN

    # ind = np.where(ind_max_extremum)
    # ind = np.asarray(ind)
    # ind2 = np.minimum((1.5 * ind[0]).astype(int), z_sorted.shape[0])
    # t1 = time.time()
    # print 'start for loop ' + str(ind.shape[1])
    # for i in range(ind.shape[1]):
    #     qi_data[ind[0,i], ind[1,i], ind[2,i]] = 1 - np.std(sigma_sorted[0:ind[0,i], ind[1,i], ind[2,i]]) / np.std(sigma_sorted[0:ind2[i], ind[1,i], ind[2,i]])
    # print 'end for loop'
    # t2 = time.time()
    # print t2-t1

    # compute quality index everywhere
    for i in range(ind_max_extremum.shape[0]):
        qi_data[i] = 1 - np.std(sigma_sorted[0:i], axis=0) / np.std(
            sigma_sorted[0:min(i * 1.5, ind_max_extremum.shape[0])], axis=0)

    masked_qi_data = np.ma.array(qi_data, mask=np.logical_not(ind_max_extremum))

    final_ind = np.empty(sigma_sorted.shape[1:])
    final_ind[:] = np.NaN

    profile_max_qi_data = np.nanmax(masked_qi_data, axis=0)
    # sub_index =  profile_max_qi_data < 0.3
    # # if qi index is small, then stratification is very weak, then pick local extremum/maxima which has bigest curvature maxima
    # masked_data = np.ma.array(sigma_curvature, mask=np.logical_not(ind_max_extremum))
    # final_ind[sub_index] = np.argmax(masked_data[:, sub_index], axis=0)
    # final_ind[sub_index] = np.argmax(sigma_curvature[:, sub_index], axis=0)
    sub_index = profile_max_qi_data >= qi_treshold
    # otherwise pick local extremum/maxima which has highest qi index
    final_ind[sub_index] = np.nanargmax(masked_qi_data[:, sub_index], axis=0)
    # if there is no local extremum or qi is weak, then pick point with max gradient value
    sub_index = np.logical_or(np.isnan(profile_max_qi_data), profile_max_qi_data < qi_treshold)
    final_ind[sub_index] = np.nanargmax(sigma_gradient[:, sub_index], axis=0)

    final_ind = final_ind.astype(int)
    fancy_indices = np.indices(ind_max_extremum.shape[1:])
    mld = z_sorted[final_ind, fancy_indices[0], fancy_indices[1]]

    # if (plot_diags):
    #     fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(12, 18))
    #     plt.sca(axes[0])
    #     plt.plot(z_sorted[final_ind], sigma_sorted[final_ind], 'g*', markersize=20)
    #     plt.plot(z_sorted, sigma_sorted, 'k-o')
    #     plt.plot(z_sorted[ind], sigma_sorted[ind], 'ro')
    #     plt.xlim((0, np.max(z_sorted)))
    #     plt.title('sigma')
    #     plt.sca(axes[1])
    #     plt.plot(z_sorted[:-1][final_ind], sigma_gradient[final_ind], 'g*', markersize=20)
    #     plt.plot(z_sorted[:-1], sigma_gradient, 'k-o')
    #     plt.plot(z_sorted[:-1][ind], sigma_gradient[ind], 'ro')
    #     plt.gca().set_yscale('symlog', linthreshy=10 ** -5)
    #     plt.xlim((0, np.max(z_sorted)))
    #     plt.title('derivative')
    #     plt.sca(axes[2])
    #     plt.plot(z_sorted[:-1][final_ind], sigma_curvature[final_ind], 'g*', markersize=20)
    #     plt.plot(z_sorted[:-1], sigma_curvature, 'k-o')
    #     plt.plot(z_sorted[:-1][ind], sigma_curvature[ind], 'ro')
    #     plt.gca().set_yscale('symlog', linthreshy=10 ** -8)
    #     plt.xlim((0, np.max(z_sorted)))
    #     plt.title('curvature')
    #
    #     plt.sca(axes[3])
    #     plt.plot(z_sorted[:-1][ind], qi_data, 'ko')
    #     # plt.plot(z_sorted[:-1][curvature_extrema_max_ind], current_sigma_curvature[curvature_extrema_max_ind], 'ro')
    #     # plt.gca().set_yscale('symlog', linthreshy=10 ** -8)
    #     plt.xlim((0, np.max(z_sorted)))
    #     plt.title('qi')

    # invert in z order everything back to original and make compatible shape

    current_sigma_gradient_temp = np.zeros(sigma_sorted.shape)
    current_sigma_gradient_temp[:-1] = sigma_gradient
    current_sigma_gradient_temp[-1] = np.NaN
    sigma_gradient = current_sigma_gradient_temp[::-1]

    current_sigma_curvature_temp = np.zeros(sigma_sorted.shape)
    current_sigma_curvature_temp[:-1] = sigma_curvature
    current_sigma_curvature_temp[-1] = np.NaN
    sigma_curvature = current_sigma_curvature_temp[::-1]

    ind_max_extremum_temp = np.zeros(sigma_sorted.shape, dtype=bool)
    ind_max_extremum_temp[:-1] = ind_max_extremum
    ind_max_extremum_temp[-1] = np.NaN
    ind_max_extremum = ind_max_extremum_temp[::-1]

    qi_data_temp = np.zeros(sigma_sorted.shape)
    qi_data_temp[:-1] = qi_data
    qi_data_temp[-1] = np.NaN
    qi_data = qi_data_temp[::-1]

    return mld, sigma_gradient, sigma_curvature, final_ind, ind_max_extremum, qi_data


def get_roms_cell_area(ds, water_only_mask=False):
    cell_area = 1/ds.pm/ds.pn
    if water_only_mask:
        cell_area = cell_area.where(ds.mask_rho)  # mask_rho is 1 over water
    return cell_area
