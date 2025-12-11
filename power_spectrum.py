import numpy as np
import matplotlib.pyplot as plt
import yt
from yt.units import kpc
from scipy import signal as ss
from matplotlib.colors import LogNorm
plt.rcParams.update({'font.size': 18})

########################################################################
#This code loads Ramses galaxy data using yt 
#and uses 3D Fourier transforms to calculate the angle-averaged power spectrum
#
#Also included is a function to calculate a helmholtz decomposition - which uses similar methods  
#
#################################################################


#set up velocity fields with rotation removed
# these function load the initial rotation curve of the galaxy and set up yt fields with it subtracted out.
#load rotation curve from file
def rot_curve(pname):
    vdata = np.loadtxt(pname+"vcirc.dat")
    nrow = np.argwhere(vdata[:,0]<40.).size
    r = vdata[:nrow,0]
    vr = vdata[:nrow,1]
    return r, vr

def linInterp(i, V, xd):
    C = V[i]*(1-xd) + V[i+1]*xd
    return C

#@derived_field(name="Rcyl", units="cm")
def _Rcyl(field, data):
    center = ds.domain_center.in_units('cm')
    return np.sqrt((data["x"]-center[0])**2 + (data["y"]-center[1])**2)

#@derived_field(name='Vr_interp', units="km/s")
def _Vr_interp(field, data):
    vpathname="../IC_spiral_t0/"
    r_rc, v_rc = rot_curve(vpathname)
    #r_rc, v_rc = profile.x,profile['velocity_cylindrical_theta']
    r_rc = (r_rc * yt.units.kpc).in_units('cm')
    v_rc = (v_rc * yt.units.km/yt.units.s)
    ind = np.fmin(np.fmax(r_rc.searchsorted(data['Rcyl'])-1, 0), r_rc.size-2)
    dr = np.array((data['Rcyl'] - r_rc[ind])/(r_rc[ind+1] - r_rc[ind]))
    Vr_interp = linInterp(ind, v_rc, dr)
    return Vr_interp

#@derived_field(name='Vx_r', units="code_velocity")
def _Vx_r(field, data):
    center = ds.domain_center.in_units('cm')
    return (data["x-velocity"].in_units('km/s') + data["Vr_interp"]/data["Rcyl"]*(data["y"]-center[1]))

#@derived_field(name='Vy_r', units="code_velocity")
def _Vy_r(field, data):
    center = ds.domain_center.in_units('cm')
    return (data["y-velocity"].in_units('km/s') - data["Vr_interp"]/data["Rcyl"]*(data["x"]-center[0]))


def helmholtz(vx,vy,vz):
# this function calculates the helmholtz decomposition of a 3D vector field
# see https://en.wikipedia.org/wiki/Helmholtz_decomposition
# I used this to anaylze curl and divergence free components of magnetic fields 
# vx, vy, and vz are 3D numpy arrays containing the 3 components of the field
#
# returns two vector fields - compressive and solenoidal components of original field
    nx, ny, nz = vx.shape

    #calculate fourier transforms
    print("Computing fourier transform on vx")
    vkx = np.fft.fftn(vx)

    print("Computing fourier transform on vy")
    vky = np.fft.fftn(vy)

    print("Computing fourier transform on vz")
    vkz = np.fft.fftn(vz)

    kx = np.fft.fftfreq(nx)
    ky = np.fft.fftfreq(ny)
    kz = np.fft.fftfreq(nz)
    kx3d,ky3d,kz3d= np.meshgrid(kx,ky,kz)


    k2 = kx3d**2 + ky3d**2 + kz3d**2
    k2[0,0,0] = 1. # ignore k=0 to avoid infinity

    #compute the decomposition
    print("Computing Helmholtz decomposition")
    kdotf = vkx*kx3d + vky*ky3d + vkz*kz3d

    v_comp_overk = kdotf / k2

    vkx_comp = (v_comp_overk * kx3d)
    vky_comp = (v_comp_overk * ky3d)
    vkz_comp = (v_comp_overk * kz3d)

    vx_comp = np.fft.ifftn(vkx_comp).real
    vy_comp = np.fft.ifftn(vky_comp).real
    vz_comp = np.fft.ifftn(vkz_comp).real

    vx_sol = np.fft.ifftn(vkx - vkx_comp).real
    vy_sol = np.fft.ifftn(vky - vky_comp).real
    vz_sol = np.fft.ifftn(vkz - vkz_comp).real

    #Check divergences -- solenoidal should have near zero
    divVs = np.fft.ifftn((np.fft.fftn(vx_sol) * kx + np.fft.fftn(vy_sol) * ky + np.fft.fftn(vz_sol) * kz))
    divVc = np.fft.ifftn((np.fft.fftn(vx_comp) * kx + np.fft.fftn(vy_comp) * ky + np.fft.fftn(vz_comp) * kz))
    print('div_solenoidal max:', np.max(abs(divVs)))
    print('div_compressive max:', np.max(abs(divVc)))

    #check total power 
    print('variance:')
    print('original field x,y,z:', vx.var(), vy.var(), vz.var())
    print('solenoidal x,y,z:', vx_sol.var(), vy_sol.var(), vz_sol.var())
    print('compressive x,y,z:', vx_comp.var(), vy_comp.var(), vz_comp.var())

    return(vx_comp,vy_comp,vz_comp,vx_sol,vy_sol,vz_sol)

def power_spectra(vx,vy,vz):
#This function takes a 3D vector field and returns the (1D)
# Angle-averaged power spectrum
#
# vx, vy, and vz are 3D numpy arrays containing the 3 components of the field
# returns:
# x (real space) bins
# k (fourier space) bins
# P total power per bin

    nx, ny, nz = vx.shape
    norm = nx * ny * nz
    print(norm)

    #DIVIDE BY NUMBER OF POINTS 
    #normaliztion must go before abs square
    fx = np.fft.fftn(vx)
    fx = fx/norm
    fx = np.abs(fx)**2
    fx = np.fft.fftshift(fx)


    fy = np.fft.fftn(vy)
    fy = fy/norm
    fy = np.abs(fy)**2
    fy = np.fft.fftshift(fy)


    fz = np.fft.fftn(vz)
    fz = fz/norm
    fz = np.abs(fz)**2
    fz = np.fft.fftshift(fz)

    kx = np.arange(-nx/2 , nx/2 , 1) / L[0]
    ky = np.arange(-ny/2 , ny/2 , 1) / L[1]
    kz = np.arange(-nz/2 , nz/2 , 1) / L[2]

    # physical limits to the wavenumbers
    kmin = np.min(1/L)
    kmax = np.min(0.5*delta/L)
    #bin the Fourier into radial kbins
    kx3d, ky3d, kz3d = np.meshgrid(kx, ky, kz, indexing="ij")
    k = np.sqrt(kx3d**2 + ky3d**2 +kz3d**2)

    Kk = fx + fy + fz


    kbins = np.arange(kmin, kmax, kmin)
    N = len(kbins)
    K,Kbins=np.histogram(k.flat,bins=kbins)    
    P,bins=np.histogram(k.flat,bins=kbins,weights=Kk.flat)
    #print(np.sum(P))

    #print(K,Kbins)
    #P2=P/K
    k = 0.5*(kbins[0:N-1] + kbins[1:N])
    x = 1/k
    return(x,k,P)

#choose which outputs to load
m=0
times=[190,180,170,160,140]
folder = 'no_boom_spiral/'
for i in times:
    m+=1
    fn=folder+"/output_{:05d}/info_{:05d}.txt".format(i,i) #highres outputs

    #set up data
    ds = yt.load(fn)
    ds.add_field(("Rcyl"), function=_Rcyl , units='cm')
    ds.add_field(("Vr_interp"), function=_Vr_interp , units='km/s')
    ds.add_field(("Vx_r"), function=_Vx_r , units='code_velocity')
    ds.add_field(("Vy_r"), function=_Vy_r , units='code_velocity')

    side = ds.quan(4.6875,'kpc')
    center = ds.domain_center

    left_corner = ds.domain_left_edge
    left_corner[0] = center[0] - side
    left_corner[1] = center[1] - side
    left_corner[2] = center[2] - side

    right_corner = ds.domain_left_edge
    right_corner[0] = center[0] + side
    right_corner[1] = center[1] + side
    right_corner[2] = center[2] + side

    box=ds.box(left_corner,right_corner)

    #levels of amr to create grid on, can only fit so many in memory, which is why we are cutting a smaller box
    max_level = ds.index.max_level-1 
    ref_factors=[2,2,2,2,2,2,2,2,2,2,2,2,2,2]
    ref = int(np.product(ref_factors[0:max_level]))
    L = (right_corner - left_corner).in_units('kpc').d
    ref = 2**(max_level)

    #Number of cells along each axis of resulting covering_grid
    # 1/64 of original domain = 9.375 kpc
    delta = ds.domain_dimensions*ref/64
    delta=delta.astype(int)
    print(ref,2**(max_level),ds.domain_dimensions,delta)
    print("Generating uniform grid of size:" + str(delta))


    grid = ds.smoothed_covering_grid(max_level, left_edge=left_corner,dims=delta,fields=['Vx_r','Vy_r','velocity_z','density'])
    rhohalf=grid['density']**(0.5)
    
    #*********generate window function, if using, multiply velocity fields by window3d
    #n=delta[0]
    #a = np.linspace(-(n-1)/2,(n-1)/2,n)
    #b = np.linspace(-(n-1)/2,(n-1)/2,n)
    #c = np.linspace(-(n-1)/2,(n-1)/2,n)
    #i,j,k = np.meshgrid(a, b, c)
    #r=np.sqrt(i**2+j**2+k**2)
    #window3d =  0.5+ 0.5*np.cos(2*np.pi*r/(n-1))
    #window3d[r>(n/2)] *= 1e-5
    ##****************

    # use velocity fields with rotation subtracted
    vx=grid['Vx_r'].in_units('km/s').d 
    vy=grid['Vy_r'].in_units('km/s').d 
    vz=grid['velocity_z'].in_units('km/s').d 


    print("Computing power spectrum")
    x,k,P = power_spectra(vx,vy,vz)


    #print("Computing compressive power spectrum")
    #x_comp,k_comp,P_comp = power_spectra(vx_comp,vy_comp,vz_comp)


    #print("Computing solenoidal power spectrum")
    #x_sol,k_sol,P_sol = power_spectra(vx_sol,vy_sol,vz_sol)

    # save results
    outarray=[k*L[0],P]
    np.save(folder+"power_spectra_velocity"+str(i),outarray)

