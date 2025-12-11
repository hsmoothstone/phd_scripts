import yt
import matplotlib.pyplot as plt
import numpy as np
from yt.units import kpc
from mpl_toolkits.axes_grid1 import AxesGrid
import multiprocessing as mp
import matplotlib as mpl
from matplotlib.colors import LogNorm

####################################################
#this script loads Ramses galaxy data using yt and generates visualzations of 
#stars, gas density, star formation, and magnetic field strength
# Uses python multiprocessing, with each output on a separate process
#####################################################

plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
mpl.use('Agg')

#Custom field definitions
def _logbx(field,data):
    bx=data[('gas','magnetic_field_x')]
    G = data.ds.quan(1, "gauss")
    sign = np.sign(bx)
    return sign * np.log10(sign*bx) 
def _logby(field,data):
    by=data[('gas','magnetic_field_y')]
    G = data.ds.quan(1, "gauss")
    sign = np.sign(by)
    return sign * np.log10(by) 

def oldstars(pfilter, data):
    m = data[pfilter.filtered_type, "particle_mass"]
    filter = m.in_units('Msun') < 1e6
    return filter
def young_stars(pfilter, data):
    age = data.ds.current_time - data[pfilter.filtered_type, "particle_birth_time"]
    filter = np.logical_and(age.in_units("Myr") <= 100, age >= 0)
    return filter

#folders that the data is located in
folders = ['toroidal/','dipole/','uniform/','hydro/']
titles = ['Toroidal','Dipole','Uniform','Hydro']
#add yt custom fields
yt.add_particle_filter("oldstars", function=oldstars,filtered_type='DM', requires=['particle_mass'])
yt.add_particle_filter("young_stars", function=young_stars,filtered_type='star', requires=['particle_birth_time'])
yt.add_field(("logbx"), function=_logbx ,units = "",sampling_type="cell")
yt.add_field(("logby"), function=_logby ,units = "",sampling_type="cell")

def plot(j):
    #set up figures
    fig = plt.figure()
    grid = AxesGrid(
        fig,
        (0.1, 0.1, 0.8, 0.8),
        nrows_ncols=(4, 4),
        axes_pad=0.05,
        label_mode="L",
        share_all=True,
        cbar_location="right",
        cbar_mode="edge",
        cbar_size="5%",
        cbar_pad="0%",
    )
    for i, fn in enumerate(folders):
        ds = yt.load(fn+"output_{:05d}/info_{:05d}.txt".format(j,j))
        ds.add_particle_filter('oldstars')
        ds.add_particle_filter('young_stars')
        center=yt.YTArray([0.5,0.5,0.5],'code_length',registry=ds.unit_registry)
        radius = ds.quan(45,'kpc')
        height = ds.quan(1,'kpc')
        normal =[0.0, 0.0, 1.0]

        disk=ds.disk(center, normal, radius, height)
        center = disk.quantities.center_of_mass(use_gas=True, use_particles=False).in_units("code_length")
        
        #first row - gas density##############################################
        #generate plot in yt
        p = yt.ProjectionPlot(ds, 'z', 'density',center=center,fontsize=8)
        p.set_cmap('density', 'rainbow')
        p.set_unit('density', 'Msun/pc**2')
        p.set_zlim('density', 0.5, 100)
        p.set_width(30*kpc)
        p.set_xlabel('')
        p.set_ylabel('')

        #access mpl plot manually
        plot = p.plots['density']
        plot.figure = fig
        plot.axes = grid[i+8].axes
        plot.cax = grid.cbar_axes[2]

        p._setup_plots()
        plot.cax.set_ylabel('$\Sigma_{gas}$ [M$_{\odot}$/pc$^2$]',size=10)
        plot.cax.tick_params(labelsize=8)
        plot.axes.yaxis.set_ticks([-10, 0,10])
        plot.axes.xaxis.set_ticks([-10, 0,10])
        plot.axes.set_yticklabels([])
        plot.axes.set_xticklabels([])

        #second row - old star density (particles from initial stellar disk)#######################
        p2 = yt.ProjectionPlot(ds, 'z', 'oldstars_density',center=center,fontsize=8)
        p2.set_cmap('oldstars_density', 'Greys')
        p2.set_unit('oldstars_density', 'Msun/pc**2')
        p2.set_zlim('oldstars_density', 8, 1700)
        p2.set_xlabel('')
        p2.set_ylabel('')
        p2.set_width(30*kpc)
        p2.annotate_scale(coeff=10,unit='kpc',text_args={'size':10},size_bar_args={'color':'black'},pos=[0.2,0.05])

        #access mpl plot manually
        plot2 = p2.plots['oldstars_density']
        plot2.figure = fig
        plot2.axes = grid[i].axes
        plot2.cax = grid.cbar_axes[0]

        p2._setup_plots()
        plot2.cax.set_ylabel('$\Sigma_{star}$ [M$_{\odot}$/pc$^2$]',size=10)
        plot2.cax.tick_params(labelsize=8)
        plot2.axes.yaxis.set_ticks([-10, 0,10])
        plot2.axes.xaxis.set_ticks([-10, 0,10])
        plot2.axes.set_yticklabels([])
        plot2.axes.set_xticklabels([])
        plot2.axes.set_title(titles[i],fontsize=12)
        
        # third row - magnetic field strength######################################
        # this field is plotted manually using pcolormesh rather than with the yt plot
        
        width = (30, "kpc") 
        num= 512
        x1 = np.linspace(-15,15,num)
        x,y = np.meshgrid(x1,x1,indexing = 'ij')

        res = [num, num]

        slc = yt.ProjectionPlot(ds, 'z', 'magnetic_field_strength',center=center,data_source=disk,weight_field='density',fontsize=8)
        slc_frb = slc.data_source.to_frb((30, "kpc"), num)
        B = slc_frb['magnetic_field_strength'].in_units('uG').T.d
        
        #generate a second projection of bx and by at lower resolution for the quiver lines
        slc2 = yt.ProjectionPlot(ds, 'z', 'magnetic_field_x',center=center,data_source=disk,weight_field='density',fontsize=8)
        slc2_frb = slc2.data_source.to_frb((30, "kpc"), int(num/16))

        x2 = np.linspace(-15,15,int(num/16))
        x3,y3 = np.meshgrid(x2,x2,indexing = 'ij')
        r = np.sqrt(x3**2+y3**2)
        bx = slc2_frb['magnetic_field_x'].in_units('uG').T.d
        by = slc2_frb['magnetic_field_y'].in_units('uG').T.d
        angles=np.arctan2(by,bx)*180.0/np.pi
        bx_plot = np.sign(bx) * np.log10(np.abs(bx*1000))
        by_plot = np.sign(by) * np.log10(np.abs(by*1000))
        

        pc=grid[i+12].pcolormesh(x,y,B,cmap='Greys',norm=LogNorm(),vmin=9e-3,vmax=1.5e2)
             
        grid[i+12].set_xticks([-10, 0,10])
        grid[i+12].set_yticks([-10, 0,10])
        grid[i+12].set_yticklabels([])
        grid[i+12].set_xticklabels([])
        grid[i+12].tick_params(axis='x', labelsize=8)
        grid[i+12].tick_params(axis='y', labelsize=8)

        cb=plt.colorbar(pc,cax=grid.cbar_axes[3],ticks = [0.01,0.1,1,10,100])
        cb.ax.tick_params(labelsize=8)
        cb.set_label(r'|B| [$\mu$G]',size=10)


        #fourth row - star formation rate##################################
        # this field is calculated by taking the surface density star formed in last 100 Myr and dividing by 100 Myr
        # units are Msun/kpc^2/yr - As in kennicutt schmidt relation
        proj3 = ds.proj('young_stars_density', 'z', data_source=disk)
        frb= proj3.to_frb(width, res, center=center)
        surface_density = frb['density'].in_cgs()
        surface_density_stars = frb['young_stars_density'].in_units("Msun/pc**2")
        pc2=grid[i+4].pcolormesh(x,y,surface_density_stars.T*1e6/1e8,cmap='Purples',norm=LogNorm(),vmin=0.0005,vmax=0.5)
        grid[i+4].set_xlabel('',fontsize=8)
        grid[i+4].set_ylabel('',fontsize=8)
        grid[i+4].set_xticks([-10, 0,10])
        grid[i+4].set_yticks([-10, 0,10])
        grid[i+4].set_yticklabels([])
        grid[i+4].set_xticklabels([])
        grid[i+4].tick_params(axis='x', labelsize=8)
        grid[i+4].tick_params(axis='y', labelsize=8)
        cb2=plt.colorbar(pc2,cax=grid.cbar_axes[1],ticks = [0.001,0.01,0.1])
        cb2.ax.tick_params(labelsize=8)
        cb2.set_label(r"$\Sigma_{SFR}$ [M$_{\odot}$kpc$^{-2}$yr$^{-1}$]",fontsize=10)
        
    #save plots
    plt.savefig('multipanel_new_{:05d}.png'.format(j),bbox_inches='tight',dpi=400)
    plt.close()

#main code - calls plot generation for each output with multiprocessing
#nums is the data file output numbers
nums=[160,170,180,190]
ncpus = 4 #int(os.environ.get('SLURM_CPUS_PER_TASK',default=1))
if __name__ == '__main__':
    pool = mp.Pool(processes=ncpus)
    pool.map(plot, nums)
