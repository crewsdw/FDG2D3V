import numpy as np
import cupy as cp
import grid as g
import variables as var
import elliptic as ell
import plotter as my_plt
import fluxes as fx
# import time as timer
import timestep as ts

# elements and order
elements, order = [4, 4, 15, 15, 15], 8

# parameters
om_pc = 1
ring_param = 0

# grid parameters
k_perp = 0.1  # 0.5
k_para = 0.1  # 0.25
length_x = 2.0 * np.pi / k_perp
length_z = 2.0 * np.pi / k_para

# grid arrays
lows = np.array([-length_x / 2.0, -length_z / 2.0, -7, -7, -7])
highs = -1.0 * lows

# grid object
grid = g.PhaseSpace(lows=lows, highs=highs, elements=elements, order=order, om_pc=om_pc)

# set up distribution
distribution = var.Distribution(resolutions=elements, order=order)
distribution.initialize(grid=grid)
distribution.fourier_transform(), distribution.inverse_fourier_transform()

# elliptic solver
elliptic = ell.Elliptic(resolutions=[elements[0], elements[1]])
elliptic.poisson(distribution=distribution, grid=grid)

# plotter
plotter = my_plt.Plotter2D(grid=grid)
# plotter.scalar_plot(scalar=elliptic.potential)
plotter.scalar_plot(scalar=distribution.zero_moment)
# # plotter.vector_plot(vector=elliptic.field)
plotter.show()

# plotter3d = my_plt.Plotter3D(grid=grid)
# plotter3d.distribution_contours3d(distribution=distribution, spectral_idx=[grid.x.zero_idx, grid.z.zero_idx],
#                                   real=True)
# plotter3d.distribution_contours3d(distribution=distribution, spectral_idx=[grid.x.zero_idx + 1, grid.z.zero_idx + 1],
#                                   real=True)

# Set up fluxes
flux = fx.DGFlux(resolutions=elements, order=order, grid=grid, om_pc=om_pc)
flux.initialize_zero_pad(grid=grid)

# Set up timestepper
dt = 3.0e-3
stop_time = 0.1  # 2.5  # np.pi
steps = int(stop_time // dt) + 1
stepper = ts.Stepper(dt=dt, resolutions=elements, order=order, steps=steps, flux=flux)
stepper.main_loop(distribution=distribution, elliptic=elliptic, grid=grid)

distribution.zero_moment.inverse_fourier_transform()
plotter.scalar_plot(scalar=distribution.zero_moment)
# plotter.vector_plot(vector=elliptic.field)

plotter.time_series_plot(time_in=stepper.time_array, series_in=stepper.field_energy,
                         y_axis='Electric energy', log=False, give_rate=False, axis=True)
plotter.time_series_plot(time_in=stepper.time_array, series_in=stepper.thermal_energy,
                         y_axis='Thermal energy', log=False, axis=True)
plotter.time_series_plot(time_in=stepper.time_array, series_in=stepper.density_array,
                         y_axis='Total density', log=False)
plotter.time_series_plot(time_in=stepper.time_array, series_in=stepper.field_energy + stepper.thermal_energy,
                         y_axis='Total energy', log=False)

plotter.show()
