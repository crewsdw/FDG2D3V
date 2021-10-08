import numpy as np
import cupy as cp
import grid as g
import variables as var
import elliptic as ell
import plotter as my_plt
# import fluxes as fx
# import time as timer
# import timestep as ts

# elements and order
elements, order = [4, 4, 10, 10, 10], 8

# parameters
om_pc = 1
ring_param = 0

# grid parameters
k_perp = 0.5
k_para = 0.25
length_x = 2.0 * np.pi / k_perp
length_z = 2.0 * np.pi / k_para

# grid arrays
lows = np.array([-length_x / 2.0, -length_z / 2.0, -4, -4, -4])
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
plotter.spatial_scalar_plot(scalar=elliptic.potential)
plotter.spatial_scalar_plot(scalar=distribution.zero_moment)
plotter.show()
