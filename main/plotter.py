import cupy as cp
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt


class Plotter2D:
    def __init__(self, grid, colormap='RdPu'):
        self.colormap = colormap
        self.grid = grid
        # Build structured grid, nodal
        # self.U, self.V = np.meshgrid(grid.u.arr.flatten(), grid.v.arr.flatten(), indexing='ij')
        self.X, self.Z = np.meshgrid(grid.x.arr.flatten(), grid.z.arr.flatten(), indexing='ij')
        self.KX, self.KZ = np.meshgrid(grid.x.wavenumbers, grid.z.wavenumbers, indexing='ij')
        # self.k = grid.x.wavenumbers / grid.x.fundamental
        self.length_x, self.length_z = grid.x.length, grid.z.length

    def spatial_scalar_plot(self, scalar, spectrum=True):
        if scalar.arr_nodal is None:
            scalar.inverse_fourier_transform()

        cb = np.linspace(np.amin(scalar.arr_nodal), np.amax(scalar.arr_nodal), num=100)

        plt.figure()
        plt.contourf(self.X, self.Z, scalar.arr_nodal, cb, cmap=self.colormap)
        plt.xlabel('x'), plt.ylabel('z')
        plt.tight_layout()

        if spectrum:
            spectrum_abs = np.absolute(scalar.arr_spectral)

            cb_x = np.linspace(np.amin(spectrum_abs), np.amax(spectrum_abs), num=100)
            plt.figure()
            plt.contourf(self.KX, self.KZ, spectrum_abs, cb_x)
            plt.tight_layout()

    def show(self):
        plt.show()
