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

    def scalar_plot(self, scalar, spectrum=True):
        if scalar.arr_nodal is None:
            scalar.inverse_fourier_transform()

        cb = cp.linspace(cp.amin(scalar.arr_nodal), cp.amax(scalar.arr_nodal), num=100).get()

        plt.figure()
        plt.contourf(self.X, self.Z, scalar.arr_nodal.get(), cb, cmap=self.colormap)
        plt.xlabel('x'), plt.ylabel('z')
        plt.colorbar(), plt.tight_layout()

        if spectrum:
            spectrum_abs = np.absolute(scalar.arr_spectral.get())

            cb_x = np.linspace(np.amin(spectrum_abs), np.amax(spectrum_abs), num=100)
            plt.figure()
            plt.contourf(self.KX, self.KZ, spectrum_abs, cb_x)
            plt.colorbar(), plt.tight_layout()

    def vector_plot(self, vector, spectrum=False):
        if vector.arr_nodal is None:
            vector.inverse_fourier_transform()

        cb_x = cp.linspace(cp.amin(vector.arr_nodal[0, :, :]), cp.amax(vector.arr_nodal[0, :, :]), num=100).get()
        cb_z = cp.linspace(cp.amin(vector.arr_nodal[1, :, :]), cp.amax(vector.arr_nodal[1, :, :]), num=100).get()

        plt.figure()
        plt.contourf(self.X, self.Z, vector.arr_nodal[0, :, :].get(), cb_x, cmap=self.colormap)
        plt.xlabel('x'), plt.ylabel('z')
        plt.colorbar(), plt.tight_layout()

        plt.figure()
        plt.contourf(self.X, self.Z, vector.arr_nodal[1, :, :].get(), cb_z, cmap=self.colormap)
        plt.xlabel('x'), plt.ylabel('z')
        plt.colorbar(), plt.tight_layout()

        if spectrum:
            print('NotImplemented')
            # spectrum_abs = np.absolute(scalar.arr_spectral.get())
            #
            # cb_x = np.linspace(np.amin(spectrum_abs), np.amax(spectrum_abs), num=100)
            # plt.figure()
            # plt.contourf(self.KX, self.KZ, spectrum_abs, cb_x)
            # plt.colorbar(), plt.tight_layout()

    def time_series_plot(self, time_in, series_in, y_axis, log=False, give_rate=False):
        time, series = time_in, series_in.get()  # / self.length
        plt.figure()
        if log:
            plt.semilogy(time, series, 'o--')
        else:
            plt.plot(time, series, 'o--')
        plt.xlabel('Time')
        plt.ylabel(y_axis)
        plt.grid(True), plt.tight_layout()
        if give_rate:
            lin_fit = np.polyfit(time, np.log(series), 1)
            exact = 2 * 0.1 * 3.48694202e-01
            print('\nNumerical rate: {:0.10e}'.format(lin_fit[0]))
            # print('cf. exact rate: {:0.10e}'.format(2 * 2.409497728e-01))  #
            print('cf. exact rate: {:0.10e}'.format(exact))
            print('The difference is {:0.10e}'.format(lin_fit[0] - exact))

    def show(self):
        plt.show()
