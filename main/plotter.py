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

    def time_series_plot(self, time_in, series_in, y_axis, log=False, give_rate=False, axis=False):
        time, series = time_in, series_in.get() / (self.length_x * self.length_z)
        plt.figure()
        if log:
            plt.semilogy(time, series, 'o--')
        else:
            plt.plot(time, series, 'o--')
        plt.xlabel('Time')
        plt.ylabel(y_axis)
        plt.grid(True), plt.tight_layout()
        if axis:
            plt.axis([0, time[-1], 0, 1.1*np.amax(series)])
        if give_rate:
            lin_fit = np.polyfit(time, np.log(series), 1)
            exact = 2 * 0.1 * 3.48694202e-01
            print('\nNumerical rate: {:0.10e}'.format(lin_fit[0]))
            # print('cf. exact rate: {:0.10e}'.format(2 * 2.409497728e-01))  #
            print('cf. exact rate: {:0.10e}'.format(exact))
            print('The difference is {:0.10e}'.format(lin_fit[0] - exact))

    def show(self):
        plt.show()


class Plotter3D:
    """
    Plots objects on 3D piecewise (as in DG) grid
    """

    def __init__(self, grid):
        # Build structured grid, full space
        (iu, iv, iw) = (cp.ones(grid.u.elements * grid.u.order),
                        cp.ones(grid.v.elements * grid.v.order),
                        cp.ones(grid.w.elements * grid.w.order))
        # modified_x = cp.append(grid.x.device_arr, grid.x.device_arr[-1] + grid.x.dx)
        # (x3, u3, v3) = (outer3(a=modified_x, b=iu, c=iv),
        #                 outer3(a=ix, b=grid.u.device_arr.flatten(), c=iv),
        #                 outer3(a=ix, b=iu, c=grid.v.device_arr.flatten()))
        (u3, v3, w3) = (outer3(a=grid.u.device_arr.flatten(), b=iu, c=iv),
                        outer3(a=iu, b=grid.v.device_arr.flatten(), c=iv),
                        outer3(a=iu, b=iu, c=grid.w.device_arr.flatten()))
        self.grid = pv.StructuredGrid(u3, v3, w3)

        # build structured grid, spectral space
        # ix2 = cp.ones(grid.x.modes)
        # u3_2, v3_2 = (outer3(a=ix2, b=grid.u.device_arr.flatten(), c=iv),
        #               outer3(a=ix2, b=iu, c=grid.v.device_arr.flatten()))
        # k3 = outer3(a=grid.x.device_wavenumbers, b=iu, c=iv)
        # self.spectral_grid = pv.StructuredGrid(k3, u3_2, v3_2)

    def distribution_contours3d(self, distribution, spectral_idx, real):
        """
        plot contours of a scalar function f=f(x,y,z) on Plotter3D's grid
        """
        if real:
            new_dist = np.real(distribution.arr[spectral_idx[0], spectral_idx[1], :, :, :].get())
        else:
            new_dist = np.imag(distribution.arr[spectral_idx[0], spectral_idx[1], :, :, :].get())

        self.grid['.'] = new_dist.reshape((new_dist.shape[0]*new_dist.shape[1], new_dist.shape[2]*new_dist.shape[3],
                                           new_dist.shape[4]*new_dist.shape[5])).transpose().flatten()

        contours = np.linspace(-0.8*np.amax(new_dist), 0.8*np.amax(new_dist), num=150)
        plot_contours = self.grid.contour(contours)

        # Create plot
        p = pv.Plotter()
        p.add_mesh(plot_contours, cmap='summer', show_scalar_bar=True, opacity=0.9)
        p.show_grid()
        p.show()  # auto_close=False)


def outer3(a, b, c):
    """
    Compute outer tensor product of vectors a, b, and c
    :param a: vector a_i
    :param b: vector b_j
    :param c: vector c_k
    :return: tensor a_i b_j c_k as numpy array
    """
    return cp.tensordot(a, cp.tensordot(b, c, axes=0), axes=0).get()

