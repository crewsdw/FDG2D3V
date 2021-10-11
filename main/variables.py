import cupy as cp


class SpaceScalar:
    def __init__(self, resolutions):
        self.res_x, self.res_y = resolutions
        self.arr_nodal, self.arr_spectral = None, None

    def fourier_transform(self):
        self.arr_spectral = cp.fft.fftshift(cp.fft.rfft2(self.arr_nodal, norm='forward'), axes=0)

    def inverse_fourier_transform(self):
        self.arr_nodal = cp.fft.irfft2(cp.fft.fftshift(self.arr_spectral, axes=0), norm='forward')

    def integrate(self, grid):
        # arr_add = cp.append(self.arr_nodal, self.arr_nodal[0])
        arr_add = cp.zeros((self.res_x + 1, self.res_y + 1))
        arr_add[:-1, :-1] = self.arr_nodal
        arr_add[-1, :-1] = self.arr_nodal[0, :]
        arr_add[:-1, -1] = self.arr_nodal[:, 0]
        arr_add[-1, -1] = self.arr_nodal[0, 0]
        return trapz(arr_add, grid.x.dx, grid.z.dx)
    #
    # def spectral_magnitude_squared(self):
    #     return self.arr_spectral ** 2.0
    #
    # def integrate_energy(self):
    #     return cp.sum(self.spectral_magnitude_squared())

    # def integrate_energy(self, grid):
    #     arr = 0.5 * self.arr_nodal ** 2.0
    #     arr_add = cp.append(arr, arr[0])
    #     return trapz(arr_add, grid.x.dx, grid.z.dx)


class SpaceVector:
    def __init__(self, resolutions):
        self.res_x, self.res_y = resolutions
        self.arr_nodal, self.arr_spectral = None, None
        self.arr_nodal = cp.zeros((2, resolutions[0], resolutions[1]))
        self.init_spectral_array()

        self.energy = SpaceScalar(resolutions=resolutions)

    def init_spectral_array(self):
        if self.arr_spectral is not None:
            return
        else:
            x_spec = cp.fft.rfft2(self.arr_nodal[0, :, :])
            y_spec = cp.fft.rfft2(self.arr_nodal[1, :, :])
            self.arr_spectral = cp.array([x_spec, y_spec])

    def fourier_transform(self):
        self.arr_spectral[0, :, :] = cp.fft.fftshift(cp.fft.rfft2(self.arr_nodal[0, :, :], norm='forward'), axes=0)
        self.arr_spectral[1, :, :] = cp.fft.fftshift(cp.fft.rfft2(self.arr_nodal[1, :, :], norm='forward'), axes=0)

    def inverse_fourier_transform(self):
        self.arr_nodal[0, :, :] = cp.fft.irfft2(cp.fft.fftshift(self.arr_spectral[0, :, :], axes=0), norm='forward')
        self.arr_nodal[1, :, :] = cp.fft.irfft2(cp.fft.fftshift(self.arr_spectral[1, :, :], axes=0), norm='forward')

    # def spectral_magnitude_squared(self):
    #     return cp.absolute(self.arr_spectral[0, :, :]) ** 2.0 + cp.absolute(self.arr_spectral[1, :, :]) ** 2.0
    #
    # def magnitude_squared(self):
    #     return cp.absolute(self.arr_spectral[0, :, :]) ** 2.0 + cp.absolute(self.arr_spectral[1, :, :]) ** 2.0

    def integrate_energy(self, grid):
        self.inverse_fourier_transform()
        self.energy.arr_nodal = self.arr_nodal[0, :, :] ** 2.0 + self.arr_nodal[1, :, :] ** 2.0
        return 0.5 * self.energy.integrate(grid=grid)
        # return 0.5 * cp.sum(self.spectral_magnitude_squared()) * (4.0 * cp.pi ** 2.0) / (self.res_x * self.res_y)


class Distribution:
    def __init__(self, resolutions, order):
        self.x_res, self.z_res, self.u_res, self.v_res, self.w_res = resolutions
        self.order = order

        # arrays
        self.arr, self.arr_nodal = None, None
        self.zero_moment = SpaceScalar(resolutions=[self.x_res, self.z_res])
        self.second_moment = SpaceScalar(resolutions=[self.x_res, self.z_res])

    def fourier_transform(self):
        self.arr = cp.fft.fftshift(cp.fft.rfft2(self.arr_nodal, axes=(0, 1), norm='forward'), axes=0)
        # print(self.arr.shape)

    def inverse_fourier_transform(self):
        self.arr_nodal = cp.fft.irfft2(cp.fft.fftshift(self.arr, axes=0), axes=(0, 1), norm='forward')
        # print(self.arr_nodal.shape)

    def total_density(self, grid):
        # self.inverse_fourier_transform()
        self.compute_zero_moment(grid=grid)
        return self.zero_moment.integrate(grid=grid)

    def compute_zero_moment(self, grid):
        # self.inverse_fourier_transform()
        self.zero_moment.arr_spectral = (
            grid.u.zero_moment(
                function=grid.v.zero_moment(
                    function=grid.w.zero_moment(
                        function=self.arr,
                        idx=[6, 7]),
                    idx=[4, 5]),
                idx=[2, 3])
        )
        self.zero_moment.inverse_fourier_transform()

    def total_thermal_energy(self, grid):
        self.inverse_fourier_transform()
        integrand = grid.v_mag_sq[None, None, :, :, :, :, :, :] * self.arr_nodal
        self.second_moment.arr_nodal = (
            grid.u.zero_moment(
                function=grid.v.zero_moment(
                    function=grid.w.zero_moment(
                        function=integrand,
                        idx=[6, 7]),
                    idx=[4, 5]),
                idx=[2, 3])
        )
        return 0.5 * self.second_moment.integrate(grid=grid)

    def initialize(self, grid):
        ix, iz = cp.ones_like(grid.x.device_arr), cp.ones_like(grid.z.device_arr)

        ring_distribution = cp.tensordot(ix,
                                         cp.tensordot(iz,
                                                      grid.ring_distribution(perp_vt=1.0,
                                                                             ring_parameter=0,
                                                                             para_vt=1.0),
                                                      axes=0),
                                         axes=0)

        # compute perturbation
        # Examples: L = 2pi, first mode:  1.16387241 + 0j
        #           L = pi, first mode: 1.03859465
        #           L = pi, second mode: 2.05498248
        #           L = pi, third mode: 3.04616847
        # perturbation = grid.eigenfunction(thermal_velocity=1,
        #                                   ring_parameter=2.0 * cp.pi,
        #                                   eigenvalue=-3.48694202e-01j,
        #                                   parity=False)
        # sin_x = cp.sin(grid.x.fundamental * grid.x.device_arr)
        # sin_z = cp.sin(grid.z.fundamental * grid.z.device_arr)
        # sin_xz = cp.sin(grid.x.fundamental * grid.x.device_arr[:, None] +
        #                 grid.z.fundamental * grid.z.device_arr[None, :])

        # perturbation = cp.multiply((sin_x[:, None, None, None, None, None, None, None] +
        #                             sin_z[None, :, None, None, None, None, None, None]), ring_distribution)
        # perturbation = sin_xz[:, :, None, None, None, None, None, None] * ring_distribution
        eigenvalue = -1.285 + 0j  # -4.13247520e-01 - 1j * 1.0e-16
        perturbation = grid.eigenfunction(1, 0, 1, eigenvalue=eigenvalue, parity=True)

        self.arr_nodal = ring_distribution + 1.0e-1 * perturbation


def trapz(f, dx, dz):
    """ Custom trapz routine using cupy """
    sum_z = cp.sum(f[:, :-1] + f[:, 1:], axis=1) * dz / 2.0
    return cp.sum(sum_z[:-1] + sum_z[1:]) * dx / 2.0
