import cupy as cp
import variables as var


class Elliptic:
    def __init__(self, resolutions):
        self.potential = var.SpaceScalar(resolutions=resolutions)
        self.field = var.SpaceVector(resolutions=resolutions)

    def poisson(self, distribution, grid, invert=True):
        # Compute zeroth moment, integrate(c_n(v)dv)
        distribution.compute_zero_moment(grid=grid)

        # Compute potential and field spectrum
        self.potential.arr_spectral = -1.0 * cp.nan_to_num(cp.divide(distribution.zero_moment.arr_spectral, grid.k_sq))
        self.field.arr_spectral[0, :, :] = -1j * cp.multiply(grid.x.device_wavenumbers[:, None],
                                                             self.potential.arr_spectral)
        self.field.arr_spectral[1, :, :] = -1j * cp.multiply(grid.z.device_wavenumbers[None, :],
                                                             self.potential.arr_spectral)

        if invert:
            self.potential.inverse_fourier_transform()
            self.field.inverse_fourier_transform()

    def compute_field_energy(self, grid):
        return self.field.integrate_energy(grid=grid)
