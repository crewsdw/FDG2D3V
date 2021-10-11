import cupy as cp
import variables as var


def basis_product(flux, basis_arr, axis, permutation):
    return cp.transpose(cp.tensordot(flux, basis_arr,
                                     axes=([axis], [1])),
                        axes=permutation)


class DGFlux:
    def __init__(self, resolutions, order, grid, om_pc):
        self.x_ele, self.z_ele, self.u_res, self.v_res, self.w_res = resolutions
        self.x_res = grid.x.wavenumbers.shape[0]
        self.z_res = grid.z.wavenumbers.shape[0]
        self.order = order

        # permutations after tensor-dot with basis array
        self.permutations = [(0, 1, 2, 7, 3, 4, 5, 6),  # for contraction with u nodes
                             (0, 1, 2, 3, 4, 7, 5, 6),  # for contraction with v nodes
                             (0, 1, 2, 3, 4, 5, 6, 7)]  # for contraction with w nodes

        # slices into the DG boundaries (list of tuples)
        self.boundary_slices = [[(slice(self.x_res), slice(self.z_res),
                                  slice(self.u_res), 0,
                                  slice(self.v_res), slice(self.order),
                                  slice(self.w_res), slice(self.order)),
                                 (slice(self.x_res), slice(self.z_res),
                                  slice(self.u_res), -1,
                                  slice(self.v_res), slice(self.order),
                                  slice(self.w_res), slice(self.order))],
                                [(slice(self.x_res), slice(self.z_res),
                                  slice(self.u_res), slice(self.order),
                                  slice(self.v_res), 0,
                                  slice(self.w_res), slice(self.order)),
                                 (slice(self.x_res), slice(self.z_res),
                                  slice(self.u_res), slice(self.order),
                                  slice(self.v_res), -1,
                                  slice(self.w_res), slice(self.order))],
                                [(slice(self.x_res), slice(self.z_res),
                                  slice(self.u_res), slice(self.order),
                                  slice(self.v_res), slice(self.order),
                                  slice(self.w_res), 0),
                                 (slice(self.x_res), slice(self.z_res),
                                  slice(self.u_res), slice(self.order),
                                  slice(self.v_res), slice(self.order),
                                  slice(self.w_res), -1)]]
        self.boundary_slices_pad = [[(slice(self.x_res), slice(self.z_res),
                                      slice(self.u_res + 2), 0,
                                      slice(self.v_res), slice(self.order),
                                      slice(self.w_res), slice(self.order)),
                                     (slice(self.x_res), slice(self.z_res),
                                      slice(self.u_res + 2), -1,
                                      slice(self.v_res), slice(self.order),
                                      slice(self.w_res), slice(self.order))],
                                    [(slice(self.x_res), slice(self.z_res),
                                      slice(self.u_res), slice(self.order),
                                      slice(self.v_res + 2), 0,
                                      slice(self.w_res), slice(self.order)),
                                     (slice(self.x_res), slice(self.z_res),
                                      slice(self.u_res), slice(self.order),
                                      slice(self.v_res + 2), -1,
                                      slice(self.w_res), slice(self.order))],
                                    [(slice(self.x_res), slice(self.z_res),
                                      slice(self.u_res), slice(self.order),
                                      slice(self.v_res), slice(self.order),
                                      slice(self.w_res + 2), 0),
                                     (slice(self.x_res), slice(self.z_res),
                                      slice(self.u_res), slice(self.order),
                                      slice(self.v_res), slice(self.order),
                                      slice(self.w_res + 2), -1)]]
        self.flux_input_slices = [(slice(self.x_res), slice(self.z_res),
                                   slice(1, self.u_res + 1), slice(self.order),
                                   slice(self.v_res), slice(self.order),
                                   slice(self.w_res), slice(self.order)),
                                  (slice(self.x_res), slice(self.z_res),
                                   slice(self.u_res), slice(self.order),
                                   slice(1, self.v_res + 1), slice(self.order),
                                   slice(self.w_res), slice(self.order)),
                                  (slice(self.x_res), slice(self.z_res),
                                   slice(self.u_res), slice(self.order),
                                   slice(self.v_res), slice(self.order),
                                   slice(1, self.w_res + 1), slice(self.order))]
        self.pad_slices = [(slice(self.x_res), slice(self.z_res),
                            slice(1, self.u_res + 1),
                            slice(self.v_res), slice(self.order),
                            slice(self.w_res), slice(self.order)),
                           (slice(self.x_res), slice(self.z_res),
                            slice(self.u_res), slice(self.order),
                            slice(1, self.v_res + 1),
                            slice(self.w_res), slice(self.order)),
                           (slice(self.x_res), slice(self.z_res),
                            slice(self.u_res), slice(self.order),
                            slice(self.v_res), slice(self.order),
                            slice(1, self.w_res + 1))]
        self.num_flux_sizes = [(self.x_res, self.z_res, self.u_res, 2, self.v_res, self.order, self.w_res, self.order),
                               (self.x_res, self.z_res, self.u_res, self.order, self.v_res, 2, self.w_res, self.order),
                               (self.x_res, self.z_res, self.u_res, self.order, self.v_res, self.order, self.w_res, 2)]
        self.padded_flux_sizes = [(self.x_res, self.z_res, self.u_res + 2, self.order,
                                   self.v_res, self.order, self.w_res, self.order),
                                  (self.x_res, self.z_res, self.u_res, self.order,
                                   self.v_res + 2, self.order, self.w_res, self.order),
                                  (self.x_res, self.z_res, self.u_res, self.order,
                                   self.v_res, self.order, self.w_res + 2, self.order)]
        self.directions = [2, 4, 6]
        self.sub_elements = [3, 5, 7]

        # arrays
        self.field_flux_u = var.Distribution(resolutions=resolutions, order=order)
        self.field_flux_w = var.Distribution(resolutions=resolutions, order=order)
        self.output = var.Distribution(resolutions=resolutions, order=order)

        # magnetic field
        self.b_field = -1.0 / om_pc  # a constant

        self.pad_field = None
        self.pad_spectrum = None

    def semi_discrete_rhs(self, distribution, elliptic, grid):
        """ Computes the semi-discrete equation """
        # Do elliptic problem
        # t0 = timer.time()
        elliptic.poisson(distribution=distribution, grid=grid, invert=False)
        # t1 = timer.time()
        # # Compute the flux
        self.compute_flux(distribution=distribution, elliptic=elliptic, grid=grid)
        # t2 = timer.time()
        self.output.arr = (grid.u.J * self.u_flux_lgl(distribution=distribution, grid=grid) +
                           grid.v.J * self.v_flux_lgl(distribution=distribution, grid=grid) +
                           grid.w.J * self.w_flux_lgl(distribution=distribution, grid=grid) +
                           self.source_term_lgl(distribution=distribution, grid=grid))

    def initialize_zero_pad(self, grid):
        self.pad_field = cp.zeros((2, grid.x.modes + 2 * grid.x.pad_width, grid.z.modes + grid.z.pad_width)) + 0j
        self.pad_spectrum = cp.zeros((grid.x.modes + 2 * grid.x.pad_width, grid.z.modes + grid.z.pad_width,
                                      grid.u.elements, grid.u.order,
                                      grid.v.elements, grid.v.order,
                                      grid.w.elements, grid.w.order)) + 0j
        # print(self.pad_field.shape)
        # quit()

    def compute_flux(self, distribution, elliptic, grid):
        """ Compute the flux convolution(field, distribution) using pseudospectral method """
        # Dealias with two-thirds rule
        self.pad_field[:, grid.x.pad_width:-grid.x.pad_width, :-grid.z.pad_width] = elliptic.field.arr_spectral
        self.pad_spectrum[grid.x.pad_width:-grid.x.pad_width, :-grid.z.pad_width, :, :, :, :, :, :] = distribution.arr

        self.field_flux_u.arr = forward_distribution_transform(cp.multiply(
            inverse_field_transform(self.pad_field, dim=0)[:, :, None, None, None, None, None, None],
            inverse_distribution_transform(self.pad_spectrum))
        )[grid.x.pad_width:-grid.x.pad_width, :-grid.z.pad_width, :, :, :, :, :, :]

        self.field_flux_w.arr = forward_distribution_transform(cp.multiply(
            inverse_field_transform(self.pad_field, dim=1)[:, :, None, None, None, None, None, None],
            inverse_distribution_transform(self.pad_spectrum))
        )[grid.x.pad_width:-grid.x.pad_width, :-grid.z.pad_width, :, :, :, :, :, :]

        # For now, no de-aliasing technique is used
        # Pseudospectral product
        # elliptic.field.inverse_fourier_transform()
        # distribution.inverse_fourier_transform()
        # nodal flux
        # self.field_flux_u.arr_nodal = cp.multiply(
        #     elliptic.field.arr_nodal[0, :, :, None, None, None, None, None, None],
        #     distribution.arr_nodal
        # )
        # self.field_flux_w.arr_nodal = cp.multiply(
        #     elliptic.field.arr_nodal[1, :, :, None, None, None, None, None, None],
        #     distribution.arr_nodal
        # )
        # # inverse transform
        # self.field_flux_u.fourier_transform()
        # self.field_flux_w.fourier_transform()

    def u_flux_lgl(self, distribution, grid):
        u_flux = (-1.0 * self.field_flux_u.arr + self.b_field *
                  cp.multiply(grid.v.device_arr[None, None, None, None, :, :, None, None], distribution.arr))
        return (basis_product(flux=u_flux, basis_arr=grid.u.local_basis.internal,
                              axis=3, permutation=self.permutations[0]) -
                self.numerical_flux_lgl(flux=u_flux, grid=grid, dim=0))

    def v_flux_lgl(self, distribution, grid):
        v_flux = -self.b_field * grid.u.device_arr[None, None, :, :, None, None, None, None] * distribution.arr
        return (basis_product(flux=v_flux, basis_arr=grid.v.local_basis.internal,
                              axis=5, permutation=self.permutations[1]) -
                self.numerical_flux_lgl(flux=v_flux, grid=grid, dim=1))

    def w_flux_lgl(self, distribution, grid):
        return (basis_product(flux=-1.0 * self.field_flux_w.arr, basis_arr=grid.w.local_basis.internal,
                              axis=7, permutation=self.permutations[2]) -
                self.numerical_flux_lgl(flux=-1.0 * self.field_flux_w.arr, grid=grid, dim=2))

    def source_term_lgl(self, distribution, grid):
        return -1j * (cp.multiply(grid.x.device_wavenumbers[:, None, None, None, None, None, None, None],
                                  cp.einsum('axb,ijabcdef->ijaxcdef', grid.u.translation_matrix, distribution.arr)) +
                      cp.multiply(grid.z.device_wavenumbers[None, :, None, None, None, None, None, None],
                                  cp.einsum('exf,ijabcdef->ijabcdex', grid.w.translation_matrix, distribution.arr)))

    def numerical_flux_lgl(self, flux, grid, dim):
        # Allocate
        num_flux = cp.zeros(self.num_flux_sizes[dim]) + 0j

        # set padded flux
        padded_flux = cp.zeros(self.padded_flux_sizes[dim]) + 0j
        padded_flux[self.flux_input_slices[dim]] = flux

        # Compute a central flux
        num_flux[self.boundary_slices[dim][0]] = -1.0 * (cp.roll(padded_flux[self.boundary_slices_pad[dim][1]],
                                                                 shift=+1,
                                                                 axis=self.directions[dim])[self.pad_slices[dim]] +
                                                         flux[self.boundary_slices[dim][0]]) / 2.0
        num_flux[self.boundary_slices[dim][1]] = (cp.roll(padded_flux[self.boundary_slices_pad[dim][0]],
                                                          shift=-1,
                                                          axis=self.directions[dim])[self.pad_slices[dim]] +
                                                  flux[self.boundary_slices[dim][1]]) / 2.0

        return basis_product(flux=num_flux, basis_arr=grid.v.local_basis.numerical,
                             axis=self.sub_elements[dim], permutation=self.permutations[dim])


def inverse_field_transform(field, dim):
    return cp.fft.irfft2(cp.fft.fftshift(field[dim, :, :], axes=0), norm='forward')


def inverse_distribution_transform(distribution):
    return cp.fft.irfft2(cp.fft.fftshift(distribution, axes=0), axes=(0, 1), norm='forward')


def forward_distribution_transform(nodal_array):
    return cp.fft.fftshift(cp.fft.rfft2(nodal_array, axes=(0, 1), norm='forward'), axes=0)
