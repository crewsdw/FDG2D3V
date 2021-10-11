import numpy as np
import cupy as cp
import basis as b
import scipy.special as sp


class SpaceGrid:
    """ In this scheme, the spatial grid is uniform and transforms are accomplished by DFT """

    def __init__(self, low, high, elements, real_freqs=False):
        # grid limits and elements
        self.low, self.high = low, high
        self.elements = elements

        # domain and element widths
        self.length = self.high - self.low
        self.dx = self.length / self.elements

        # element Jacobian
        self.J = 2.0 / self.dx

        # arrays
        self.arr, self.device_arr = None, None
        self.create_grid()

        # spectral properties
        self.fundamental = 2.0 * np.pi / self.length
        if real_freqs:
            self.modes = int(elements // 2.0 + 1)  # Nyquist frequency
            self.wavenumbers = self.fundamental * np.arange(self.modes)
            # de-aliasing parameters
            self.zero_idx = 0
            self.pad_width = int((1 * self.modes) // 3 + 1)
        else:
            self.half_modes = int(elements // 2.0)
            self.wavenumbers = self.fundamental * np.arange(-self.half_modes, self.half_modes)
            self.modes = self.wavenumbers.shape[0]
            self.zero_idx = self.half_modes
            self.pad_width = int((1 * self.modes) // 3 + 1)
        # send to device
        # print(self.modes)
        # print(self.pad_width)
        self.device_wavenumbers = cp.array(self.wavenumbers)

    def create_grid(self):
        """ Build evenly spaced grid, assumed periodic """
        self.arr = np.linspace(self.low, self.high - self.dx, num=self.elements)
        self.device_arr = cp.asarray(self.arr)


class VelocityGrid:
    """ In this experiment, the velocity grid is an LGL quadrature grid """

    def __init__(self, low, high, elements, order):
        self.low, self.high = low, high
        self.elements, self.order = elements, order
        self.local_basis = b.LGLBasis1D(order=self.order)
        # self.local_basis = b.GLBasis1D(order=self.order)

        # domain and element widths
        self.length = self.high - self.low
        self.dx = self.length / self.elements

        # jacobian
        self.J = 2.0 / self.dx

        # global quad weights
        self.global_quads = cp.tensordot(cp.ones(elements),
                                         cp.asarray(self.local_basis.weights), axes=0)

        # arrays
        self.arr, self.device_arr = None, None
        self.mid_points = None
        self.create_grid()

        # global translation matrix
        self.translation_matrix = None
        self.set_translation_matrix()

    def create_grid(self):
        """ Build global grid """
        # translate to [0, 1]
        nodes_iso = (np.array(self.local_basis.nodes) + 1) / 2
        # element left boundaries (including ghost elements)
        xl = np.linspace(self.low, self.high - self.dx, num=self.elements)
        # construct coordinates
        self.arr = np.zeros((self.elements, self.order))
        for i in range(self.elements):
            self.arr[i, :] = xl[i] + self.dx * np.array(nodes_iso)
        # send to device
        self.device_arr = cp.asarray(self.arr)
        self.mid_points = np.array([0.5 * (self.arr[i, -1] + self.arr[i, 0]) for i in range(self.elements)])

    def set_translation_matrix(self):
        mid_identity = np.tensordot(self.mid_points, np.eye(self.local_basis.order), axes=0)
        self.translation_matrix = cp.asarray(mid_identity + self.local_basis.translation_matrix / self.J)

    def zero_moment(self, function, idx):
        return cp.tensordot(self.global_quads, function, axes=([0, 1], idx)) / self.J

    def second_moment(self, function, dim, idx):
        if dim == 2:
            return cp.tensordot(self.global_quads, cp.multiply(self.device_arr[None, None, None, :, :] ** 2.0,
                                                               function),
                                axes=([0, 1], idx)) / self.J
        if dim == 1:
            return cp.tensordot(self.global_quads, cp.multiply(self.device_arr[None, :, :] ** 2.0,
                                                               function),
                                axes=([0, 1], idx)) / self.J

    def compute_maxwellian(self, thermal_velocity, drift_velocity):
        return cp.exp(-0.5 * ((self.device_arr - drift_velocity) /
                              thermal_velocity) ** 2.0) / (np.sqrt(2.0 * np.pi) * thermal_velocity)

    def compute_maxwellian_gradient(self, thermal_velocity, drift_velocity):
        return (-1.0 * ((self.device_arr - drift_velocity) / thermal_velocity ** 2.0) *
                self.compute_maxwellian(thermal_velocity=thermal_velocity, drift_velocity=drift_velocity))


class PhaseSpace:
    """ In this experiment, PhaseSpace consists of equispaced spatial nodes and a
    LGL tensor-product grid in truncated velocity space """

    def __init__(self, lows, highs, elements, order, om_pc):
        # Grids
        self.x = SpaceGrid(low=lows[0], high=highs[0], elements=elements[0])
        self.z = SpaceGrid(low=lows[1], high=highs[1], elements=elements[1], real_freqs=True)
        self.u = VelocityGrid(low=lows[2], high=highs[2], elements=elements[2], order=order)
        self.v = VelocityGrid(low=lows[3], high=highs[3], elements=elements[3], order=order)
        self.w = VelocityGrid(low=lows[4], high=highs[4], elements=elements[4], order=order)

        # Square quantities
        self.v_mag_sq = (self.u.device_arr[:, :, None, None, None, None] ** 2.0 +
                         self.v.device_arr[None, None, :, :, None, None] ** 2.0 +
                         self.w.device_arr[None, None, None, None, :, :] ** 2.0)
        self.k_sq = (self.x.device_wavenumbers[:, None] ** 2.0 + self.z.device_wavenumbers[None, :] ** 2.0)

        # Parameters
        self.om_pc = om_pc  # cyclotron freq. ratio

    def ring_distribution(self, perp_vt, ring_parameter, para_vt):
        # Cylindrical coordinates grid set-up, using wave-number x.k1
        iu, iv, iw = np.ones_like(self.u.arr), np.ones_like(self.v.arr), np.ones_like(self.w.arr)
        u = np.tensordot(self.u.arr, np.tensordot(iv, iw, axes=0), axes=0)
        v = np.tensordot(iu, np.tensordot(self.v.arr, iw, axes=0), axes=0)
        w = np.tensordot(iu, np.tensordot(iv, self.w.arr, axes=0), axes=0)
        r = np.sqrt(u ** 2.0 + v ** 2.0)

        # Set perpendicular distribution
        x = 0.5 * (r / perp_vt) ** 2.0
        perp_factor = 1 / (2.0 * np.pi * (perp_vt ** 2.0) * sp.gamma(ring_parameter + 1.0))
        ring = perp_factor * np.multiply(x ** ring_parameter, np.exp(-x))

        # Set parallel distribution
        para_factor = 1 / np.sqrt(2.0 * np.pi * para_vt ** 2.0)
        maxwellian = para_factor * np.exp(-0.5 * w ** 2.0)

        return cp.asarray(np.multiply(ring, maxwellian))

    def eigenfunction(self, perp_vt, ring_parameter, para_vt, eigenvalue, parity):
        # Cylindrical coordinates grid set-up, using wave-number x.k1
        iu, iv, iw = np.ones_like(self.u.arr), np.ones_like(self.v.arr), np.ones_like(self.w.arr)
        u = np.tensordot(self.u.arr, np.tensordot(iv, iw, axes=0), axes=0)
        v = np.tensordot(iu, np.tensordot(self.v.arr, iw, axes=0), axes=0)
        w = np.tensordot(iu, np.tensordot(iv, self.w.arr, axes=0), axes=0)
        r = np.sqrt(u ** 2.0 + v ** 2.0)
        phi = np.arctan2(v, u)

        # -k * v / om_c in units of debye length
        beta = - self.x.fundamental * r * self.om_pc

        # radial gradient of distribution
        if ring_parameter > 0:
            df_dv_perp = ((self.ring_distribution(perp_vt, ring_parameter - 1, para_vt) -
                          self.ring_distribution(perp_vt, ring_parameter, para_vt)) / (perp_vt ** 2.0)).get()
        else:
            df_dv_perp = (-1.0 / perp_vt ** 2.0) * self.ring_distribution(perp_vt, ring_parameter, para_vt).get()

        df_dv_para = np.multiply(-w / para_vt ** 2.0, self.ring_distribution(perp_vt, ring_parameter, para_vt).get())

        perp_series = 0 + 0j
        para_series = 0 + 0j
        k_para = -1.0 * self.z.fundamental / self.om_pc
        terms_n = 10
        if parity:
            om1 = eigenvalue
            om2 = -1.0 * np.real(eigenvalue) + 1j * np.imag(eigenvalue)
            frequencies = [om1, om2]
            for om in frequencies:
                upsilon_series = np.array([
                    n / (om - k_para * w - n) * np.multiply(sp.jv(n, beta), np.exp(-1j * n * phi))
                    for n in range(1 - terms_n, terms_n)]).sum(axis=0)
                perp_series += np.multiply(df_dv_perp, upsilon_series)

                lambda_series = np.array([
                    1 / (om - k_para * w - n) * np.multiply(sp.jv(n, beta), np.exp(-1j * n * phi))
                    for n in range(1 - terms_n, terms_n)]).sum(axis=0)
                para_series += np.multiply(df_dv_para, lambda_series)
        else:
            upsilon_series = np.array([
                n / (eigenvalue - k_para * w - n) * np.multiply(sp.jv(n, beta), np.exp(-1j * n * phi))
                for n in range(1 - terms_n, terms_n)]).sum(axis=0)
            perp_series += np.multiply(df_dv_perp, upsilon_series)

            lambda_series = np.array([
                1 / (eigenvalue - k_para * w - n) * np.multiply(sp.jv(n, beta), np.exp(-1j * n * phi))
                for n in range(1 - terms_n, terms_n)]).sum(axis=0)
            para_series += np.multiply(df_dv_para, lambda_series)

        # Construct total eigen mode
        vel_mode = -1j * np.exp(1j * beta * np.sin(phi)) * (perp_series + k_para * para_series)
        potential_phase = np.tensordot(np.exp(-1j * self.x.fundamental * self.x.arr),
                                       np.exp(-1j * self.z.fundamental * self.z.arr), axes=0)
        return cp.asarray(np.real(
            np.tensordot(potential_phase, vel_mode, axes=0)
        ))
