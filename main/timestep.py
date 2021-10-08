import numpy as np
import time as timer
import variables as var
# import cupy as cp


nonlinear_ssp_rk_switch = {
    2: [[1 / 2, 1 / 2, 1 / 2]],
    3: [[3 / 4, 1 / 4, 1 / 4],
        [1 / 3, 2 / 3, 2 / 3]]
}


class Stepper:
    def __init__(self, dt, resolutions, order, steps, flux):
        self.x_res, self.u_res, self.v_res = resolutions
        self.resolutions = resolutions
        self.order = order
        self.dt = dt
        self.steps = steps
        self.flux = flux  # fx.DGFlux(resolutions=resolutions, order=order)

        # RK coefficients
        self.rk_coefficients = np.array(nonlinear_ssp_rk_switch.get(3, "nothing"))

        # tracking arrays
        self.time = 0
        self.next_time = 0
        self.field_energy = np.array([])
        self.time_array = np.array([])
        self.thermal_energy = np.array([])
        self.density_array = np.array([])

    def main_loop(self, distribution, elliptic, grid):
        print('Beginning main loop')
        t0 = timer.time()
        for i in range(self.steps):
            # self.next_time = self.time + self.step
            # span = [self.time, self.next_time]
            self.ssprk3(distribution=distribution, elliptic=elliptic, grid=grid)
            self.time += self.dt
            if i % 50 == 0:
                self.time_array = np.append(self.time_array, self.time)
                elliptic.poisson_solve(distribution=distribution, grid=grid)
                self.field_energy = np.append(self.field_energy, elliptic.compute_field_energy(grid=grid))
                self.thermal_energy = np.append(self.thermal_energy, distribution.total_thermal_energy(grid=grid))
                self.density_array = np.append(self.density_array, distribution.total_density(grid=grid))
                print('\nTook x steps, time is {:0.3e}'.format(self.time))
                print('Time since start is ' + str((timer.time() - t0) / 60.0) + ' minutes')

            print('\nAll done at time is {:0.3e}'.format(self.time))
            print('Total steps were ' + str(self.steps))
            print('Time since start is {:0.3e}'.format((timer.time() - t0)))

    def ssprk3(self, distribution, elliptic, grid):
        stage0 = var.Distribution(resolutions=self.resolutions, order=self.order)
        stage1 = var.Distribution(resolutions=self.resolutions, order=self.order)
        # zero stage
        self.flux.semi_discrete_rhs(distribution=distribution, elliptic=elliptic, grid=grid)
        stage0.arr = distribution.arr + self.dt * self.flux.output.arr
        # first stage
        self.flux.semi_discrete_rhs(distribution=stage0, elliptic=elliptic, grid=grid)
        stage1.arr = (
                self.rk_coefficients[0, 0] * distribution.arr +
                self.rk_coefficients[0, 1] * stage0.arr +
                self.rk_coefficients[0, 2] * self.dt * self.flux.output.arr
        )
        # second stage
        self.flux.semi_discrete_rhs(distribution=stage1, elliptic=elliptic, grid=grid)
        distribution.arr = (
                self.rk_coefficients[1, 0] * distribution.arr +
                self.rk_coefficients[1, 1] * stage1.arr +
                self.rk_coefficients[1, 2] * self.dt * self.flux.output.arr
        )
