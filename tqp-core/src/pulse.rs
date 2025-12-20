use crate::state::TQPState;
use ndarray::{Array1, Array2};
use num_complex::Complex64;

/// Pulse Solver (Hamiltonian Simulation)
/// Simulates the continuous time evolution of the quantum state under a time-dependent Hamiltonian.
/// d|psi>/dt = -i H(t) |psi>
/// Uses 4th-order Runge-Kutta (RK4) method.
pub struct PulseSolver {
    pub dt: f64, // Time step (ns)
}

impl PulseSolver {
    pub fn new(dt: f64) -> Self {
        PulseSolver { dt }
    }

    /// Evolves the state for a total duration `duration` under the given Hamiltonian.
    /// H_func: A closure that returns the Hamiltonian matrix H(t) at time t.
    pub fn evolve<F>(&self, state: &mut TQPState, duration: f64, h_func: F)
    where
        F: Fn(f64) -> Array2<Complex64>,
    {
        let steps = (duration / self.dt).ceil() as usize;
        let mut t = 0.0;

        for _ in 0..steps {
            self.rk4_step(state, t, &h_func);
            t += self.dt;
        }
    }

    fn rk4_step<F>(&self, state: &mut TQPState, t: f64, h_func: &F)
    where
        F: Fn(f64) -> Array2<Complex64>,
    {
        // RK4:
        // k1 = f(t, y)
        // k2 = f(t + dt/2, y + dt/2 * k1)
        // k3 = f(t + dt/2, y + dt/2 * k2)
        // k4 = f(t + dt, y + dt * k3)
        // y_new = y + dt/6 * (k1 + 2k2 + 2k3 + k4)
        // f(t, psi) = -i H(t) psi

        let dt = self.dt;
        let psi = &state.state_vector;

        let k1 = self.f(t, psi, h_func);

        let psi_k1 = psi + &(&k1 * (dt * 0.5));
        let k2 = self.f(t + dt * 0.5, &psi_k1, h_func);

        let psi_k2 = psi + &(&k2 * (dt * 0.5));
        let k3 = self.f(t + dt * 0.5, &psi_k2, h_func);

        let psi_k3 = psi + &(&k3 * dt);
        let k4 = self.f(t + dt, &psi_k3, h_func);

        // Update state
        // state.state_vector = psi + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        let delta = (k1 + &k2 * 2.0 + &k3 * 2.0 + k4) * (dt / 6.0);
        state.state_vector = psi + &delta;

        // Normalize to correct numerical drift
        state.normalize();
    }

    fn f<F>(&self, t: f64, psi: &Array1<Complex64>, h_func: &F) -> Array1<Complex64>
    where
        F: Fn(f64) -> Array2<Complex64>,
    {
        let h = h_func(t);
        // Result = -i * H * psi
        // H is Array2, psi is Array1.
        // dot product.

        let h_psi = h.dot(psi);
        let neg_i = Complex64::new(0.0, -1.0);

        h_psi.mapv(|x| x * neg_i)
    }
}
