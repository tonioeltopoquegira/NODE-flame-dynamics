import flax.linen as nn
import jax.numpy as jnp
import matplotlib.pyplot as plt
import jax

# NOTE!
# (1) When to stop adaptive integration?

class Integrator(nn.Module):
    strategy: str
    method: str
    interp: nn.Module 

    def setup(self):
        pass

    def __call__(self, fun, t_evaluation, y0, deriv_eval):
        if self.strategy == 'fixed-grid':
            return self.fixed_grid(fun, t_evaluation, y0, deriv_eval)
        elif self.strategy == 'adaptive':
            return self.adaptive(fun, t_evaluation, y0, deriv_eval)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def fixed_grid(self, fun, t_evaluation, y0, deriv_eval):
        """
        Perform numerical integration using fixed grid.
        """
        delta_ts = jnp.diff(t_evaluation)
        y = jnp.zeros_like(t_evaluation)
        y = y.at[0].set(y0)
        
        for en, (dt, t) in enumerate(zip(delta_ts, t_evaluation[:-1])):  # Exclude the last time step for iteration
            dy = self.step(fun, dt, t, deriv_eval[en, :])
            y = y.at[en + 1].set(y[en] + dy)
        
        return (t_evaluation, y)

    def adaptive(self, fun, t_evaluation, y0, rtol=1e-6, atol=1e-8):
        """
        Perform numerical integration using adaptive time stepping.
        """
        y = y0
        result = []
        t = t_evaluation[0]
        result.append((t, y))
        
        # Initial step size
        dt = t_evaluation[1] - t_evaluation[0]
        
        i = 1
        while t < t_evaluation[-1]:

            # Step 
            y1 = y + self.step(fun, dt, t)
            
            # TODO: which criterias? (1) Magnitude of gradient (2) Error with two steps of half size
            
            dt_half = dt / 2
            y_half_1 = y + self.step(fun, dt_half, t)
            y_half_2 = y_half_1 + self.step(fun, dt_half, t + dt_half)
            error = jnp.linalg.norm(y1 - y_half_2)  # Error between full step and two half steps
            
            if error < atol + rtol * jnp.linalg.norm(y1):
                # Accept the step
                t = t + dt
                y = y1
                result.append((t, y))
                
                # Increase the step size for the next iteration
                dt *= 1.5
            else:
                # Decrease the step size
                dt *= 0.5
            
            i += 1
        
        print(f"Evaluated at {i} points against the {len(t_evaluation)} of fixed-grid")
        
        return jnp.array([r[0] for r in result]), jnp.array([r[1] for r in result])

    def step(self, fun, dt, t, deriv_eval):
        # My model doesn't really take x in as a value... It could take whichever number of interpolator evaluations to compute my result

        if self.method == 'euler':
            # Create jnp.array of u's observations from t_eval using interpolator ---> one evaluation of derivative at time t
            # TODO
            xs = self.interp_expand(t, deriv_eval)
            dy = fun(xs) * dt 
            dy = jnp.squeeze(dy, 0)
            dy = jnp.squeeze(dy, 0)
            return dy * 100

        elif self.method == 'rk4':

            # For RK4 method
            # Create jnp.array of u's observations from t_eval using interpolator ---> one evaluation of derivative at time t0
            # TODO
            x0s = self.interp_expand(t, deriv_eval)
            # After we want the u's obbservations necessary to evaluate at t1 = t+0.5*dt and at t2 = t+dt ---> other 2 u jnp.array's
            x1s = self.interp_expand(t+0.5*dt, deriv_eval)
            x2s = self.interp_expand(t+dt, deriv_eval)

            k1 = fun(x0s)
            k2 = fun(x1s)  
            k3 = fun(x1s)  
            k4 = fun(x2s)
            dy = (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            return dy
        
    def interp_expand(self, t, deriv_eval):
        # Compute the differences between t and deriv_eval in a JAX-native way
        t_shifted = t - deriv_eval
        # Apply self.interp to each element in t_shifted using vmap, which vectorizes the function
        us = jax.vmap(self.interp)(t_shifted)
        xs = jnp.stack([deriv_eval, us])
        xs = jnp.expand_dims(xs, axis = 0)
        return xs



def test_integrator():
    # Define the function u(t)
    def u_function(t):
        return jnp.sin(t)  # Example function

    # Generate discrete time points and corresponding u values
    t_values = jnp.linspace(0, 5, 20)
    u_values = u_function(t_values)

    from interpolator import Interpolator1D

    interp = Interpolator1D(t_values, u_values, method='linear')

    integ = Integrator(strategy='fixed-grid', method='rk4', interpolator=interp)

    def example_fun(u):
        return -0.5 * u

    # Initial condition
    y0 = jnp.array(1.0)
    # Time range for integration
    t_evaluation = jnp.linspace(0, 5, 50)

    t_evaluated, result = integ.integrate(fun=example_fun, t_evaluation=t_evaluation, y0 = y0)

    def analytical_solution(t, y0, lambda_val):
        return 0.5 * jnp.cos(t) + 0.5

    # Analytical solution parameters
    lambda_val = 0.5

    # Compute analytical solution
    analytical_results = analytical_solution(t_evaluation, y0, lambda_val)

    fun_values = example_fun(u_values)

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(t_values, fun_values, '.', label='Discrete u(t) Values', color='blue')
    plt.plot(t_evaluated, result, label='Numerical Solution', color='red', linestyle='--')
    plt.plot(t_evaluation, analytical_results, label='Analytical Solution', color='black')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title(f'Numerical Integration with Euler Method')
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    test_integrator()