import flax.linen as nn
import jax.numpy as jnp
import matplotlib.pyplot as plt

# NOTE!
# (1) When to stop adaptive integration?

class Integrator(nn.Module):
    strategy: str
    method: str
    interp: nn.Module 

    def setup(self):
        pass

    @nn.compact
    def __call__(self, fun, t_evaluation, y0):
        if self.strategy == 'fixed-grid':
            return self.fixed_grid(fun, t_evaluation, y0)
        elif self.strategy == 'adaptive':
            return self.adaptive(fun, t_evaluation, y0)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def fixed_grid(self, fun, t_evaluation, y0):
        """
        Perform numerical integration using fixed grid.
        """
        delta_ts = jnp.diff(t_evaluation)
        y = jnp.zeros_like(t_evaluation)
        y = y.at[0].set(y0)
        
        for en, (dt, t) in enumerate(zip(delta_ts, t_evaluation[:-1])):  # Exclude the last time step for iteration
            dy = self.step(fun, dt, t[:, en, :])  # Replace with the actual step function
            dy = jnp.squeeze(dy, axis = 0)
            dy = jnp.squeeze(dy, axis = 0)
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

    def step(self, fun, dt, t):
        # My model doesn't really take x in as a value... It could take whichever number of interpolator evaluations to compute my result

        if self.method == 'euler':
            u = self.interp(t) # Maybe move the interpolator inside the model ? Make the model depend on t only from the outside
            dy = fun(u) * dt
            return dy

        elif self.method == 'rk4':
            # For RK4 method
            u = self.interp(t) 
            k1 = fun(u)
            k2 = fun(self.interp(t + 0.5 * dt))  
            k3 = fun(self.interp(t + 0.5 * dt))  
            k4 = fun(self.interp(t + dt))
            dy = (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            return dy


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