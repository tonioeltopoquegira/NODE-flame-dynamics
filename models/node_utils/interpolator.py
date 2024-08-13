import flax.linen as nn
import jax.numpy as jnp
import jax
import jax.random as random

class Interpolator1D(nn.Module):
    times: jnp.ndarray
    values: jnp.ndarray
    method: str

    def setup(self):
        pass 

    def __call__(self, t_evaluation, deg=None):
        if self.method == "linear":
            return self._linear_interpolate(t_evaluation)
        elif self.method == "cubic-poly":
            if deg is None:
                raise ValueError("Degree (deg) must be specified for cubic-polynomial interpolation.")
            return self._polynomial_interpolate(t_evaluation, deg)
        else:
            raise ValueError(f"Interpolation method '{self.method}' is not supported.")
    
    def _linear_interpolate(self, t_evaluation):
        idx = jnp.searchsorted(self.times, t_evaluation) - 1
        idx = jnp.clip(idx, 0, len(self.times) - 2)
        x0 = self.times[idx]
        x1 = self.times[idx + 1]
        y0 = self.values[idx]
        y1 = self.values[idx + 1]
        return y0 + (y1 - y0) * (t_evaluation - x0) / (x1 - x0)
    
    def _polynomial_interpolate(self, t_evaluation, deg):
        idx = jnp.searchsorted(self.times, t_evaluation) - 1
        idx = jnp.clip(idx, 1, len(self.times) - deg)

        def interpolate_single(t, idx):
            idxs = jnp.arange(idx - 1, idx + 2)  # Adjusted for cubic interpolation (degree 2)
            coefs = jnp.polyfit(self.times[idxs], self.values[idxs], deg=deg)
            return jnp.polyval(coefs, t)

        return jax.vmap(interpolate_single)(t_evaluation, idx)


import matplotlib.pyplot as plt

def test_interpolation():

    times = jnp.array([1.0, 3.0, 4.0, 5.0])
    values = jnp.array([3.0, 5.0, 6.0, 7.0])
    inter = Interpolator1D(times, values, method='linear')
    eval = jnp.array([3.0])
    out = inter.interpolate(eval, 3)
    print(out)

    
    # Data
    import h5py
    a = '050'
    hf = h5py.File('data/Kornilov_Haeringer_all.h5', 'r')
    values = jnp.array(hf.get('BB_A' + a+ '_U'))
    times = jnp.array(hf.get('BB_time'))

    values = values[::100]
    times = times[::100]
    values = values[:100]
    times = times[:100]

    # Randomly exclude some times and values
    key = random.PRNGKey(42)
    mask = random.bernoulli(key, p=0.8, shape=times.shape)  
    times_provided = times[mask]
    values_provided = values[mask]

    # Times to interpolate
    times_to_interpolate = times[~mask]
    values_to_interpolate = values[~mask]

    # Initialize interpolator
    inter = Interpolator1D(times_provided, values_provided, method = 'linear')
    inter_cubic = Interpolator1D(times_provided, values, method='cubic-poly')

    # Interpolate values at the excluded times
    interpolated_values_linear = inter(times_to_interpolate, deg = 1)
    interpolated_values_cubic = inter_cubic(times_to_interpolate, deg = 4)

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(times_provided, values_provided, '.', color='black', label='Points Provided', markersize= 5)
    plt.plot(times_to_interpolate, values_to_interpolate, '.', color='red', label='Real Values', markersize = 5)
    plt.plot(times_to_interpolate, interpolated_values_linear, '.', color='blue', label='Linear Interpolation',markersize = 5.5)
    plt.plot(times_to_interpolate, interpolated_values_cubic, '.', color='green', label='Cubic Polynomial Interpolation', markersize = 5.5)
    plt.plot(times, values, '-', color='red', alpha=0.5)  # Connect interpolated points

    # Add grid, title, and labels
    plt.grid(True)
    plt.title("Interpolation with Missing Data")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.show()
    plt.close()


if __name__ == "__main__":
    test_interpolation()