import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

def cov_func(times, x, T, omega, gamma):
    raise NotImplementedError("Not implemented yet")

def model(times, data, data_err, omega_bounds, gamma_bounds, T_bounds):
    A = numpyro.sample('A', dist.Normal(0,1))
    B = numpyro.sample('B', dist.Normal(0,1))

    x = numpyro.sample('x', dist.Exponential(1))
    T = numpyro.sample('T', dist.Uniform(*T_bounds))

    omega = numpyro.sample('omega', dist.Uniform(*omega_bounds))
    gamma = numpyro.sample('gamma', dist.Uniform(*gamma_bounds))

    h_mu = jnp.exp(-gamma*times) * (A * jnp.cos(omega*times) + B * jnp.sin(omega*times))
    h_cov = cov_func(times, x, T, omega, gamma)

    numpyro.sample('data', dist.MultivariateNormal(h_mu, h_cov + jnp.diag(data_err**2)), obs=data)