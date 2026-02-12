import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

def cov_func(times, x, T, omega, gamma):

    t = times[:, None]
    tprime = times[None, :]
    upper = jnp.minimum(t, tprime)
    upper = jnp.minimum(upper, T)
    fac = jnp.exp(-gamma*(t+tprime)) / jnp.square(omega) * jnp.square(x)
    
    term1 = jnp.cos(omega*(t-tprime)) / (4*gamma) * (jnp.exp(2*gamma*upper) - 1)

    def eval2(s):
        return jnp.exp(2*gamma*s) * (gamma*jnp.cos(2*omega*s) + omega*jnp.sin(2*omega*s))
    term2 = -jnp.cos(omega*(t+tprime)) / (4*(jnp.square(gamma) + jnp.square(omega))) * (eval2(upper) - eval2(0))

    def eval3(s):
        return jnp.exp(2*gamma*s) * (gamma*jnp.sin(2*omega*s) - omega*jnp.cos(2*omega*s))
    term3 = -jnp.sin(omega*(t+tprime)) / (4*(jnp.square(gamma) + jnp.square(omega))) * (eval3(upper) - eval3(0))

    return fac * (term1 + term2 + term3)

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


def cov_func_test(times, x, T, omega, gamma):
    t1 = times[:, None]
    t2 = times[None, :]

    tmin = jnp.minimum(t1, t2)
    tmin = jnp.minimum(tmin, T)

    gamma2 = jnp.square(gamma)
    omega2 = jnp.square(omega)

    factor = jnp.square(x) / (4 * gamma * omega2 * (gamma2 + omega2))

    term1 = jnp.exp(-gamma * (t1 + t2)) * (gamma * (gamma * jnp.cos(omega * (t1 + t2)) - omega * jnp.sin(omega * (t1 + t2))) - (gamma2 + omega2) * jnp.cos(omega * (t1 - t2)))
    term2 = jnp.exp(-gamma * (t1 + t2 - 2 * tmin)) * ((gamma2 + omega2) * jnp.cos(omega * (t1 - t2)) + gamma * (omega * jnp.sin(omega * (t1 + t2 - 2*tmin)) - gamma * jnp.cos(omega * (t1 + t2 - 2*tmin))))

    return factor * (term1 + term2)