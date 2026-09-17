import jax

# Applications choose precision; tests exercise the intended scientific regime.
jax.config.update("jax_enable_x64", True)
