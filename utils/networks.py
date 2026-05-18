from typing import Any, Optional, Sequence

import distrax
import flax.linen as nn
import jax.numpy as jnp
import jax

from typing import Sequence, Optional


def default_init(scale=1.0):
    """Default kernel initializer."""
    return nn.initializers.variance_scaling(scale, 'fan_avg', 'uniform')


def ensemblize(cls, num_qs, in_axes=None, out_axes=0, **kwargs):
    """Ensemblize a module."""
    return nn.vmap(
        cls,
        variable_axes={'params': 0, 'intermediates': 0},
        split_rngs={'params': True},
        in_axes=in_axes,
        out_axes=out_axes,
        axis_size=num_qs,
        **kwargs,
    )


class Identity(nn.Module):
    """Identity layer."""

    def __call__(self, x):
        return x


class MLP(nn.Module):
    """Multi-layer perceptron.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        activations: Activation function.
        activate_final: Whether to apply activation to the final layer.
        kernel_init: Kernel initializer.
        layer_norm: Whether to apply layer normalization.
    """

    hidden_dims: Sequence[int]
    activations: Any = nn.gelu
    activate_final: bool = False
    kernel_init: Any = default_init()
    layer_norm: bool = False

    @nn.compact
    def __call__(self, x):
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=self.kernel_init)(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = self.activations(x)
                if self.layer_norm:
                    x = nn.LayerNorm()(x)
            if i == len(self.hidden_dims) - 2:
                self.sow('intermediates', 'feature', x)
        return x


class LogParam(nn.Module):
    """Scalar parameter module with log scale."""

    init_value: float = 1.0

    @nn.compact
    def __call__(self):
        log_value = self.param('log_value', init_fn=lambda key: jnp.full((), jnp.log(self.init_value)))
        return jnp.exp(log_value)


class TransformedWithMode(distrax.Transformed):
    """Transformed distribution with mode calculation."""

    def mode(self):
        return self.bijector.forward(self.distribution.mode())

class Actor(nn.Module):
    """Gaussian actor network.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        action_dim: Action dimension.
        layer_norm: Whether to apply layer normalization.
        log_std_min: Minimum value of log standard deviation.
        log_std_max: Maximum value of log standard deviation.
        tanh_squash: Whether to squash the action with tanh.
        state_dependent_std: Whether to use state-dependent standard deviation.
        const_std: Whether to use constant standard deviation.
        final_fc_init_scale: Initial scale of the final fully-connected layer.
        encoder: Optional encoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    action_dim: int
    layer_norm: bool = False
    log_std_min: Optional[float] = -5
    log_std_max: Optional[float] = 2
    tanh_squash: bool = False
    state_dependent_std: bool = False
    const_std: bool = True
    final_fc_init_scale: float = 1e-2
    encoder: nn.Module = None

    def setup(self):
        self.actor_net = MLP(self.hidden_dims, activate_final=True, layer_norm=self.layer_norm)
        self.mean_net = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))
        if self.state_dependent_std:
            self.log_std_net = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))
        else:
            if not self.const_std:
                self.log_stds = self.param('log_stds', nn.initializers.zeros, (self.action_dim,))

    def __call__(
        self,
        observations,
        temperature=1.0,
    ):
        """Return action distributions.

        Args:
            observations: Observations.
            temperature: Scaling factor for the standard deviation.
        """
        if self.encoder is not None:
            inputs = self.encoder(observations)
        else:
            inputs = observations
        outputs = self.actor_net(inputs)

        means = self.mean_net(outputs)
        if self.state_dependent_std:
            log_stds = self.log_std_net(outputs)
        else:
            if self.const_std:
                log_stds = jnp.zeros_like(means)
            else:
                log_stds = self.log_stds

        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        distribution = distrax.MultivariateNormalDiag(loc=means, scale_diag=jnp.exp(log_stds) * temperature)
        if self.tanh_squash:
            distribution = TransformedWithMode(distribution, distrax.Block(distrax.Tanh(), ndims=1))

        return distribution


class Actor_Noise(nn.Module):
    """Gaussian actor network.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        action_dim: Action dimension.
        layer_norm: Whether to apply layer normalization.
        log_std_min: Minimum value of log standard deviation.
        log_std_max: Maximum value of log standard deviation.
        tanh_squash: Whether to squash the action with tanh.
        state_dependent_std: Whether to use state-dependent standard deviation.
        const_std: Whether to use constant standard deviation.
        final_fc_init_scale: Initial scale of the final fully-connected layer.
        encoder: Optional encoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    action_dim: int
    layer_norm: bool = False
    log_std_min: Optional[float] = -5
    log_std_max: Optional[float] = 2
    tanh_squash: bool = False
    state_dependent_std: bool = False
    const_std: bool = True
    encoder: nn.Module = None
    output_mean: bool = False
    final_fc_init_scale: float = 1e-2

    def setup(self):
        self.actor_net = MLP(self.hidden_dims, activate_final=True, layer_norm=self.layer_norm)
        self.mean_net = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))
        if self.state_dependent_std:
            self.log_std_net = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))
        else:
            if not self.const_std:
                self.log_stds = self.param('log_stds', nn.initializers.zeros, (self.action_dim,))

    def __call__(
        self,
        observations,
        actions,
        temperature=1.0,
        is_encoded=False
    ):
        """Return action distributions.

        Args:
            observations: Observations.
            temperature: Scaling factor for the standard deviation.
        """
        if not is_encoded and self.encoder is not None:
            inputs = self.encoder(observations)
        else:
            inputs = observations
        inputs = jnp.concatenate([inputs, actions], axis=-1)
        outputs = self.actor_net(inputs)

        means = self.mean_net(outputs)
        if self.state_dependent_std:
            log_stds = self.log_std_net(outputs)
        else:
            if self.const_std:
                log_stds = jnp.zeros_like(means)
            else:
                log_stds = self.log_stds


        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)
        
        scale_diag = jnp.exp(log_stds) * temperature
        distribution = distrax.MultivariateNormalDiag(loc=means, scale_diag=scale_diag)
        if self.tanh_squash:
            distribution = TransformedWithMode(distribution, distrax.Block(distrax.Tanh(), ndims=1))
        
        return distribution


class Value(nn.Module):
    """Value/critic network.

    This module can be used for both value V(s, g) and critic Q(s, a, g) functions.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        layer_norm: Whether to apply layer normalization.
        num_ensembles: Number of ensemble components.
        encoder: Optional encoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    layer_norm: bool = True
    num_ensembles: int = 2
    encoder: nn.Module = None

    def setup(self):
        mlp_class = MLP
        if self.num_ensembles > 1:
            mlp_class = ensemblize(mlp_class, self.num_ensembles)
        value_net = mlp_class((*self.hidden_dims, 1), activate_final=False, layer_norm=self.layer_norm)

        self.value_net = value_net

    def __call__(self, observations, actions=None, is_encoded=False, return_encoder=False):
        """Return values or critic values.

        Args:
            observations: Observations.
            actions: Actions (optional).
        """
        if not is_encoded and self.encoder is not None:
            inputs = [self.encoder(observations)]
            if return_encoder:
                return inputs[0]
        else:
            inputs = [observations]
        if actions is not None:
            inputs.append(actions)
        inputs = jnp.concatenate(inputs, axis=-1)

        v = self.value_net(inputs).squeeze(-1)

        return v
    
    
class ActorVectorField(nn.Module):
    """Actor vector field network for flow matching.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        action_dim: Action dimension.
        layer_norm: Whether to apply layer normalization.
        encoder: Optional encoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    action_dim: int
    layer_norm: bool = False
    encoder: nn.Module = None

    def setup(self) -> None:
        self.mlp = MLP((*self.hidden_dims, self.action_dim), activate_final=False, layer_norm=self.layer_norm)

    @nn.compact
    def __call__(self, observations, actions, times=None, is_encoded=False, temperature=0.0):
        """Return the vectors at the given states, actions, and times (optional).

        Args:
            observations: Observations.
            actions: Actions.
            times: Times (optional).
            is_encoded: Whether the observations are already encoded.
        """
        if not is_encoded and self.encoder is not None:
            observations = self.encoder(observations)
        if times is None:
            inputs = jnp.concatenate([observations, actions], axis=-1)
        else:
            inputs = jnp.concatenate([observations, actions, times], axis=-1)

        v = self.mlp(inputs)

        return v
    


class GaussianActor(nn.Module):
    """Gaussian actor network.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        action_dim: Action dimension.
        min_std: Minimum standard deviation.
        max_std: Maximum standard deviation.
        layer_norm: Whether to apply layer normalization.
        encoder: Optional encoder module to encode the inputs.
    """
    hidden_dims: Sequence[int]
    action_dim: int
    min_std: float = 1e-4
    max_std: float = 5e-2
    layer_norm: bool = False
    encoder: Optional[nn.Module] = None

    def setup(self) -> None:
        self.mlp = nn.Sequential([
            nn.Dense(dim) if not self.layer_norm else nn.Sequential([nn.Dense(dim), nn.LayerNorm()])
            for dim in self.hidden_dims
        ])
        self.mean_layer = nn.Dense(self.action_dim)
        self.log_std_layer = nn.Dense(self.action_dim)

    def __call__(self, observations, actions, temperature=1.0, is_encoded=False, outtype='mean_std', rng=None):
        if not is_encoded and self.encoder is not None:
            observations = self.encoder(observations)

        inputs = jnp.concatenate([observations, actions], axis=-1)
        
        hidden = self.mlp(inputs)
        mean = self.mean_layer(hidden)
        # tanh
        mean = jnp.tanh(mean)
        std = self.log_std_layer(hidden)
        std = jnp.tanh(std)
        std = (std + 1) / 2 * (self.max_std - self.min_std) + self.min_std
        std = std * temperature
        
        dist = distrax.MultivariateNormalDiag(mean, std)

        
        if outtype == 'sample':
            assert rng is not None
            return dist.sample(seed=rng), dist
        elif outtype == 'mean_std':
            return mean, std
        elif outtype == 'log_prob':
            assert actions is not None
            return dist.log_prob(actions)

        return mean, std

    def log_prob(self, observations, actions, is_encoded=False):
        mean, std = self(observations, is_encoded=is_encoded)
        dist = distrax.MultivariateNormalDiag(loc=mean, scale_diag=std)
        return dist.log_prob(actions)

    def sample(self, observations, rng, is_encoded=False):
        mean, std = self(observations, is_encoded=is_encoded)
        dist = distrax.MultivariateNormalDiag(loc=mean, scale_diag=std)
        return dist.sample(seed=rng), dist


class NoiseSelect(nn.Module):
    """
    select noise to denoise
    """
    action_dim: int
    hidden_dims: Sequence[int]
    final_fc_init_scale: float = 1e-2
    layer_norm: bool = False

    def setup(self):
        self.dim_net = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))
        self.noise_net = MLP(self.hidden_dims, activate_final=True, layer_norm=self.layer_norm)


    def __call__(
        self,
        observations,
        noises,
    ):
        """
        now dense net to test
        """
        # flatten the [batch, candidate, action_dim] to [batch, candidate * action_dim]
        if noises.ndim == 3:
            inputs = jnp.reshape(noises, (noises.shape[0], -1))
        elif noises.ndim == 2:
            inputs = jnp.reshape(noises, -1)
        inputs = jnp.concatenate([observations, inputs], axis=-1)
        outputs = self.noise_net(inputs)
        outputs = self.dim_net(outputs)

        return outputs
    
    
    
class Encoder(nn.Module):
    latent_dim: int = 3
    encoder: Optional[nn.Module] = None
    
    @nn.compact
    def __call__(self, observations, noise, is_encoded=False):
        if not is_encoded and self.encoder is not None:
            observations = self.encoder(observations)
        inputs = jnp.concatenate([observations, noise], axis=-1)
        for i in range(3):
            x = nn.Dense(512)(inputs)
            x = nn.relu(x)
            
        mean = nn.Dense(self.latent_dim, name='mean_head')(x)
        log_std = nn.Dense(self.latent_dim, name='log_std_head')(x)
        return mean, log_std
    

class Decoder(nn.Module):
    action_dim: int
    latent_dim: int = 3
    encoder: Optional[nn.Module] = None
    
    @nn.compact
    def __call__(self, observations, z, is_encoded=False):
        if not is_encoded and self.encoder is not None:
            observations = self.encoder(observations)
        inputs = jnp.concatenate([observations, z], axis=-1)
        for i in range(3):
            x = nn.Dense(512)(inputs)
            x = nn.relu(x)
        reconstructed_noise = nn.Dense(self.action_dim)(x)
        
        return reconstructed_noise
    
