import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ActorVectorField, Value, LogParam
from utils.networks import *


class GSflow(flax.struct.PyTreeNode):
    """GS-Flow agent."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def vae_loss(self, batch, grad_params, rng):
        batch_size, action_dim = batch['actions'].shape
        
        if self.config['encoder'] is not None:
            observations = self.network.select('actor_guassian_flow_encoder')(batch['observations'])
            observations = jax.lax.stop_gradient(observations) 
            observations_critic = self.network.select('critic')(batch['observations'], return_encoder=True)

        else:
            observations = batch['observations']
            observations_critic = batch['observations']
        
        def transfer_dimsions(x, rng):
            """Transfer dimensions to match the expected shape."""
            rng, noise_rng = jax.random.split(rng, 2)
            num_candidate = self.config['num_candidate']
            noises = jax.random.normal(noise_rng, (batch_size, num_candidate, action_dim))
            noises_expanded = noises.reshape(batch_size * num_candidate, -1)
            observations_expanded = jnp.expand_dims(x, axis=0) 
            observations_expanded = jnp.tile(observations_expanded, (num_candidate, 1, 1)) 
            observations_expanded = jnp.transpose(observations_expanded, (1, 0, 2))
            observations_expanded = observations_expanded.reshape(batch_size * num_candidate, -1)
            return observations_expanded, noises_expanded, rng
        
        observations_expanded, noises_expanded, rng = transfer_dimsions(observations, rng)
        observations_critic, noises_expanded, rng = transfer_dimsions(observations_critic, rng)
                      
        action_candidate, noises_best, q_noises_best = self.compute_flow_actions_noises(observations_expanded, observations_critic, noises=noises_expanded)
        # --VAE ---
        mean_z, log_std_z = self.network.select('noise_encoder')(observations, noises_best, is_encoded=True, params=grad_params)
        std_z = jnp.exp(log_std_z)
        rng, vae_rng = jax.random.split(rng, 2)
        z_noise = jax.random.normal(vae_rng, mean_z.shape)
        z = mean_z + std_z * z_noise  # Differentiable sampling 
        reconstructed_noise = self.network.select('noise_decoder')(observations, z, is_encoded=True, params=grad_params)
        recon_loss = jnp.mean(jnp.square(reconstructed_noise - noises_best).mean(axis=-1))
        kl_loss = -0.5 * jnp.mean(1 + 2 * log_std_z - jnp.square(mean_z) - jnp.square(std_z))
        vae_loss = self.config['recon_weight'] * recon_loss + self.config['kl_weight'] * kl_loss
        
        return vae_loss, {
            'vae_loss': vae_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
        }
      
    def critic_loss(self, batch, grad_params, rng, temperature=1.0):
        """Compute the FQL critic loss."""
        rng, sample_rng = jax.random.split(rng)
        next_actions = self.sample_actions(batch['next_observations'], seed=sample_rng, temperature=temperature)
        next_actions = jnp.clip(next_actions, -1, 1)

        next_qs = self.network.select('target_critic')(batch['next_observations'], actions=next_actions)
        if self.config['q_agg'] == 'min':
            next_q = next_qs.min(axis=0)
        else:
            next_q = next_qs.mean(axis=0)

        target_q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_q
        
        q = self.network.select('critic')(batch['observations'], actions=batch['actions'], params=grad_params)
        critic_loss = jnp.square(q - target_q).mean()

        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
        }

    def actor_loss(self, batch, grad_params, rng, alpha=300):
        """Compute the FQL actor loss."""
        batch_size, action_dim = batch['actions'].shape
        rng, x_rng, t_rng = jax.random.split(rng, 3)

        # BC flow loss.
        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = batch['actions']
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        pred = self.network.select('actor_bc_flow')(batch['observations'], x_t, t, params=grad_params)
        bc_flow_loss = jnp.mean((pred - vel) ** 2)
        
        # ! vae:
        rng, z_rng = jax.random.split(rng)
        latent_dim = self.config['latent_dim']
        z = jax.random.normal(
            z_rng, 
            (
                *batch['observations'].shape[: -len(self.config['ob_dims'])],
                latent_dim
            ),
        )        
        noises = self.network.select('noise_decoder')(batch['observations'], z)
                
        # 4. Sample noise from noise_policy; Train the distill model;
        target_flow_actions = self.compute_flow_actions(batch['observations'], noises=noises)
        dist = self.network.select('actor_guassian_flow')(batch['observations'], noises, params=grad_params)
        
        # ! vae
        rng, action_rng = jax.random.split(rng, 2)
        actions, log_probs = dist.sample_and_log_prob(seed=action_rng)    
        distill_loss = jnp.mean((dist.mode() - target_flow_actions) ** 2)
        actor_actions = jnp.clip(actions, -1, 1)
        qs = self.network.select('critic')(batch['observations'], actions=actor_actions)
        q = jnp.mean(qs, axis=0)
        sac_loss = (log_probs * self.network.select('alpha')() - q).mean()
        sac_alpha = self.network.select('alpha')(params=grad_params)
        entropy = -jax.lax.stop_gradient(log_probs).mean()
        alpha_loss = (sac_alpha * (entropy - self.config['target_entropy'])).mean()
        sac_a_loss = sac_loss + alpha_loss
        actor_loss = self.config['alpha'] * distill_loss + sac_a_loss
        actor_loss = alpha * distill_loss + sac_a_loss
    
        # Total loss.
        actor_loss = bc_flow_loss + actor_loss
        
        if self.config['tanh_squash']:
            std = dist._distribution.stddev()
        else:
            std = dist.stddev().mean()
        
        # Additional metrics for logging.
        actions = self.sample_actions(batch['observations'], seed=rng)
        mse = jnp.mean((actions - batch['actions']) ** 2)

        return actor_loss, {
            'actor_loss': actor_loss,
            'bc_flow_loss': bc_flow_loss,
            'distill_loss': distill_loss,
            'q': q.mean(),
            'mse': mse,
            'alpha': alpha,
            'entropy': -log_probs.mean(),
            'std': std.mean(),
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None, temperature=1.0, alpha=300):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        rng, actor_rng, critic_rng = jax.random.split(rng, 3)

        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng, temperature=temperature)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng, alpha=alpha)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        vae_loss, vae_info = self.vae_loss(batch, grad_params, actor_rng)
        for k, v in vae_info.items():
            info[f'vae/{k}'] = v

        loss = critic_loss + actor_loss + 1 * vae_loss
        
        return loss, info
    

    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @jax.jit
    def update(self, batch, temperature=1, alpha=300):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng, temperature=temperature, alpha=alpha)
        
        def critic_loss(grad_params):
            return self.critic_loss(batch, grad_params, rng=rng, temperature=temperature)
        
        def vae_loss(grad_params):
            return self.vae_loss(batch, grad_params, rng=rng)
        
        for i in range(5):
            # critic loss
            new_rng, rng = jax.random.split(new_rng)
            new_network, info = self.network.apply_loss_fn(loss_fn=critic_loss)
            self.target_update(new_network, 'critic')
            self.replace(network=new_network, rng=new_rng)

        for i in range(5):
            # vae loss
            new_rng, rng = jax.random.split(new_rng)
            new_network, info = self.network.apply_loss_fn(loss_fn=vae_loss)
            self.replace(network=new_network, rng=new_rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'critic')

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(
        self,
        observations,
        seed=None,
        temperature=1.0,
    ):
        """Sample actions from the one-step policy."""
        # ! if vae:        
        if self.config['encoder'] is not None:
            obs_emb = self.network.select('actor_guassian_flow_encoder')(observations)
        else:
            obs_emb = observations

        action_seed, z_rng = jax.random.split(seed)
        latent_dim = self.config['latent_dim']
        z = jax.random.normal(
            z_rng, 
            (
                *observations.shape[: -len(self.config['ob_dims'])],
                latent_dim
            ),
        )        
        reconstructed_noise = self.network.select('noise_decoder')(obs_emb, z, is_encoded=True)
        
        dist = self.network.select('actor_guassian_flow')(obs_emb, reconstructed_noise, is_encoded=True, temperature=temperature)
        actions = dist.sample(seed=seed)
        actions = jnp.clip(actions, -1, 1)
        
        return actions

    @jax.jit
    def sample_actions_noises(
        self,
        observations,
        seed=None,
        temperature=1.0,
    ):
        """Sample actions from the one-step policy."""
        # ! if vae:        
        if self.config['encoder'] is not None:
            obs_emb = self.network.select('actor_guassian_flow_encoder')(observations)
        else:
            obs_emb = observations

        action_seed, z_rng = jax.random.split(seed)
        latent_dim = self.config['latent_dim']
        z = jax.random.normal(
            z_rng, 
            (
                *observations.shape[: -len(self.config['ob_dims'])],
                latent_dim
            ),
        )        
        reconstructed_noise = self.network.select('noise_decoder')(obs_emb, z, is_encoded=True)
        dist = self.network.select('actor_guassian_flow')(obs_emb, reconstructed_noise, is_encoded=True, temperature=temperature)
        actions = dist.sample(seed=seed)
        actions = jnp.clip(actions, -1, 1)
        
        actions_fm = self.compute_flow_actions(observations, reconstructed_noise)
        
        return actions, reconstructed_noise, actions_fm

    @jax.jit
    def compute_flow_actions(
        self,
        observations,
        noises,
    ):
        """Compute actions from the BC flow model using the Euler method."""
        if self.config['encoder'] is not None:
            observations = self.network.select('actor_bc_flow_encoder')(observations)    
        
        actions = noises
        # Euler method.
        for i in range(self.config['flow_steps']):
            t = jnp.full((*observations.shape[:-1], 1), i / self.config['flow_steps'])
            vels = self.network.select('actor_bc_flow')(observations, actions, t, is_encoded=True)
            actions = actions + vels / self.config['flow_steps']
        actions = jnp.clip(actions, -1, 1)
        return actions
    

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
    ):
        """Create a new agent.

        Args:
            seed: Random seed.
            ex_observations: Example batch of observations.
            ex_actions: Example batch of actions.
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_times = ex_actions[..., :1]
        ob_dims = ex_observations.shape[1:]
        action_dim = ex_actions.shape[-1]
        
        if config['target_entropy'] is None:
            config['target_entropy'] = -config['target_entropy_multiplier'] * action_dim

        # Define encoders.
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['critic'] = encoder_module()
            encoders['actor_bc_flow'] = encoder_module()
            encoders['actor_guassian_flow'] = encoder_module()


        # Define networks.
        critic_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=2,
            encoder=encoders.get('critic'),
        )
        actor_bc_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor_bc_flow'),
        )
        
        actor_guassian_flow_def = Actor_Noise(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor_guassian_flow'),
            tanh_squash=config['tanh_squash'],
            state_dependent_std=config['state_dependent_std'],
            const_std=False,
            final_fc_init_scale=config['actor_fc_scale'],
            log_std_min=-2,
            log_std_max=0,
        )
        
        noise_encoder_def = Encoder(
            latent_dim=config['latent_dim'],
            encoder=encoders.get('actor_guassian_flow'),
        )
        noise_decoder_def = Decoder(
            action_dim=action_dim,
            latent_dim=config['latent_dim'],
            encoder=encoders.get('actor_guassian_flow'),
        )
        
        alpha_def = LogParam()
        ex_vae_latent = jax.random.normal(rng, (ex_actions.shape[0], config['latent_dim']))
        
        network_info = dict(
            critic=(critic_def, (ex_observations, ex_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, ex_actions)),
            actor_bc_flow=(actor_bc_flow_def, (ex_observations, ex_actions, ex_times)),
            actor_guassian_flow=(actor_guassian_flow_def, (ex_observations, ex_actions)),
            alpha=(alpha_def, ()),
            noise_encoder=(noise_encoder_def, (ex_observations, ex_actions)),
            noise_decoder=(noise_decoder_def, (ex_observations, ex_vae_latent)),
        )
        if encoders.get('actor_bc_flow') is not None:
            network_info['actor_bc_flow_encoder'] = (encoders.get('actor_bc_flow'), (ex_observations,))
            network_info['actor_guassian_flow_encoder'] = (encoders.get('actor_guassian_flow'), (ex_observations,))
                         
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params['modules_target_critic'] = params['modules_critic']

        config['ob_dims'] = ob_dims
        config['action_dim'] = action_dim
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='gsflow',  # Agent name.
            ob_dims=ml_collections.config_dict.placeholder(list),  # Observation dimensions (will be set automatically).
            action_dim=ml_collections.config_dict.placeholder(int),  # Action dimension (will be set automatically).
            lr=3e-4,  # Learning rate.
            batch_size=256,  # Batch size.
            actor_hidden_dims=(512, 512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512, 512),  # Value network hidden dimensions.
            layer_norm=True,  # Whether to use layer normalization.
            actor_layer_norm=False,  # Whether to use layer normalization for the actor.
            discount=0.99,  # Discount factor.
            tau=0.005,  # Target network update rate.
            target_entropy=ml_collections.config_dict.placeholder(float),  # Target entropy (None for automatic tuning).
            target_entropy_multiplier=0.5,  # Multiplier to dim(A) for target entropy.
            q_agg='mean',  # Aggregation method for target Q values.
            alpha=10.0,  # BC coefficient (need to be tuned for each environment).
            flow_steps=10,  # Number of flow steps.
            normalize_q_loss=False,  # Whether to normalize the Q loss.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
            actor_fc_scale=0.01,  # Final layer initialization scale for actor.
            tanh_squash=True,  # Whether to squash actions with tanh.
            actor_noise=0.2,  # Actor noise scale.
            actor_noise_clip=0.5,  # Actor noise clipping threshold.
            state_dependent_std=True,
            num_candidate = 10,
            recon_weight = 1,
            kl_weight = 0.01,
            latent_dim = 3,
        )
    )
    return config
