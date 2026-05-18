from collections import defaultdict

import os
import jax
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from datetime import datetime
import time
from scipy.stats import gaussian_kde

font_properties = {
    'family': 'Liberation Serif',
    'size': 28
}


def supply_rng(f, rng=jax.random.PRNGKey(0)):
    """Helper function to split the random number generator key before each call to the function."""

    def wrapped(*args, **kwargs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        return f(*args, seed=key, **kwargs)

    return wrapped


def flatten(d, parent_key='', sep='.'):
    """Flatten a dictionary."""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, 'items'):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def add_to(dict_of_lists, single_dict):
    """Append values to the corresponding lists in the dictionary."""
    for k, v in single_dict.items():
        dict_of_lists[k].append(v)


def evaluate(
    agent,
    env,
    config=None,
    num_eval_episodes=50,
    num_video_episodes=0,
    video_frame_skip=3,
    eval_temperature=0,
    monte_carlo=0,
    viz_GMM=False,
    save_dir=None,
    epoch=None,
):
    """Evaluate the agent in the environment.

    Args:
        agent: Agent.
        env: Environment.
        config: Configuration dictionary.
        num_eval_episodes: Number of episodes to evaluate the agent.
        num_video_episodes: Number of episodes to render. These episodes are not included in the statistics.
        video_frame_skip: Number of frames to skip between renders.
        eval_temperature: Action sampling temperature.

    Returns:
        A tuple containing the statistics, trajectories, and rendered videos.
    """
    if viz_GMM:
        actor_fn = supply_rng(agent.sample_actions_noises, rng=jax.random.PRNGKey(np.random.randint(0, 2**32)))
        actor_fn_step_time = supply_rng(agent.sample_actions, rng=jax.random.PRNGKey(np.random.randint(0, 2**32)))
    else:
        actor_fn = supply_rng(agent.sample_actions, rng=jax.random.PRNGKey(np.random.randint(0, 2**32)))
    if monte_carlo:
        actor_generate_fn = supply_rng(agent.generate_actions, rng=jax.random.PRNGKey(np.random.randint(0, 2**32)))
        critic_fn = agent.calculate_q_value
    trajs = []
    stats = defaultdict(list)

    renders = []
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")  # Format: YYYYMMDD_HHMMSS

    actions_list = []
    noises_list = []
    actions_fm_list = []
    
    
    # [viz] viz for Noise and Action
    if viz_GMM:
        observation, info = env.reset()
        for i in trange(50):
            action, noises, actioin_fm, q = actor_fn(observations=observation, temperature=eval_temperature)
            noises_list.append(noises)
            actions_list.append(action)
            
        noises_array = np.array(noises_list)
        ax = env.vizualize()
        # --- 开始：绘制覆盖散点的阴影区域 (KDE方法) ---
        # 检查是否有足够的数据点来进行密度估计（至少需要2个点）
        # if noises_array.shape[0] > 1:
        #     # 准备KDE所需的数据格式 (2, N)，即2行、N列
        #     noise_data_for_kde = np.vstack([noises_array[:, 0], noises_array[:, 1]])

        #     # 使用高斯核密度估计来估算数据点的概率密度
        #     kde = gaussian_kde(noise_data_for_kde)

        #     # 在绘图区域 [-1.5, 1.5] 之间创建一个网格，用于评估每个点的密度
        #     x_grid, y_grid = np.mgrid[-1.5:1.5:100j, -1.5:1.5:100j]
        #     positions = np.vstack([x_grid.ravel(), y_grid.ravel()])
        #     z_grid = np.reshape(kde(positions).T, x_grid.shape)

        #     # 使用 contourf 绘制填充等高线图作为阴影
        #     # levels 参数是关键，它决定了阴影的范围。
        #     # [0.05 * z_grid.max(), z_grid.max()] 表示填充密度大于最大密度5%的所有区域。
        #     # 你可以调整 0.05 这个值来控制阴影区域的大小。
        #     # zorder 控制绘图层级，确保阴影在背景之上、散点之下。
        #     ax.contourf(x_grid, y_grid, z_grid, levels=[0.1 * z_grid.max(), z_grid.max()],
        #                 cmap='Reds',  # 使用红色系的颜色映射
        #                 alpha=0.3,    # 设置透明度
        #                 zorder=2)     # 设置绘图层级
        # # --- 结束：绘制阴影区域 ---

        # 绘制原始的 noise 散点，设置 zorder=3 确保它在最顶层
        plt.scatter(noises_array[:, 0], noises_array[:, 1], s=10, color='red', alpha=0.5, label='Noises', zorder=3)
        
        plt.xlim(-1.5,1.5)
        plt.ylim(-1.5,1.5)
        plt.tight_layout() 
        plt.savefig(f'{save_dir}/viz-noise-region-{epoch}.pdf', dpi=300)
        plt.close()
        
        # plot action distribution
        action_array = np.array(actions_list)
        ax = env.vizualize()
        plt.scatter(action_array[:, 0], action_array[:, 1], marker='*', s=300, color='yellow', alpha=1, edgecolor='black', label='Actions')
        plt.tight_layout() 
        plt.savefig(f'{save_dir}/viz-action-{epoch}.pdf', dpi=300)
        plt.close()
    
    
    actions_list = []
    noises_list = []
    actions_fm_list = []
    
    infer_time_list = []
    
    for i in trange(num_eval_episodes + num_video_episodes):
        traj = defaultdict(list)
        should_render = i >= num_eval_episodes

        observation, info = env.reset()
        done = False
        step = 0
        render = []
        
        # using distilled model    
        if viz_GMM:
            action, noises, actioin_fm, q = actor_fn(observations=observation, temperature=eval_temperature)
        else:
            action = actor_fn(observations=observation, temperature=eval_temperature)
            
            
        action = np.array(action)
        action = np.clip(action, -1, 1)
        
        # for monte_carlo, the settings
        current_episode_actions = []
        n_bins = 100
        bin_edges = np.linspace(-1, 1, n_bins + 1)
        action_dim = action.shape[-1]

        while not done:
            if viz_GMM:
                action, noises, action_fm, q = actor_fn(observations=observation, temperature=eval_temperature)
            else:
                start_time = time.time()
                action = actor_fn(observations=observation, temperature=eval_temperature)
                end_time = time.time()
                infer_time_list.append((end_time - start_time)*1000)
                # print(f'Actor inference time per step: {(end_time - start_time)*1000:.4f} ms')

            action_interact = np.array(action)
            action_interact = np.clip(action_interact, -1, 1)

            if monte_carlo:
                def mc_sample_actions(sampling_fn, observation, monte_carlo, eval_temperature):
                    """Sample actions using the given sampling function."""
                    step_actions = []
                    for _ in range(monte_carlo):
                        action = sampling_fn(observations=observation, temperature=eval_temperature)
                        action = np.clip(np.array(action), -1, 1)
                        step_actions.append(action)
                    return np.stack(step_actions)
                
                def plot_action_distribution(ax, actions, dim, title_suffix):
                    """Plot action distribution for a single dimension."""
                    counts, bins, _ = ax.hist(
                        actions[:, dim],
                        bins=bin_edges,
                        color='skyblue',
                        edgecolor='black',
                        alpha=0.7
                    )
                    ax.set_title(f'Action Dim {dim} {title_suffix}')
                    ax.set_xlabel(f'Action Dim {dim} Value')
                    ax.set_ylabel('Frequency')
                    ax.set_xlim(-1.1, 1.1)
                    ax.grid(True, linestyle='--', alpha=0.5)
                    
                    # Mark mean and std
                    mean = np.mean(actions[:, dim])
                    std = np.std(actions[:, dim])
                    ax.axvline(mean, color='red', linestyle='--', label=f'Mean: {mean:.3f}')
                    ax.axvline(mean + std, color='orange', linestyle=':', label=f'±Std: {std:.3f}')
                    ax.axvline(mean - std, color='orange', linestyle=':')
                    ax.legend()
                    
                    
                def plot_q_value_distribution(ax, critic_fn, observations, actions, dim):
                    """
                    Plot Q-value distribution for a specific action dimension.
                    
                    Args:
                        ax: Matplotlib axis object
                        critic_fn: Function that calculates Q-values (agent.calculate_q_value)
                        observations: Array of observations (shape: [batch_size, obs_dim])
                        actions: Array of actions (shape: [batch_size, action_dim])
                        dim: Action dimension to analyze
                    """
                    # Create a range of values for the target dimension (-1 to 1)
                    test_actions = np.tile(actions, (100, 1))  # Copy actions 100 times
                    test_values = np.linspace(-1, 1, 100)  # Test 100 values across action range
                    test_actions[:, dim] = test_values  # Vary only the target dimension
                    
                    # Calculate Q-values for all test actions
                    q_values = critic_fn(observations=np.tile(observations, (100, 1)),
                                        actions=test_actions)
                    
                    # Plot Q-values vs action values
                    ax.plot(test_values, q_values, 'b-', linewidth=2)
                    ax.set_title(f'Q-values vs Action Dim {dim}')
                    ax.set_xlabel(f'Action Dim {dim} Value')
                    ax.set_ylabel('Q-value')
                    ax.set_xlim(-1.1, 1.1)
                    ax.grid(True, linestyle='--', alpha=0.5)
                    
                    # Mark optimal action
                    optimal_idx = np.argmax(q_values)
                    optimal_value = test_values[optimal_idx]
                    optimal_q = q_values[optimal_idx]
                    ax.axvline(optimal_value, color='r', linestyle='--', 
                            label=f'Optimal: {optimal_value:.2f}')
                    ax.legend()
                    
                    return optimal_value, optimal_q
                                    
                    
    
                # Create figure
                fig, axes = plt.subplots(3, action_dim + 1, figsize=(15, 6))
                
                # Sample and plot for actor_generate_fn
                gen_actions = mc_sample_actions(actor_generate_fn, observation, monte_carlo, eval_temperature)
                current_episode_actions.append(gen_actions)
                for dim in trange(action_dim):
                    plot_action_distribution(axes[0][dim+1], gen_actions, dim, '-Generate_fn')
                    
                # Sample and plot for actor_fn
                distilled_actions = mc_sample_actions(actor_fn, observation, monte_carlo, eval_temperature)
                current_episode_actions.append(distilled_actions)
                for dim in trange(action_dim):
                    plot_action_distribution(axes[1][dim+1], distilled_actions, dim, '--Distilled')
                    plot_q_value_distribution(
                            ax=axes[2][dim+1],
                            critic_fn=critic_fn,
                            observations=np.array([observation]),  # Wrap as batch of 1
                            actions=action_interact,  # Use first action as template
                            dim=dim  # Analyze first action dimension
                    )

                # put the frame on the plt
                frame = env.render().copy()
                axes[0][0].imshow(frame)
                
                # save the image
                plt.tight_layout()
                dir_path = f'./figures/{current_time}/action'
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)
                plt.savefig(f'{dir_path}/action_distribution_{i}_step_{step}.png')
                print(f'Action distribution saved to: {dir_path}/action_distribution_{i}_step_{step}.png')
                plt.close(fig)

            next_observation, reward, terminated, truncated, info = env.step(action_interact)
            
            # for logging Q
            info['q_value'] = q.item() if viz_GMM else 0.0
            info['reward'] = reward
            info['q-reward_diff'] = info['q_value'] - reward
            info['abs_q-reward_diff'] = abs(info['q_value'] - reward)
            
            done = terminated or truncated
            step += 1

            if should_render and (step % video_frame_skip == 0 or done):
                frame = env.render().copy()
                render.append(frame)

            transition = dict(
                observation=observation,
                next_observation=next_observation,
                action=action,
                reward=reward,
                done=done,
                info=info,
            )
            add_to(traj, transition)
            observation = next_observation
            
            if viz_GMM:
                actions_list.append(action)
                noises_list.append(noises)
                actions_fm_list.append(action_fm)


        print(f'Time per Step: {np.mean(infer_time_list):.4f} ms')
        
        # print(f' info_success: {info["success"]}; info_episode: {info["episode"]}')
        
        if i < num_eval_episodes:
            add_to(stats, flatten(info))
            trajs.append(traj)
        else:
            renders.append(np.array(render))
    
    print(f'Average Inference Time per Step: {np.mean(infer_time_list):.4f} ms')
          

    for k, v in stats.items():
        stats[k] = np.mean(v)

    print(f' info_success: {stats["success"]}, info_reward: {stats["reward"]}')
    if viz_GMM:
        actions_array = np.array(actions_list)
        noises_array = np.array(noises_list)
        actions_fm_array = np.array(actions_fm_list)
        ax = env.vizualize()
        # plot action
        plt.scatter(actions_array[:, 0], actions_array[:, 1], s=50, color='blue', alpha=0.5, label='Actions')    
        plt.scatter(noises_array[:, 0], noises_array[:, 1], s=10, color='red', alpha=0.5, label='Noises')    
        plt.scatter(actions_fm_array[:, 0], actions_fm_array[:, 1], s=20, color='green', alpha=0.5, label='Actions_FM')
        dir_path = f'{save_dir}/action'
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        # plt.savefig(f'{dir_path}/viz_epoch_{epoch}.png', dpi=300)
        plt.xlim(-1.5,1.5)
        plt.ylim(-1.5,1.5)
        plt.savefig(f'{save_dir}/viz-{epoch}.pdf', dpi=300)
        plt.close()
        print(f'Action distribution saved to: {save_dir}/viz-{epoch}.pdf')
        print()

    return stats, trajs, renders
