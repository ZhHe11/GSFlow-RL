import gymnasium as gym
from gymnasium import spaces
import numpy as np
from utils.datasets import Dataset
import matplotlib.pyplot as plt
import seaborn as sns

font_properties = {
    'family': 'Liberation Serif',
    'size': 28
}


class MulitCrescentEnv(gym.Env):
    """
    一个定制的2D动作空间环境。
    - 奖励在动作空间的右上角最高。
    - 状态是固定的，不影响奖励。
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, radius=1, reward_std=0.4, base_reward=0):
        super(MulitCrescentEnv, self).__init__()
        
        # 定义动作空间: 2D, 每个维度在 [-1, 1]
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        
        # 定义观察空间: 固定的2D状态
        self.observation_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        
        # 固定的初始状态
        self._initial_state = np.array([0.0, 0.0], dtype=np.float32)
        
        self.radius = radius
        self.reward_std = reward_std
        self.base_reward = base_reward
        
        self.centers = self._create_hexagonal_centers()
        self.rewards_for_centers = self._assign_rewards()
        
        # 用于高斯奖励计算的参数
        self.cov = np.array([[self.reward_std**2, 0], [0, self.reward_std**2]])
        self.inv_cov = np.linalg.inv(self.cov)
        
        self._sorted_centers = sorted(self.rewards_for_centers.items(), key=lambda item: item[1], reverse=True)
        
    def _create_hexagonal_centers(self):
        """使用极坐标计算正六边形的顶点。"""
        centers = {}
        # 角度: 0, 60, 120, 180, 240, 300 度
        angles_deg = [0, 60, 120, 180, 240, 300]
        names = ['right', 'top_right', 'top_left', 'left', 'bottom_left', 'bottom_right']
        
        for name, angle_deg in zip(names, angles_deg):
            angle_rad = np.deg2rad(angle_deg)
            x = self.radius * np.cos(angle_rad)
            y = self.radius * np.sin(angle_rad)
            centers[name] = np.array([x, y], dtype=np.float32)
        
        print(centers)
        return centers

    def _assign_rewards(self):
        """为六个中心分配奖励值。"""
        return {
            'top_right': 15.0,    # 最高奖励
            'left': 10.0,        # 次高奖励
            'bottom_right': 5.0,  # 次高奖励
            'right': 10.0,       # 低奖励
            'top_left': 5.0,    # 低奖励
            'bottom_left': 15.0, # 低奖励
        }


    def _get_reward(self, position):
        """
        计算给定位置的奖励值。
        奖励 = 基础惩罚 + 所有中心点的高斯奖励之和。
        
        Args:
            position (np.ndarray): 当前位置，形状为 (2,)。
            
        Returns:
            float: 该位置的奖励值。
        """
        position = np.array(position, dtype=np.float32)
        
        # # 定义月牙区域
        # center_main = np.array([0.5, 0.8660254])
        # center_offset = np.array([0.5+0.2, 0.8660254+0.2])
        # dist_main = np.linalg.norm(position - center_main)
        # dist_offset = np.linalg.norm(position - center_offset)
        
        # # 如果点在主圆内，且在偏移圆之外，则为高奖励
        # if dist_main < 0.4 and dist_offset > 0.4:
        #     return 20 # High reward
        
        
        
        # names = ['right', 'top_right', 'top_left', 'left', 'bottom_left', 'bottom_right']
        
        rewards_in_range = []
        
        for name, center_pos in self.centers.items():
            # 计算欧式距离
            distance = np.linalg.norm(position - center_pos)
            
            if name == 'right':
                center_offset = center_pos + np.array([0.2828, 0])
            elif name == 'left':
                center_offset = center_pos + np.array([-0.2828, 0])
            elif name == 'top_left':
                center_offset = center_pos + np.array([-0.2, 0.2])
            elif name == 'bottom_right':
                center_offset = center_pos + np.array([0.2, -0.2])
            elif name == 'bottom_left':
                center_offset = center_pos + np.array([-0.2, -0.2])
            elif name == 'top_right':
                center_offset = center_pos + np.array([0.2, 0.2])
            else:
                break
            
            dist_offset = np.linalg.norm(position - center_offset)
            
            if distance < 0.4 and dist_offset > 0.4:
                rewards_in_range.append(self.rewards_for_centers[name])
    
        if not rewards_in_range:
            # 如果不在任何圆圈内
            return self.base_reward
        else:
            # 如果在一个或多个圆圈内，返回其中的最高奖励
            return max(rewards_in_range)
        
        
    def step(self, action):
        """执行一步。由于状态固定，这一步总是结束。"""
        # 确保动作在有效范围内
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        reward = self._get_reward(action)
        
        # 状态不改变
        observation = self._initial_state
        
        # 每一-步都是终止状态
        terminated = True
        truncated = False
        
        # success 的判断：
        if reward >= 10:
            success = True
        else:
            success = False
        
        info = {'success':success,  # 如果动作在右上象限，则成功
                'reward': reward,  # 包含奖励信息
                'episode': 1,  # 每个 episode 只有一步
                'action': action.tolist()  # 包含执行的动作
                }
        
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """重置环境到初始状态。"""
        super().reset(seed=seed)
        observation = self._initial_state
        info = {}
        return observation, info

    def render(self, mode='human'):
        # 对于这个环境，可视化将在外部脚本中完成
        pass

    def close(self):
        pass
    
    
    def vizualize(self, ax=None):
        
        """
        可视化环境的奖励函数。
        在右上象限的奖励最高，左下象限的奖励最低。
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))
        else:
            fig = ax.get_figure() # <--- 新增：如果ax存在，就获取它所属的fig

        sns.set_theme(style="white")
        x = np.linspace(-1.5, 1.5, 100)
        y = np.linspace(-1.5, 1.5, 100)
        X, Y = np.meshgrid(x, y)
        
        # 计算每个点的奖励值
        Z = np.array([self._get_reward(np.array([i, j])) for i, j in zip(np.ravel(X), np.ravel(Y))])
        Z = Z.reshape(X.shape)
        
        # 绘制等高线图
        contour = ax.contourf(X, Y, Z, levels=50, cmap='Blues', alpha=0.8)
        ax.set_aspect('equal', adjustable='box')
        
        # --- 修改：使用 fig.colorbar 而不是 plt.colorbar ---
        cbar = fig.colorbar(contour, ax=ax, shrink=0.7) # <--- 在这里添加 shrink 参数
        # font_properties2 = {
        #     'family': 'Liberation Serif',
        #     'size': 12
        # }
        font_properties2 = {
            'family': 'Liberation Serif',
            'size': 26
        }
        cbar.set_label('Reward Value', fontdict=font_properties2)
        
        # 设置坐标轴和标题的字体
        ax.set_xlabel("X", fontdict=font_properties)
        ax.set_ylabel("Y", fontdict=font_properties)
        ax.set_title("Reward Landscape", fontdict=font_properties)
        
        # 自动调整布局以防止标签被截断
        plt.tight_layout() 
        
        # 保存图像
        plt.savefig('ToyEnv.png')
        
        return ax
    
    
    
    

def generate_offline_dataset(env, num_points_per_quadrant=250, skip=None):
    """
    为 FourQuadrantsEnv 生成一个多模态的、次优的离线数据集。

    Args:
        env (gym.Env): FourQuadrantsEnv 的一个实例。
        num_points_per_quadrant (int): 每个象限采样的数据点数量。

    Returns:
        dict: 一个类似 D4RL 格式的数据集字典，包含 'observations', 'actions',
              'rewards', 'terminals', 'timeouts'。
    """
    print(f"Generating offline dataset with {num_points_per_quadrant * 4} total points...")

    # 1. 定义四个高斯分布的中心点和协方差
    centers = env.centers
    # 较小的协方差，让数据点聚集在中心附近
    cov = [[0.02, 0], [0, 0.02]]
    
    # 2. 从每个分布中采样动作
    all_actions = []
    for center_name in centers.keys():
        print(center_name)
        
        if center_name in skip:
            continue
        actions = np.random.multivariate_normal(centers[center_name], cov, num_points_per_quadrant)
        
        for action in actions:
            for name, center_pos in env.centers.items():
                # 计算欧式距离
                distance = np.linalg.norm(action - center_pos)
                
                if name == 'right':
                    center_offset = center_pos + np.array([0.2828, 0])
                elif name == 'left':
                    center_offset = center_pos + np.array([-0.2828, 0])
                elif name == 'top_left':
                    center_offset = center_pos + np.array([-0.2, 0.2])
                elif name == 'bottom_right':
                    center_offset = center_pos + np.array([0.2, -0.2])
                elif name == 'bottom_left':
                    center_offset = center_pos + np.array([-0.2, -0.2])
                elif name == 'top_right':
                    center_offset = center_pos + np.array([0.2, 0.2])
                else:
                    break
                
                dist_offset = np.linalg.norm(action - center_offset)
                print(dist_offset)
                
                if distance < 0.4 and dist_offset > 0.4:
                    all_actions.append(action)
        
    
    # 将所有动作合并成一个大数组
    actions_np = np.vstack(all_actions).astype(np.float32)
    
    # 打乱数据顺序
    np.random.shuffle(actions_np)
    
    num_total_points = len(actions_np)

    # 3. 初始化用于存储数据集的列表
    obs_list = []
    rewards_list = []
    terminals_list = []
    timeouts_list = []

    # 4. 与环境交互，获取每个动作对应的数据
    for i in range(num_total_points):
        action = actions_np[i]
        
        # 因为状态是固定的，所以每次都重置环境以获取初始状态
        obs, _ = env.reset()
        
        # 执行动作
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        obs_list.append(obs)
        rewards_list.append(reward)
        terminals_list.append(terminated)
        timeouts_list.append(truncated)
        
    # return actions_np
    
    dataset = Dataset.create(
        observations=np.array(obs_list, dtype=np.float32),
        actions=actions_np,
        next_observations=np.array(obs_list, dtype=np.float32), 
        rewards=np.array(rewards_list, dtype=np.float32),
        terminals=np.array(terminals_list, dtype=np.bool_),
        masks= np.array(timeouts_list, dtype=np.bool_) 
    )
    
    print("Dataset generation complete.")
    print("Dataset keys:", dataset.keys())
    print("Observations shape:", dataset['observations'].shape)
    print("Actions shape:", dataset['actions'].shape)
    
    # 验证一下平均动作是否接近 (0,0)
    mean_action = np.mean(dataset['actions'], axis=0)
    print(f"Mean action of the dataset: {mean_action}") # 应该非常接近 [0, 0]
    
    return dataset


if __name__ == "__main__":
    env = MulitCrescentEnv()
    ax = env.vizualize()
    # obs = generate_offline_dataset(env, num_points_per_quadrant=250, skip=['top_right', 'bottom_left'])
    # plt.scatter(obs[:,0], obs[:,1], s=1, color='red', alpha=0.5, label='Sampled Actions')
    sns.set_theme(style="white")
    plt.savefig("four_quadrants_reward_landscape.png")
    
    