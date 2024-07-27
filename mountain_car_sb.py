import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
from stable_baselines3 import DQN
from gymnasium.wrappers import RecordVideo
import os


def run_q(episodes, is_training=True, render=False):
    env = gym.make('MountainCar-v0', render_mode='human' if render else None)

    pos_space = np.linspace(env.observation_space.low[0], env.observation_space.high[0], 20)
    vel_space = np.linspace(env.observation_space.low[1], env.observation_space.high[1], 20)

    if is_training:
        q = np.zeros((len(pos_space), len(vel_space), env.action_space.n))
    else:
        with open('mountain_car_q_table.pkl', 'rb') as f:
            q = pickle.load(f)

    learning_rate = 0.9
    discount_factor = 0.9
    epsilon = 1
    epsilon_decay = 2 / episodes
    rng = np.random.default_rng()

    rewards_per_episode = np.zeros(episodes)

    for i in range(episodes):
        state = env.reset()[0]
        state_p = np.digitize(state[0], pos_space)
        state_v = np.digitize(state[1], vel_space)
        terminated = False
        rewards = 0

        while not terminated and rewards > -1000:
            if is_training and rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q[state_p, state_v, :])

            new_state, reward, terminated, _, _ = env.step(action)
            new_state_p = np.digitize(new_state[0], pos_space)
            new_state_v = np.digitize(new_state[1], vel_space)

            if is_training:
                q[state_p][state_v][action] = q[state_p][state_v][action] + learning_rate * (
                    reward + discount_factor * np.max(q[new_state_p, new_state_v, :]) - q[state_p, state_v, action]
                )

            state = new_state
            state_p = new_state_p
            state_v = new_state_v
            rewards += reward

        epsilon = max(epsilon - epsilon_decay, 0)
        rewards_per_episode[i] = rewards

    env.close()

    if is_training:
        with open('mountain_car_q_table.pkl', 'wb') as f:
            pickle.dump(q, f)

    mean_rewards = np.zeros(episodes)
    for t in range(episodes):
        mean_rewards[t] = np.mean(rewards_per_episode[max(0, t - 100):t + 1])
    plt.plot(mean_rewards)
    plt.savefig('mountain_car_rewards.png')


def train_sb3():
    model_dir = "models"
    log_dir = "logs"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    env = gym.make('MountainCar-v0')

    model = DQN('MlpPolicy', env, verbose=1, device='cuda', tensorboard_log=log_dir)

    TIMESTEPS = 1000
    iters = 0
    while True:
        iters += 1
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
        model.save(f"{model_dir}/dqn_{TIMESTEPS * iters}")


def test_sb3(render=True, video_dir='./videos'):
    
    env = gym.make('MountainCar-v0', render_mode='rgb_array')
    
   
    os.makedirs(video_dir)
    env = RecordVideo(env, video_dir, episode_trigger=lambda episode_id: True)
    
    model = DQN.load('models/dqn_1499000', env=env)

    obs = env.reset()[0]
    terminated = False
    while not terminated:
        action, _ = model.predict(observation=obs, deterministic=True)
        obs, _, terminated, _, _ = env.step(action)

    env.close()
    print(f"Vídeo salvo no diretório: {video_dir}")

if __name__ == '__main__':
    # Treinamento/teste usando Q-Learning
    # run_q(1000, is_training=True, render=False)
    # run_q(1, is_training=False, render=True)

    # Treinamento/teste usando Stable Baselines3
    # train_sb3()
    test_sb3(render=False, video_dir='./videos')
