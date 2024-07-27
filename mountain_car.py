import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
from gymnasium.wrappers import RecordVideo
import os

def run(episodes, is_training=False, render=True, video_path='./my_videos'):
    # Cria o diretório de vídeos se não existir
    if render and not is_training:
        os.makedirs(video_path, exist_ok=True)

    # Cria o ambiente com a configuração de renderização apropriada
    env = gym.make('MountainCar-v0', render_mode='rgb_array' if render else None)
    
    # Envolve o ambiente com RecordVideo apenas no modo de teste
    if render and not is_training:
        env = RecordVideo(env, video_path, episode_trigger=lambda episode_id: True)
    
    pos_space = np.linspace(env.observation_space.low[0], env.observation_space.high[0], 20)
    vel_space = np.linspace(env.observation_space.low[1], env.observation_space.high[1], 20)
    
    if is_training:
        q = np.zeros((len(pos_space), len(vel_space), env.action_space.n))
    else:
        with open('mountain_car.pkl', 'rb') as f:
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
        with open('mountain_car.pkl', 'wb') as f:
            pickle.dump(q, f)

    mean_rewards = np.zeros(episodes)
    for t in range(episodes):
        mean_rewards[t] = np.mean(rewards_per_episode[max(0, t - 100):t + 1])
    plt.plot(mean_rewards)
    plt.savefig('mountain_car.png')

if __name__ == '__main__':
    # Para treinar:
    # run(5000, is_training=True, render=False)
    
    # Para testar e gravar vídeo:
    run(1, is_training=False, render=True, video_path='./my_videos')
