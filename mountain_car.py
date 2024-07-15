import gymnasium as gym
import numpy as np

def run():
    env = gym.make('MountainCar-v0', render_mode='human')

    pos_space = np.linspace(env.observation_space.low[0], env.observation_space.high[0], 20)
    vel_space = np.linspace(env.observation_space.low[1], env.observation_space.high[1], 20)
    

    state = env.reset()[0]
    terminated = False

    rewards = 0

    while(not terminated and rewards >-1000):

        action = env.action_space.sample()

        new_state, reward, terminated,_,_ = env.step(action)

        state = new_state

        rewards += reward
    env.close()

if __name__ == '__main__':
    run()