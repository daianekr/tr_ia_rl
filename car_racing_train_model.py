import argparse
import gymnasium as gym
from collections import deque
import os
from car_racing import CarRacingDQNAgent
from car_racing import process_state_image
from car_racing import generate_state_frame_stack_from_queue
from tensorflow.keras.callbacks import TensorBoard

# Configurações
RENDER = False
STARTING_EPISODE = 1
ENDING_EPISODE = 1000
SKIP_FRAMES = 2
TRAINING_BATCH_SIZE = 64
SAVE_TRAINING_FREQUENCY = 25
UPDATE_TARGET_MODEL_FREQUENCY = 5
LOG_DIR = './logs'
LEARNING_RATE = 0.001
GAMMA = 0.99
EPSILON_DECAY = 0.995

def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

if __name__ == '__main__':
    # Argumentos da linha de comando
    parser = argparse.ArgumentParser(description='Training a DQN agent to play CarRacing.')
    parser.add_argument('-m', '--model', help='Specify the last trained model path if you want to continue training after it.')
    parser.add_argument('-s', '--start', type=int, help='The starting episode, default to 1.')
    parser.add_argument('-e', '--end', type=int, help='The ending episode, default to 1000.')
    parser.add_argument('-p', '--epsilon', type=float, default=1.0, help='The starting epsilon of the agent, default to 1.0.')
    args = parser.parse_args()

    # Cria os diretórios 'save' e 'logs' se não existirem
    save_dir = './save_1'
    ensure_directory_exists(save_dir)
    ensure_directory_exists(LOG_DIR)

    # Configura o ambiente e o agente
    env = gym.make('CarRacing-v2', render_mode='human' if RENDER else 'rgb_array')
    agent = CarRacingDQNAgent(epsilon=args.epsilon, learning_rate=LEARNING_RATE, gamma=GAMMA)
    if args.model:
        agent.load(args.model)
    if args.start:
        STARTING_EPISODE = args.start
    if args.end:
        ENDING_EPISODE = args.end

    # Inicializa o TensorBoard
    tensorboard_callback = TensorBoard(log_dir=LOG_DIR, histogram_freq=1)

    # Loop de treinamento
    for e in range(STARTING_EPISODE, ENDING_EPISODE + 1):
        init_state, _ = env.reset()
        init_state = process_state_image(init_state)

        total_reward = 0
        negative_reward_counter = 0
        state_frame_stack_queue = deque([init_state] * agent.frame_stack_num, maxlen=agent.frame_stack_num)
        time_frame_counter = 1
        done = False

        while True:
            if RENDER:
                env.render()

            current_state_frame_stack = generate_state_frame_stack_from_queue(state_frame_stack_queue)
            action = agent.act(current_state_frame_stack)

            reward = 0
            for _ in range(SKIP_FRAMES + 1):
                next_state, r, done, _, _ = env.step(action)
                reward += r
                if done:
                    break

            negative_reward_counter = negative_reward_counter + 1 if time_frame_counter > 100 and reward < 0 else 0

            if action[1] == 1 and action[2] == 0:
                reward *= 1.5

            total_reward += reward

            next_state = process_state_image(next_state)
            state_frame_stack_queue.append(next_state)
            next_state_frame_stack = generate_state_frame_stack_from_queue(state_frame_stack_queue)

            agent.memorize(current_state_frame_stack, action, reward, next_state_frame_stack, done)

            if done or negative_reward_counter >= 25 or total_reward < 0:
                print(f'Episode: {e}/{ENDING_EPISODE}, Scores(Time Frames): {time_frame_counter}, Total Rewards(adjusted): {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}')
                break
            if len(agent.memory) > TRAINING_BATCH_SIZE:
                agent.replay(TRAINING_BATCH_SIZE)
            time_frame_counter += 1

        if e % UPDATE_TARGET_MODEL_FREQUENCY == 0:
            agent.update_target_model()

        if e % SAVE_TRAINING_FREQUENCY == 0:
            model_save_path = os.path.join(save_dir, f'trial_{e}.weights.h5')
            agent.save(model_save_path)

        # Decaimento do epsilon
        if agent.epsilon > 0.1:  # Epsilon mínimo
            agent.epsilon *= EPSILON_DECAY

    env.close()
