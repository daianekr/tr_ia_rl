import argparse
import gymnasium as gym
import os
from car_racing import CarRacingDQNAgent
from car_racing import process_state_image
from car_racing import generate_state_frame_stack_from_queue
from collections import deque
from gymnasium.wrappers import RecordVideo

# Configurações
RENDER = False  # Desabilitar a renderização humana para gravação de vídeo
NUM_EPISODES = 5  # Número de episódios para avaliação
VIDEO_DIR = './video'  # Diretório para salvar os vídeos

def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def evaluate_model(model_path, env, agent):
    # Carregar os pesos do modelo treinado
    agent.load(model_path)
    
    for e in range(1, NUM_EPISODES + 1):
        init_state, _ = env.reset()
        init_state = process_state_image(init_state)

        total_reward = 0
        state_frame_stack_queue = deque([init_state] * agent.frame_stack_num, maxlen=agent.frame_stack_num)
        done = False

        while not done:
            if RENDER:
                env.render()

            current_state_frame_stack = generate_state_frame_stack_from_queue(state_frame_stack_queue)
            action = agent.act(current_state_frame_stack)  # Remover argumento 'train'

            next_state, reward, done, _, _ = env.step(action)
            total_reward += reward

            next_state = process_state_image(next_state)
            state_frame_stack_queue.append(next_state)

        print(f'Episode: {e}/{NUM_EPISODES}, Total Reward: {total_reward:.2f}')

    env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a trained DQN agent to play CarRacing.')
    parser.add_argument('-m', '--model', required=True, help='Specify the model path to evaluate.')
    args = parser.parse_args()

    # Cria o diretório de vídeo se não existir
    ensure_directory_exists(VIDEO_DIR)

    # Configurar o ambiente com gravação de vídeo e o agente
    env = gym.make('CarRacing-v2', render_mode='rgb_array')
    env = RecordVideo(env, video_folder=VIDEO_DIR, episode_trigger=lambda e: True)
    agent = CarRacingDQNAgent(epsilon=0.0)  # Epsilon zero para avaliação (não queremos exploração durante a avaliação)

    try:
        # Avaliar o modelo
        evaluate_model(args.model, env, agent)
    finally:
        # Garantir que o ambiente seja fechado corretamente
        env.close()
