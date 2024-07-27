import argparse
import gymnasium as gym
import os
from car_racing import CarRacingDQNAgent
from car_racing import process_state_image
from car_racing import generate_state_frame_stack_from_queue
from collections import deque

# Configurações
RENDER = True  # Habilitar a renderização para visualizar o desempenho do agente
NUM_EPISODES = 5  # Número de episódios para avaliação

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
            
            print(f"Ação tomada: {action}")  # Adicione esta linha para verificar as ações

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

    # Configurar o ambiente e o agente
    env = gym.make('CarRacing-v2', render_mode='human' if RENDER else 'rgb_array')
    agent = CarRacingDQNAgent(epsilon=0.0)  # Epsilon zero para avaliação (não queremos exploração durante a avaliação)

    # Avaliar o modelo
    evaluate_model(args.model, env, agent)
