import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import namedtuple, deque
import tqdm

import matplotlib.pyplot as plt
import random
import gymnasium as gym
import time
import pickle
import pygame


# Set random seeds
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

# Create CartPole environment
env = gym.make('CartPole-v1', render_mode='rgb_array')

# Define Q-Network
class FullyConnectedModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(FullyConnectedModel, self).__init__()

        # Define layers with ReLU activation
        self.linear1 = nn.Linear(input_size, 16)
        self.activation1 = nn.ReLU()
        self.linear2 = nn.Linear(16, 16)
        self.activation2 = nn.ReLU()
        self.linear3 = nn.Linear(16, 16)
        self.activation3 = nn.ReLU()

        # Output layer without activation function
        self.output_layer = nn.Linear(16, output_size)

        # Initialization using Xavier uniform
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.xavier_uniform_(self.linear3.weight)
        nn.init.xavier_uniform_(self.output_layer.weight)

    def forward(self, inputs):
        x = self.activation1(self.linear1(inputs))
        x = self.activation2(self.linear2(x))
        x = self.activation3(self.linear3(x))
        x = self.output_layer(x)
        return x

class QNetwork:
    def __init__(self, lr):
        self.net = FullyConnectedModel(4, 2)
        self.lr = lr
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)

    def load_model(self, model_file):
        self.net.load_state_dict(torch.load(model_file))

    def load_model_weights(self, weight_file):
        self.net.load_state_dict(torch.load(weight_file))

# Define Replay Memory
class ReplayMemory:
    def __init__(self, memory_size=50000, burn_in=10000):
        self.memory_size = memory_size
        self.burn_in = burn_in
        self.memory = deque([], maxlen=memory_size)

    def sample_batch(self, batch_size=32):
        return random.sample(self.memory, batch_size)

    def append(self, transition):
        self.memory.append(transition)

# Define DQN Agent
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class DQN_Agent:
    def __init__(self, environment_name, lr=5e-4, render=False):
        self.env = gym.make(environment_name)
        self.lr = lr
        self.policy_net = QNetwork(self.lr)
        self.target_net = QNetwork(self.lr)
        self.target_net.net.load_state_dict(self.policy_net.net.state_dict())  # Copy the weight of the policy network
        self.rm = ReplayMemory()
        self.burn_in_memory()
        self.batch_size = 32
        self.gamma = 0.99
        self.c = 0
        self.render = render

    def burn_in_memory(self):
        cnt = 0
        terminated = False
        truncated = False
        state, _ = self.env.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        while cnt < self.rm.burn_in:
            if terminated or truncated:
                state, _ = self.env.reset()
                state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            action = torch.tensor([self.env.action_space.sample()], dtype=torch.long).unsqueeze(0)
            next_state, reward, terminated, truncated, _ = self.env.step(action.item())
            reward = torch.tensor([reward], dtype=torch.float32)
            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)

            transition = Transition(state, action, next_state, reward)
            self.rm.append(transition)
            state = next_state
            cnt += 1

    def epsilon_greedy_policy(self, q_values, epsilon=0.05):
        if random.random() > epsilon:
            with torch.no_grad():
                return self.greedy_policy(q_values)
        else:
            return torch.tensor([[self.env.action_space.sample()]], dtype=torch.long)

    def greedy_policy(self, q_values):
        return torch.argmax(q_values, dim=1).unsqueeze(1)

    def train(self):
        state, _ = self.env.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        terminated = False
        truncated = False

        while not (terminated or truncated):
            with torch.no_grad():
                q_values = self.policy_net.net(state)

            action = self.epsilon_greedy_policy(q_values).reshape(1, 1)

            next_state, reward, terminated, truncated, _ = self.env.step(action.item())
            reward = torch.tensor([reward], dtype=torch.float32)
            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)

            transition = Transition(state, action, next_state, reward)
            self.rm.append(transition)

            state = next_state

            if len(self.rm.memory) >= self.batch_size:
                transitions = self.rm.sample_batch(self.batch_size)
                batch = Transition(*zip(*transitions))
                non_final_mask = torch.tensor([s is not None for s in batch.next_state], dtype=torch.bool)
                non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
                state_batch = torch.cat(batch.state)
                action_batch = torch.cat(batch.action)
                reward_batch = torch.cat(batch.reward)

                state_action_values = self.policy_net.net(state_batch).gather(1, action_batch)
                next_state_values = torch.zeros(self.batch_size, dtype=torch.float32)

                with torch.no_grad():
                    next_state_values[non_final_mask] = self.target_net.net(non_final_next_states).max(1)[0]

                expected_state_action_values = (next_state_values * self.gamma) + reward_batch
                criterion = nn.MSELoss()
                loss = criterion(state_action_values.squeeze(), expected_state_action_values)
                self.policy_net.optimizer.zero_grad()
                loss.backward()
                self.policy_net.optimizer.step()

            self.c += 1
            if self.c % 50 == 0:
                self.target_net.net.load_state_dict(self.policy_net.net.state_dict())

    def test(self, model_file=None):
        if model_file:
            self.policy_net.load_model(model_file)

        state, _ = self.env.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        terminated = False
        truncated = False
        rewards = []

        while not (terminated or truncated):
            with torch.no_grad():
                q_values = self.policy_net.net(state)
            action = self.greedy_policy(q_values)
            state, reward, terminated, truncated, _ = self.env.step(action.item())
            rewards.append(reward)
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        return np.sum(rewards)

# Train the agent
def train_agent(agent, num_episodes=500):
    scores = []
    repeats = 0

    for i_episode in tqdm.tqdm(range(num_episodes)):
        agent.train()
        
        if i_episode % 10 == 0:
            score = agent.test()
            scores.append(score)
            print(f'Episode {i_episode}, Score: {score}')
    
    plt.plot(scores)
    plt.xlabel('Episodes (x10)')
    plt.ylabel('Rewards')
    plt.title('Training Training Progress')
    plt.show()



def visualize_agent(agent, environment_name='CartPole-v1', num_episodes=5):
    # Create environment with human rendering mode
    env = gym.make(environment_name, render_mode='human')

    for episode in range(num_episodes):
        state, _ = env.reset()
        terminated = False
        truncated = False
        episode_reward = 0

        while not (terminated or truncated):
            # Convert state to tensor
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

            # Get Q-values from the policy network
            with torch.no_grad():
                q_values = agent.policy_net.net(state_tensor)

            # Select action based on the greedy policy
            action = agent.greedy_policy(q_values).item()

            # Take action in the environment
            state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward

        print(f"Episode {episode + 1} finished with reward: {episode_reward}")
        time.sleep(2)

    env.close()
    
    
def train():
    num_episodes = int(input("Episode Number? "))
    agent = DQN_Agent('CartPole-v1', lr=5e-4)
    train_agent(agent, num_episodes)
    num_visualize_episodes = int(input("Visulization Number? "))
    visualize_agent(agent, 'CartPole-v1', num_visualize_episodes)
    with open('agent500.pkl', 'wb') as file:
        pickle.dump(agent, file)
        
        
def human_play():
    pygame.init()
    env = gym.make('CartPole-v1', render_mode='human')
    env.reset()
    done = False
    time.sleep(5)
    episode_reward = 0
    action = 0
    while not done:
        # Handle events for quitting and key presses
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True  # Exit the loop if the window is closed
            
        # Check for key presses
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            action = 0  # Move left
        elif keys[pygame.K_RIGHT]:
            action = 1  # Move right
        
        if action is not None:
            # Step the environment with the chosen action
            observation, reward, done, _, _ = env.step(action)
            episode_reward += 1

            # Render the environment
            env.render()
    print(f"Episode finished with reward: {episode_reward}")

print("0")
task = 5
agent500 = 1
agent300 = 1
while task != 4:
    task = 5
    task = int(input("Train [0], 500 AI Play [1], 300 AI Play [2], Human Play [3] or End [4]? "))
        
    if task == 0:
        train()
    elif task == 1:
        if agent500 == 1:
            with open('agent.pkl', 'rb') as file:
                agent500 = pickle.load(file)
        num_visualize_episodes = int(input("Visulization Number? "))
        visualize_agent(agent500, 'CartPole-v1', num_visualize_episodes)
    elif task == 2:
        if agent300 == 1:
            with open('agent300.pkl', 'rb') as file:
                agent300 = pickle.load(file)
        xxnum_visualize_episodes = int(input("Visulization Number? "))
        visualize_agent(agent300, 'CartPole-v1', num_visualize_episodes)
    elif task == 3:
        human_play()
