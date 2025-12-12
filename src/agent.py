import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

class ActorCritic(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim = 256):
        super(ActorCritic, self).__init__()

        self.shared = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )

        self.actor = nn.Sequential(
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Softmax(dim = -1)
        )

        self.critic = nn.Linear(hidden_dim // 2, 1)
        
    def forward(self, state):
        """Forward Pass"""

        features = self.shared(state)
        action_probs = self.actor(features)
        state_value = self.critic(features)
        return action_probs, state_value

    def get_action(self, state):
        action_probs, state_value = self.forward(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        action_log_prob = dist.log_prob(action)
        return action.item(), action_log_prob, state_value

class PPOAgent:
    def __init__(self, input_dim, action_dim, lr = 3e-4, gamma = 0.99, gae_lambda = 0.95, clip_epsilon = 0.2, entropy_coeff = 0.01, value_loss_coeff = 0.5, epochs = 10, batch_size = 64, device = 'cpu'):

        # Hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coeff = entropy_coeff
        self.value_loss_coeff = value_loss_coeff
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device

        # Initialize network
        self.policy = ActorCritic(input_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr = lr)

        # Storage for training
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def get_action(self, state, training = True):
        state_tensor = torch.FloatTensor(state).to(self.device).unsqueeze(0)

        with torch.no_grad():
            action, log_prob, value = self.policy.get_action(state_tensor)

        if training:
            self.states.append(state)
            self.actions.append(action)
            self.log_probs.append(log_prob.item())
            self.values.append(value.item())

        return action

    def store_transition(self, reward, done):
        self.rewards.append(reward)
        self.dones.append(done)

    def compute_gae(self, next_value):
        advantages = []
        gae = 0

        values = self.values + [next_value]

        for t in reversed(range(len(self.rewards))):
            delta = self.rewards[t] + self.gamma * values[t + 1] * (1 - self.dones[t]) - values[t]

            gae = delta + self.gamma * self.gae_lambda * (1 - self.dones[t]) * gae

            advantages.insert(0, gae)

        returns = [adv + val for adv, val in zip(advantages, self.values)]

        return advantages, returns
    
    def train(self):
        if len(self.states) == 0:
            return 0, 0, 0 

        last_state = torch.FloatTensor(self.states[-1]).to(self.device).unsqueeze(0)
        
        with torch.no_grad():
            _, next_value = self.policy(last_state)
            next_value = next_value.item()

        advantages, returns = self.compute_gae(next_value)

        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy_loss = 0

        for epoch in range(self.epochs):
            action_probs, values = self.policy(states)
            values = values.squeeze()

            dist = Categorical(action_probs)
            entropy = dist.entropy().mean()

            new_log_probs = dist.log_prob(actions)
            ratio = torch.exp(new_log_probs - old_log_probs)

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
                
            actor_loss = -torch.min(surr1, surr2).mean()

            # MSE
            critic_loss = nn.MSELoss()(values, returns)
            
            # Total loss
            loss = actor_loss + self.value_loss_coeff * critic_loss - self.entropy_coeff * entropy

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()

            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            total_entropy_loss += entropy.item()

        self.clear_memory()

        return (total_actor_loss / self.epochs,
        total_critic_loss / self.epochs,
        total_entropy_loss / self.epochs)

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []

    def save(self, filepath):
        torch.save(self.policy.state_dict(), filepath)
        print(f"Model saved to {filepath}")

    def load(self, filepath):
        self.policy.load_state_dict(torch.load(filepath, map_location=self.device))
        print(f"Model loaded from {filepath}")