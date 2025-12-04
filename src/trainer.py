import numpy as np
import os
from .environment import TradingEnvironment
from .agent import PPOAgent

class Trainer:
    def __init__(self, env, agent, save_dir = 'models'):
        self.env = env
        self.agent = agent
        self.save_dir = save_dir

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        self.episode_rewards = []
        self.episode_lengths = []

    def train(self, num_episodes = 100, max_steps = 500, save_interval = 50):
        print(f"Starting training fotr{num_episodes} episodes...")

        for episode in range(1, num_episodes + 1):
            state, info = self.env.reset()
            episode_reward = 0
            episode_length = 0

            for step in range(max_steps):
                action = self.agent.get_action(state, training = True)

                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                self.agent.store_transition(reward, done)

                episode_reward += reward
                episode_length += 1
                state = next_state

                if done:
                    break

            actor_loss, critic_loss, entropy = self.agent.train()

            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)

            if episode % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-10:])
                avg_length = np.mean(self.episode_lengths[-10:])
                print(
                    f"Episode {episode}/{num_episodes} | "
                    f'Avg Reward: {avg_reward:.4f} | '
                    f"Avg Length: {avg_length:.4f} | "
                    f"Balance: ${info.get('balance', 0):.2f}"
                )

            if episode % save_interval == 0:
                model_path = os.path.join(self.save_dir, f"ppo_ep{episode}.pth")
                self.agent.save(model_path)

        print('Training completed.')

        final_path = os.path.join(self.save_dir, 'ppo_final.pth')
        self.agent.save(final_path)

        return self.episode_rewards, self.episode_lengths