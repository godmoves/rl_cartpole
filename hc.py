import gym
import matplotlib.pyplot as plt
import numpy as np
from collections import deque

from reduceobs import ReduceObs


is_reduce_ob = False

env_name = 'CartPole-v1'
state_size = 4
action_size = 2

max_episodes = 10000
max_timesteps = 500
gamma = 1.0

learning_rate = 1e-3
search_rate = 1e-2


class Policy():
    def __init__(self, state_size, action_size):
        self.w = 1e-4 * np.random.rand(state_size, action_size)

    def forward(self, state):
        x = np.dot(state, self.w)
        return np.exp(x) / sum(np.exp(x))

    def act(self, state):
        probs = self.forward(state)
        action = np.argmax(probs)
        return action


def hill_climbing(env, policy, max_e, max_t, gamma):
    scores_deque = deque(maxlen=20)
    scores = []
    best_R = -np.Inf
    best_w = policy.w

    for e in range(max_e):
        rewards = []
        state = env.reset()
        state = np.append(state, state)[-state_size:]
        prev_state = state

        for t in range(max_t):
            action = policy.act(state)
            state, reward, done, _ = env.step(action)
            rewards.append(reward)
            state = np.append(prev_state, state)[-state_size:]
            prev_state = state
            if done:
                break
        scores_deque.append(sum(rewards))
        scores.append(sum(rewards))

        discounts = [gamma ** i for i in range(len(rewards) + 1)]
        R = sum([a * b for a, b in zip(discounts, rewards)])

        if R >= best_R:  # found better weights
            best_R = R
            best_w = policy.w
            policy.w += learning_rate * np.random.rand(*policy.w.shape)
        else:
            policy.w = best_w + search_rate * np.random.rand(*policy.w.shape)

        if e % 10 == 0:
            print('episode: {} | avg score: {:.2f}'.format(e, np.mean(scores_deque)))

        if len(scores) >= 20 and all(np.array(scores[-20:]) >= 200.0):
            print('Solved in {:d} episodes'.format(e - 20))
            policy.w = best_w
            break

    return scores


def main():
    env = ReduceObs(gym.make(env_name)) if is_reduce_ob else gym.make(env_name)
    # env.seed(500)
    # np.random.seed(500)

    print('state size:', state_size)
    print('action size:', action_size)
    policy = Policy(state_size, action_size)
    scores = hill_climbing(env, policy, max_episodes, max_timesteps, gamma)

    plt.plot(scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()


if __name__ == '__main__':
    main()
