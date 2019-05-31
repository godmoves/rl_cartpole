from collections import namedtuple, deque

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from reduceobs import ReduceObs

env_name = 'CartPole-v1'
gamma = 0.99
batch_size = 32
lr = 0.001
initial_exploration = 1000
goal_score = 200
log_interval = 10
update_target = 100
replay_memory_capacity = 1000
device = torch.device('cpu')

sequence_length = 32
burn_in_length = 4
eta = 0.9
local_mini_batch = 8
n_step = 2
over_lapping_length = 16


Transition = namedtuple('Transition', ('state', 'next_state', 'action', 'reward', 'mask', 'step', 'rnn_state'))


class LocalBuffer(object):
    def __init__(self):
        self.n_step_memory = []
        self.local_memory = []
        self.memory = []
        self.over_lapping_from_prev = []

    def push(self, state, next_state, action, reward, mask, rnn_state):
        self.n_step_memory.append(
            [state, next_state, action, reward, mask, rnn_state])
        if len(self.n_step_memory) == n_step or mask == 0:
            [state, _, action, _, _, rnn_state] = self.n_step_memory[0]
            [_, next_state, _, _, mask, _] = self.n_step_memory[-1]

            sum_reward = 0
            for t in reversed(range(len(self.n_step_memory))):
                [_, _, _, reward, _, _] = self.n_step_memory[t]
                sum_reward += reward + gamma * sum_reward
            reward = sum_reward
            step = len(self.n_step_memory)
            self.push_local_memory(
                state, next_state, action, reward, mask, step, rnn_state)
            self.n_step_memory = []

    def push_local_memory(self, state, next_state, action, reward, mask, step, rnn_state):
        self.local_memory.append(Transition(
            state, next_state, action, reward, mask, step, torch.stack(rnn_state).view(2, -1)))
        if (len(self.local_memory) + len(self.over_lapping_from_prev)) == sequence_length or mask == 0:
            self.local_memory = self.over_lapping_from_prev + self.local_memory
            length = len(self.local_memory)
            while len(self.local_memory) < sequence_length:
                self.local_memory.append(Transition(
                    torch.Tensor([0, 0]),
                    torch.Tensor([0, 0]),
                    0, 0, 0, 0,
                    torch.zeros([2, 1, 16]).view(2, -1)
                ))
            self.memory.append([self.local_memory, length])
            if mask == 0:
                self.over_lapping_from_prev = []
            else:
                self.over_lapping_from_prev = self.local_memory[len(
                    self.local_memory) - over_lapping_length:]
            self.local_memory = []

    def sample(self):
        episodes = self.memory
        batch_state, batch_next_state, batch_action, batch_reward, batch_mask, batch_step, batch_rnn_state = [
        ], [], [], [], [], [], []
        lengths = []
        for episode, length in episodes:
            batch = Transition(*zip(*episode))

            batch_state.append(torch.stack(list(batch.state)))
            batch_next_state.append(torch.stack(list(batch.next_state)))
            batch_action.append(torch.Tensor(list(batch.action)))
            batch_reward.append(torch.Tensor(list(batch.reward)))
            batch_mask.append(torch.Tensor(list(batch.mask)))
            batch_step.append(torch.Tensor(list(batch.step)))
            batch_rnn_state.append(torch.stack(list(batch.rnn_state)))

            lengths.append(length)
        self.memory = []
        return Transition(batch_state, batch_next_state, batch_action, batch_reward, batch_mask, batch_step, batch_rnn_state), lengths


class Memory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        self.memory_probability = deque(maxlen=capacity)

    def td_error_to_prior(self, td_error, lengths):
        abs_td_error_sum = td_error.abs().sum(
            dim=1, keepdim=True).view(-1).detach().numpy()
        lengths_burn = [length - burn_in_length for length in lengths]

        prior_max = td_error.abs().max(dim=1, keepdim=True)[
            0].view(-1).detach().numpy()

        prior_mean = abs_td_error_sum / lengths_burn
        prior = eta * prior_max + (1 - eta) * prior_mean
        return prior

    def push(self, td_error, batch, lengths):
        prior = self.td_error_to_prior(td_error, lengths)

        for i in range(len(batch)):
            self.memory.append([Transition(batch.state[i], batch.next_state[i], batch.action[i],
                                           batch.reward[i], batch.mask[i], batch.step[i], batch.rnn_state[i]), lengths[i]])
            self.memory_probability.append(prior[i])

    def sample(self, batch_size):
        probability = np.array(self.memory_probability)
        probability = probability / probability.sum()

        indexes = np.random.choice(
            range(len(self.memory_probability)), batch_size, p=probability)
        episodes = [self.memory[idx][0] for idx in indexes]
        lengths = [self.memory[idx][1] for idx in indexes]

        batch_state, batch_next_state, batch_action, batch_reward, batch_mask, batch_step, batch_rnn_state = [], [], [], [], [], [], []
        for episode in episodes:
            batch_state.append(episode.state)
            batch_next_state.append(episode.next_state)
            batch_action.append(episode.action)
            batch_reward.append(episode.reward)
            batch_mask.append(episode.mask)
            batch_step.append(episode.step)
            batch_rnn_state.append(episode.rnn_state)

        return Transition(batch_state, batch_next_state, batch_action, batch_reward, batch_mask, batch_step, batch_rnn_state), indexes, lengths

    def update_prior(self, indexes, td_error, lengths):
        prior = self.td_error_to_prior(td_error, lengths)
        priors_idx = 0
        for idx in indexes:
            self.memory_probability[idx] = prior[priors_idx]
            priors_idx += 1

    def __len__(self):
        return len(self.memory)


class R2D2(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(R2D2, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        self.lstm = nn.LSTM(input_size=num_inputs,
                            hidden_size=16, batch_first=True)
        self.fc = nn.Linear(16, 128)
        self.fc_adv = nn.Linear(128, num_outputs)
        self.fc_val = nn.Linear(128, 1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)

    def forward(self, x, hidden=None):
        # shape of x: [batch_size, sequence_length, num_inputs]
        batch_size = x.size()[0]
        sequence_length = x.size()[1]
        out, hidden = self.lstm(x, hidden)

        out = F.relu(self.fc(out))
        adv = self.fc_adv(out)
        adv = adv.view(batch_size, sequence_length, self.num_outputs)
        val = self.fc_val(out)
        val = val.view(batch_size, sequence_length, 1)

        qvalue = val + (adv - adv.mean(dim=2, keepdim=True))

        return qvalue, hidden

    @classmethod
    def get_td_error(cls, online_net, target_net, batch, lengths):
        def slice_burn_in(item):
            return item[:, burn_in_length:, :]
        batch_size = torch.stack(batch.state).size()[0]
        states = torch.stack(batch.state).view(
            batch_size, sequence_length, online_net.num_inputs)
        next_states = torch.stack(batch.next_state).view(
            batch_size, sequence_length, online_net.num_inputs)
        actions = torch.stack(batch.action).view(
            batch_size, sequence_length, -1).long()
        rewards = torch.stack(batch.reward).view(
            batch_size, sequence_length, -1)
        masks = torch.stack(batch.mask).view(batch_size, sequence_length, -1)
        steps = torch.stack(batch.step).view(batch_size, sequence_length, -1)
        rnn_state = torch.stack(batch.rnn_state).view(
            batch_size, sequence_length, 2, -1)

        [h0, c0] = rnn_state[:, 0, :, :].transpose(0, 1)
        h0 = h0.unsqueeze(0).detach()
        c0 = c0.unsqueeze(0).detach()

        [h1, c1] = rnn_state[:, 1, :, :].transpose(0, 1)
        h1 = h1.unsqueeze(0).detach()
        c1 = c1.unsqueeze(0).detach()

        pred, _ = online_net(states, (h0, c0))
        next_pred, _ = target_net(next_states, (h1, c1))

        next_pred_online, _ = online_net(next_states, (h1, c1))

        pred = slice_burn_in(pred)
        next_pred = slice_burn_in(next_pred)
        actions = slice_burn_in(actions)
        rewards = slice_burn_in(rewards)
        masks = slice_burn_in(masks)
        steps = slice_burn_in(steps)
        next_pred_online = slice_burn_in(next_pred_online)

        pred = pred.gather(2, actions)

        _, next_pred_online_action = next_pred_online.max(2)

        target = rewards + masks * \
            pow(gamma, steps) * next_pred.gather(2,
                                                 next_pred_online_action.unsqueeze(2))

        td_error = pred - target.detach()

        for idx, length in enumerate(lengths):
            td_error[idx][length - burn_in_length:][:] = 0

        return td_error

    @classmethod
    def train_model(cls, online_net, target_net, optimizer, batch, lengths):
        td_error = cls.get_td_error(online_net, target_net, batch, lengths)

        loss = pow(td_error, 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss, td_error

    def get_action(self, state, hidden):
        state = state.unsqueeze(0).unsqueeze(0)

        qvalue, hidden = self.forward(state, hidden)

        _, action = torch.max(qvalue, 2)
        return action.numpy()[0][0], hidden


def get_action(state, target_net, epsilon, env, hidden):
    action, hidden = target_net.get_action(state, hidden)

    if np.random.rand() <= epsilon:
        return env.action_space.sample(), hidden
    else:
        return action, hidden


def update_target_model(online_net, target_net):
    # Target <- Net
    target_net.load_state_dict(online_net.state_dict())


def main():
    env = ReduceObs(gym.make(env_name))
    # env.seed(500)
    # torch.manual_seed(500)
    # np.random.seed(500)

    num_inputs = 2
    num_actions = env.action_space.n
    print('state size:', num_inputs)
    print('action size:', num_actions)

    online_net = R2D2(num_inputs, num_actions)
    target_net = R2D2(num_inputs, num_actions)
    update_target_model(online_net, target_net)

    optimizer = optim.Adam(online_net.parameters(), lr=lr)

    online_net.to(device)
    target_net.to(device)
    online_net.train()
    target_net.train()
    memory = Memory(replay_memory_capacity)
    epsilon = 1.0
    steps = 0
    loss = 0
    local_buffer = LocalBuffer()

    score_list = []

    for e in range(10000):
        done = False

        score = 0
        state = env.reset()
        state = torch.Tensor(state).to(device)

        hidden = (torch.Tensor().new_zeros(1, 1, 16),
                  torch.Tensor().new_zeros(1, 1, 16))

        while not done:
            steps += 1

            action, new_hidden = get_action(state, target_net, epsilon, env, hidden)

            next_state, reward, done, _ = env.step(action)
            next_state = torch.Tensor(next_state).to(device)

            mask = 0 if done else 1
            reward = reward if not done or score == 499 else -1

            local_buffer.push(state, next_state, action, reward, mask, hidden)
            hidden = new_hidden
            if len(local_buffer.memory) == local_mini_batch:
                batch, lengths = local_buffer.sample()
                td_error = R2D2.get_td_error(
                    online_net, target_net, batch, lengths)
                memory.push(td_error, batch, lengths)

            score += reward
            state = next_state

            if steps > initial_exploration and len(memory) > batch_size:
                epsilon -= 0.00005
                epsilon = max(epsilon, 0.01)

                batch, indexes, lengths = memory.sample(batch_size)
                loss, td_error = R2D2.train_model(
                    online_net, target_net, optimizer, batch, lengths)

                memory.update_prior(indexes, td_error, lengths)

                if steps % update_target == 0:
                    update_target_model(online_net, target_net)

        score_list.append(score)
        if e % log_interval == 0:
            print('{} episode | score: {:.2f} | epsilon: {:.2f}'.format(
                e, score, epsilon))

        if all(np.array(score_list[-20:]) >= goal_score):
            print('Solved in {} episodes'.format(e - 20))
            break

    plt.plot(score_list)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()


if __name__ == '__main__':
    main()
