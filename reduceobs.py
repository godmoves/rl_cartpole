"""
Simple RL tasks based on the CartPole env in OpenAI Gym

@author: Inspir.ai
"""


import gym

class ReduceObs(gym.ObservationWrapper):
    def __init__(self, env):
        """
        Restrict to observe only part of the original observations, i.e.,
        Observation:
            Type: Box(2)
            Num	Observation                 Min         Max
            0	Cart Position             -4.8            4.8
            1	Pole Angle                 -24°           24°
        """
        gym.ObservationWrapper.__init__(self, env)
        old_obs_space = env.observation_space
        self.observation_space = gym.spaces.Box(low=old_obs_space.low[[0, 2]],
                                                high=old_obs_space.high[[0, 2]],
                                                dtype=old_obs_space.dtype)

    def observation(self, obs):
        return obs[[0, 2]]


if __name__ == '__main__':
    # Test the env of task 1
    env = gym.make('CartPole-v1')
    print('The observation shape of the env for task 1 is: {}'.format(env.observation_space.shape))

    episode_count = 10
    max_steps = 200
    for i in range(episode_count):
        observation = env.reset()
        ep_reward = 0
        for t in range(max_steps):
            env.render()

            action = env.action_space.sample()  # your agent here (this takes random actions)

            observation, reward, done, info = env.step(action)
            ep_reward += reward
            if done:
                print("Episode {} finished with reward {}".format(i+1, ep_reward))
                break
    env.close()


    # Test the env of task 2
    env = ReduceObs(gym.make('CartPole-v1'))   # only Cart Position and Pole Angle can be observed
    print('The observation shape of the env for task 2 is: {}'.format(env.observation_space.shape))

    episode_count = 10
    max_steps = 200
    for i in range(episode_count):
        observation = env.reset()
        ep_reward = 0
        for t in range(max_steps):
            env.render()

            action = env.action_space.sample()  # your agent here (this takes random actions)

            observation, reward, done, info = env.step(action)
            ep_reward += reward
            if done:
                print("Episode {} finished with reward {}".format(i + 1, ep_reward))
                break
    env.close()