import numpy as np
import gym 
import random
from IPython.display import clear_output
import time

env = gym.make("FrozenLake-v0")
action_space_size = env.action_space.n
state_space_size = env.observation_space.n
print(state_space_size)
print(action_space_size)
# time.sleep(10)

q_table = np.zeros((state_space_size,action_space_size))
print(q_table)
#* Q-learning parameter
num_episodes = 10000
max_steps = 100

lr = 0.1
gamma = 0.99
exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate = 0.01

rewards = []

'''
Q learning
'''
for episode in range(num_episodes):
    state = env.reset()
    # print(state)
    # time.sleep(3)
    done = False
    reward_current = 0
    for step in range(max_steps):
        rand = random.uniform(0,1) #uniform?
        # print(f'rand no {rand}')
        # print(f'exp rate{exploration_rate}')
        if rand > exploration_rate :
            action = np.argmax(q_table[state,:])
            # print(f'we are action is {action}')
            # time.sleep(5)
        else:
            action = env.action_space.sample()
            # print(f' action is {action}')
        new_state, reward, done, _ = env.step(action)
        # print(f'THE REWARD IS {reward}')

        q_table[state,action] = (1-lr)*q_table[state,action] +\
             lr*(reward + gamma*np.max(q_table[new_state,:]))
        # print(q_table)
        state = new_state
        # print(f"new state {state}")
        reward_current+=reward
        # print(reward_current)
        if done :
            break
        
        # Exploration rate decay
        exploration_rate = min_exploration_rate + \
            (max_exploration_rate - min_exploration_rate) * \
            np.exp(-exploration_decay_rate*episode)
        print(f'episode {episode} : reward {reward_current}')
    rewards.append(reward_current)


rewards_per_thosand_episodes = np.split(np.array(rewards)\
    ,num_episodes/1000)
count = 1000

print("********Average reward per thousand episodes********\n")
for r in rewards_per_thosand_episodes:
    print(count, ": ", str(sum(r/1000)))
    count += 1000
print("\n\n********Q-table********\n")
print(q_table)





