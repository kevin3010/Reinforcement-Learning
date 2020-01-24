import gym
import numpy as np
import sys

from collections import defaultdict

def create_uniform_grid(low, high, bins=(10, 10)):
  grid = [np.linspace(low[dim], high[dim], bins[dim] + 1)[1:-1] for dim in range(len(bins))]
  return grid

def discretize(sample, grid):
  return list(int(np.digitize(s, g)) for s, g in zip(sample, grid))

def make_epsilon_greedy_policy(Q, epsilon, nA):
   
  def policy_fn(observation):
    do_exploration = np.random.uniform(0, 1) < epsilon
    if do_exploration:
        # Pick a random action
        action = np.random.randint(0, nA)
    else:
        # Pick the best action from Q table
        action = np.argmax(Q[observation])
    return action 

  return policy_fn



def mc_control_epsilon_greedy(env, num_episodes, discount_factor=0.99,epsilon=1.0,min_epsilon=0.01,epsilon_decay_rate=0.9995, learning_rate=0.05):

  Q = defaultdict(lambda: np.zeros(env.action_space.n))

  
  scores = []
  max_avg_score = -np.inf
  for i in range(1,num_episodes+1):
    # if i % 100 == 0:
    #   print("\rEpisode {}/{}.".format(i, num_episodes), end="")
    #   sys.stdout.flush()

    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)
    total_reward=0.0
    epsilon *= epsilon_decay_rate
    epsilon = max(epsilon,min_epsilon)

    state = env.reset()
    state = discretize(state,state_grid)
    state = (state[0],state[1])

    
    
    done=False
    # print("Episode:"+str(i))
    while not done:
      action = policy(state)
      next_state, reward, done, _ = env.step(action)
      total_reward += reward

      next_state = discretize(next_state,state_grid)
      next_state = (next_state[0],next_state[1])

      best_next_action = np.argmax(Q[next_state])    
      td_target = reward + discount_factor * Q[next_state][best_next_action]
      td_delta = td_target - Q[state][action]
      Q[state][action] += learning_rate * td_delta

      # print("Current state:"+str(state)+" "+str(Q[state])+"===>"+"next state:" + str(next_state)+" "+str(Q[next_state]))
      state = next_state
      
      # Print episode stats
        
      

    scores.append(total_reward)
    if len(scores) > 100 and i % 100 == 0:
      avg_score = np.mean(scores[-100:])
      if avg_score > max_avg_score:
          max_avg_score = avg_score
    if i % 250 == 0:
      
        for j in range(1):
            state = env.reset()
            score = 0
      
            for t in range(200):
                state = discretize(state,state_grid)
                state = (state[0],state[1])
                action = np.argmax(Q[state])
                env.render()
                state, reward, done, _ = env.step(action)
                score += reward
                if done:
                    break 
            
        print("\rEpisode {}/{} | Max Average Score: {} | final score: {}".format(i, num_episodes, max_avg_score,score), end="")
        sys.stdout.flush()
    

  return Q



env = gym.make('MountainCar-v0')
state_grid = create_uniform_grid(env.observation_space.low, env.observation_space.high, bins=(20, 20))
Q = mc_control_epsilon_greedy(env,num_episodes=20000)

for i in Q.items():
  print(i)
  
env.close()
  

