from matplotlib import pyplot as plt
import numpy as np, random, sys

def env_factory(noise_sd, bandit_val):
    def env(action):
        return bandit_val[np.arange(2000), action] + np.random.normal(scale=noise_sd, size=(2000))
    return env

def epsilon_greedy(epsilon, env, episodes=1000):
    num_experiments = np.zeros(shape=(2000, 10), dtype=np.int)
    approximate_val = np.zeros(shape=(2000, 10), dtype=np.float)
    rewards = []
    actions = []

    for x in range(10):
        approximate_val[ np.arange(2000), x ] = approximate_val[ np.arange(2000), x ] * num_experiments[ np.arange(2000), x ] + env(np.repeat(x, 2000))
        num_experiments[ np.arange(2000), x ] += 1

    for _ in range(episodes):
        if random.random() > epsilon: action = np.argmax(approximate_val, axis=1)
        else: action = np.random.randint(10, size=2000)

        reward = env(action)
        approximate_val[ np.arange(2000), action ] = (approximate_val[ np.arange(2000), action ] * num_experiments[ np.arange(2000), action ] + reward) / (num_experiments[ np.arange(2000), action ] + 1)
        num_experiments[ np.arange(2000), action ] += 1

        rewards.append(reward)
        actions.append(action)

    return np.array(actions), np.array(rewards)

def greedy():
    return epsilon_greedy(0)

def softmax(temperature, env, episodes=1000):
    num_experiments = np.zeros(shape=(2000, 10), dtype=np.int)
    approximate_val = np.zeros(shape=(2000, 10), dtype=np.float)
    rewards = []
    actions = []

    for x in range(10):
        approximate_val[ np.arange(2000), x ] = approximate_val[ np.arange(2000), x ] * num_experiments[ np.arange(2000), x ] + env(np.repeat(x, 2000))
        num_experiments[ np.arange(2000), x ] += 1

    for _ in range(episodes):
        approximate_val_exp = np.exp(approximate_val / temperature)
        probs = ( approximate_val_exp.T / approximate_val_exp.sum(axis=1) ).T
        cumulative_probs = np.zeros_like(probs)
        cumulative_probs[:, 0] = probs[:, 0]
        for t in range(1, 10):
            cumulative_probs[:, t] = cumulative_probs[:, t-1] + probs[:, t]
        action = np.sum(np.repeat(np.random.random(2000), 10).reshape(2000, 10) > cumulative_probs, axis=1)

        reward = env(action)
        approximate_val[ np.arange(2000), action ] = (approximate_val[ np.arange(2000), action ] * num_experiments[ np.arange(2000), action ] + reward) / (num_experiments[ np.arange(2000), action ] + 1)
        num_experiments[ np.arange(2000), action ] += 1

        rewards.append(reward)
        actions.append(action)

    return np.array(actions), np.array(rewards), cumulative_probs, probs

if __name__ == '__main__':
    if len(sys.argv) != 6:
        print("Usage: %s %s %s %s %s %s" %(sys.argv[0], "<bandit-sd>", "<bandit-noise>", "<method>", "<param>", "<episodes>"), file=sys.stderr)
        sys.exit(1)

    bandit_val = 2 * (np.random.random(size=(2000, 10)) - 0.5)
    env = env_factory(eval(sys.argv[2]), bandit_val)
    
    method = sys.argv[3]
    episodes = eval(sys.argv[5])
    if method == 'epsilon_greedy':
        rewards = epsilon_greedy(eval(sys.argv[4]), env, episodes)
    elif method == 'softmax':
        rewards = softmax(eval(sys.argv[4]), env, episodes)
    else: 
        print("Invalid method: use either epsilon_greedy or softmax", file=sys.stderr)
        sys.exit(1)

    plt.plot(rewards[1].mean(axis=1), '1')
    plt.show()
