import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

epochs = 100

class DQN:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim =self.state_size, activation ='relu'))
        model.add(Dense(24, activation ='relu'))
        model.add(Dense(self.action_size, activation ='linear'))
        model.compile(loss ='mean_squared_error', optimizer =Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        action_values = self.model.predict(state)
        return np.argmax(action_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
            final_target = self.model.predict(state)
            final_target[0][action] = target
            self.model.fit(state, final_target, epochs = 1, verbose = 0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


if __name__ == "__main__":
    game = gym.make('CartPole-v1')
    state_size = game.observation_space.shape[0]
    action_size = game.action_space.n
    agent = DQN(state_size, action_size)
    done = False
    batch_size = 30

    for x in range(epochs):
        state = game.reset()
        state = np.reshape(state, [1, 4])
        for time in range(500):
            game.render()
            action = agent.act(state)
            (next_state, reward, done, _)= game.step(action)
            next_state = np.reshape(next_state, [1, 4])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print("epoch: {}/{}, score: {}, epsilon: {:.2}".format(x, epochs, time, agent.epsilon))
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
