import gym
import numpy as np
import collections
import cv2

class NoopENV(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        super().__init__(env)
        self.noop_max = 30
        assert env.unwrapped.get_action_meanings()[0] == "NOOP"
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[1] == "FIRE"
        self.fire_action = 1
        
    def step(self, action):
        return self.env.step(action)
    
    def reset(self):
        obs = self.env.reset()
        obs, _, _, _ =  self.env.step(self.fire_action)
        noops = self.env.unwrapped.np_random.randint(0, self.noop_max + 1)
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                self.env.reset()
                obs, _, _, _ = self.env.step(self.fire_action)
        return obs

class SkipENV(gym.Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self.skip = skip
        self.obs = collections.deque(maxlen=3)
        
    def step(self, action):
        rewards = 0
        for _ in range(self.skip):
            obs, reward, done, info = self.env.step(action)
            self.obs.append(obs)
            rewards += reward
            if done:
                break
        obs = np.max(np.stack(self.obs), axis=0)
        return obs, rewards, done, info

    def reset(self):
        self.obs.clear()
        obs = self.env.reset()
        self.obs.append(obs)
        return obs
    
class ResizeENV(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), 
                                                dtype=np.uint8)
        
    def observation(self, obs):
        obs = cv2.cvtColor(obs,  cv2.COLOR_RGB2GRAY)
        obs = obs[35:-15]
        obs = cv2.resize(obs, (84,84), interpolation=cv2.INTER_AREA)
        return obs.reshape(84, 84, 1)
    
class ToFloatENV(gym.ObservationWrapper):
    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0
    
class ToPytorchDimENV(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        old_shape = env.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, 
                        shape=(old_shape[-1], old_shape[0], old_shape[1]), dtype=np.float32)
        
    def observation(self, obs):
        return np.moveaxis(obs, 2, 0)
    
class BufferENV(gym.ObservationWrapper):
    def __init__(self, env, n_step=4, dtype=np.float32):
        super().__init__(env)
        self.dtype = dtype
        old_space = env.observation_space
        self.observation_space = gym.spaces.Box(old_space.low.repeat(n_step, axis=0),
                                               old_space.high.repeat(n_step, axis=0),
                                               dtype=dtype)
    def reset(self):
        self.buffer = np.zeros_like(self.observation_space.low, dtype=self.dtype)
        return self.observation(self.env.reset())

    def observation(self, obs):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = obs
        return self.buffer
    
def make_env(ENV):
    def _x():
        env = gym.make(ENV)
        env = SkipENV(env)
        env = NoopENV(env)
        env = ResizeENV(env)
        env = ToFloatENV(env)
        env = ToPytorchDimENV(env)
        return BufferENV(env)
    return _x