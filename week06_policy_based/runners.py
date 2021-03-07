""" RL env runner """
from collections import defaultdict
import numpy as np


class EnvRunner:
  """ Reinforcement learning runner in an environment with given policy """
  def __init__(self, env, policy, nsteps, transforms=None, step_var=None):
    self.env = env
    self.policy = policy
    self.nsteps = nsteps
    self.transforms = transforms or []
    self.step_var = step_var if step_var is not None else 0
    self.state = {"latest_observation": self.env.reset()}
    #print('status len :{}'.format(len(self.state)))
  @property
  def nenvs(self):
    """ Returns number of batched envs or `None` if env is not batched """
    return getattr(self.env.unwrapped, "nenvs", None)

  def reset(self):
    """ Resets env and runner states. """
    self.state["latest_observation"] = self.env.reset()
    self.policy.reset()

  def get_next(self):
    """ Runs the agent in the environment.  """
    trajectory = defaultdict(list, {"actions": []})
    observations = []
    rewards = []
    resets = []
    self.state["env_steps"] = self.nsteps

    for i in range(self.nsteps):  # 行5步
      observations.append(self.state["latest_observation"])
      act = self.policy.act(self.state["latest_observation"])
      if "actions" not in act:
        raise ValueError("result of policy.act must contain 'actions' "
                         f"but has keys {list(act.keys())}")
      for key, val in act.items():   # 每步用每次策略后1个动作；6个log；6个logit；1个值增加轨集合中的值
        trajectory[key].append(val)

      obs, rew, done, _ = self.env.step(trajectory["actions"][-1]) # 在8个环境中执行策略的动作
      self.state["latest_observation"] = obs     # 更新状态值[8=envs,48,48,4=帧]
      rewards.append(rew)                        # 得到8个r  [8,1]
      resets.append(done)                        # 得到8个done的T or F [8,1]
      self.step_var += self.nenvs or 1           

      # Only reset if the env is not batched. Batched envs should auto-reset.
      if not self.nenvs and np.all(done):
        self.state["env_steps"] = i + 1
        self.state["latest_observation"] = self.env.reset()

    trajectory.update(observations=observations, rewards=rewards, resets=resets)  
    # 轨迹放入新的逐步增加的 observations, rewards, done keys 和值
    trajectory["state"] = self.state

    for transform in self.transforms:   # 把列表变形，收缩维度
      transform(trajectory)
    return trajectory
