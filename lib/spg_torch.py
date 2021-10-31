import ptan
import numpy as np
import torch
import torch.nn as nn

HID_SIZE = 256

class SPG(nn.Module):
    def __init__(self, obs_size, act_size):
        super(SPG, self).__init__()

        self.base = nn.Sequential(
            nn.Linear(obs_size, HID_SIZE),
            nn.ReLU(),
        )
        self.mu = nn.Sequential(
            nn.Linear(HID_SIZE, act_size),
            nn.Tanh(),
        )
        self.var = nn.Sequential(
            nn.Linear(HID_SIZE, act_size),
            nn.Softplus(),
        )
        self.value = nn.Linear(HID_SIZE, 1)

    def forward(self, x):
        base_out = self.base(x)
        return self.mu(base_out), self.var(base_out), \
               self.value(base_out)

class AgentA2C(ptan.agent.BaseAgent):
    '''
    Extended stochastic A2C agent 
    '''
    def __init__(self, net, device="cpu", explore=True):
        self.net = net
        self.device = device
        self.explore = explore

    def __call__(self, states, agent_states):
        states_v = ptan.agent.float32_preprocessor(states)
        states_v = states_v.to(self.device)

        mu_s, var_v, Qvalue = self.net(states_v)
        mu = mu_s.data.cpu().numpy()
        sigma = torch.sqrt(var_v).data.cpu().numpy()
        actions = np.random.normal(mu, sigma)
        actions = np.clip(actions, -1, 1)
        return actions, agent_states


class SPGActor(nn.Module):
    '''
    SPG Actor net
    '''
    def __init__(self, obs_size, act_size):
        super(SPGActor, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_size, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, act_size),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)

class SPGCritic(nn.Module):
    '''
    Q-value Critic net
    '''
    def __init__(self, obs_size, act_size):
        super(SPGCritic, self).__init__()

        self.obs_net = nn.Sequential(
            nn.Linear(obs_size, 400),
            nn.ReLU(),
        )

        self.out_net = nn.Sequential(
            nn.Linear(400 + act_size, 300),
            nn.ReLU(),
            nn.Linear(300, 1)
        )

    def forward(self, x, a):
        obs = self.obs_net(x)
        return self.out_net(torch.cat([obs, a], dim=1))

class Critic(nn.Module):
    '''
    Double Q-value Critic net
    '''
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.obs_net = nn.Sequential(
            nn.Linear(state_dim, HID_SIZE),
            nn.ReLU()
        )

        self.Q1 = nn.Sequential(
            nn.Linear(HID_SIZE + action_dim, HID_SIZE),
            nn.ReLU(),
            nn.Linear(HID_SIZE, 1)
        )

        self.Q2 = nn.Sequential(
            nn.Linear(HID_SIZE + action_dim, HID_SIZE),
            nn.ReLU(),
            nn.Linear(HID_SIZE, 1)
        )
    def forward(self, state, action):
        obs = self.obs_net(state)
        return self.Q1(torch.cat([obs, action], dim=1)), self.Q2(torch.cat([obs, action], dim=1))

class AgentDDPG(ptan.agent.BaseAgent):
    """
    Agent implementing DDPG
    """
    def __init__(self, net, device="cpu", epsilon=0.3, explore=True):
        self.net = net
        self.device = device
        self.epsilon = epsilon
        self.explore = explore

    def __call__(self, states, agent_states):
        states_v = ptan.agent.float32_preprocessor(states)
        states_v = states_v.to(self.device)
        mu_s = self.net(states_v)
        actions = mu_s.data.cpu().numpy()

        actions += self.epsilon * np.random.normal(
            size=actions.shape)
        actions = np.clip(actions, -1, 1)
        return actions, agent_states

class AgentSPG(ptan.agent.BaseAgent):
    """
    Agent implementing SPG
    """
    def __init__(self, net, device="cpu", epsilon=0.3, explore=True):
        self.net = net
        self.device = device
        self.epsilon = epsilon
        self.explore = explore

    def __call__(self, states, agent_states):
        states_v = ptan.agent.float32_preprocessor(states)
        states_v = states_v.to(self.device)
        mu_s = self.net(states_v)
        actions = mu_s.data.cpu().numpy()
        actions = np.clip(actions, -1, 1)

        return actions, agent_states

 
class SafeLayer(nn.Module):
    def __init__(self, obs_size, act_size):
        super(SafeLayer, self).__init__()

        self.obs_net = nn.Sequential(
            nn.Linear(obs_size, 400),
            nn.ReLU(),
        )
        self.out_net = nn.Sequential(
            nn.Linear(400 + act_size, 300),
            nn.ReLU(),
            nn.Linear(300, 1)
            )
        self.squash = nn.Sigmoid()
    def forward(self, x, a):
        obs = self.obs_net(x)
        value = self.out_net(torch.cat([obs, a], dim=1))
        #return F.log_softmax(value, dim=1)
        return self.squash(value)