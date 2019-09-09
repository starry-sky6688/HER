import torch
import os
import numpy as np
from network.actor_critic import Actor, Critic


class DDPG:
    def __init__(self, args):
        self.args = args
        
        # create the network
        self.actor_network = Actor(args)
        self.critic_network = Critic(args)

        # build up the target network
        self.actor_target_network = Actor(args)
        self.critic_target_network = Critic(args)

        # load the weights into the target networks
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())
        self.critic_target_network.load_state_dict(self.critic_network.state_dict())

        # create the optimizer
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.args.lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=self.args.lr_critic)

        self.train_step = 0

    # soft update
    def _soft_update_target_network(self):
        for target_param, param in zip(self.actor_target_network.parameters(), self.actor_network.parameters()):
            target_param.data.copy_((1 - self.args.tau) * param.data + self.args.tau * target_param.data)

        for target_param, param in zip(self.critic_target_network.parameters(), self.critic_network.parameters()):
            target_param.data.copy_((1 - self.args.tau) * param.data + self.args.tau * target_param.data)

    def _get_inputs(self, trans):
        o, o_next, g = trans['o'], trans['o_next'], trans['g']
        inputs = np.concatenate([o, g], axis=1)
        inputs_next = np.concatenate([o_next, g], axis=1)  # 同一个episode中的g是不变的

        # transfer into the tensor
        inputs = torch.tensor(inputs, dtype=torch.float32)
        inputs_next = torch.tensor(inputs_next, dtype=torch.float32)
        actions = torch.tensor(trans['u'], dtype=torch.float32)

        return inputs, inputs_next, actions

    # update the AC network
    def train(self, transitions):  # 传入的transitions里的数据已经经过clip和正则化
        inputs, inputs_next, actions = self._get_inputs(transitions)
        r = torch.tensor(transitions['r'], dtype=torch.float32)
        if self.args.cuda:
            inputs = inputs.cuda()
            inputs_next = inputs_next.cuda()
            actions = actions.cuda()
            r = r.cuda()

        # calculate the target Q value function
        with torch.no_grad():
            # 得到下一个状态对应的动作
            actions_next = self.actor_target_network(inputs_next)
            q_next = self.critic_target_network(inputs_next, actions_next).detach()  # target网络的参数不通过loss函数来训练

            target_q = (r.unsqueeze(1).expand((-1, 1)) + self.args.gamma * q_next).detach()
            # clip the q value
            clip_return = 1 / (1 - self.args.gamma)
            target_q = torch.clamp(target_q, -clip_return, 0)

        # critic loss
        q_value = self.critic_network(inputs, actions)
        critic_loss = (target_q - q_value).pow(2).mean()

        # the actor loss
        actions_real = self.actor_network(inputs)
        actor_loss = -self.critic_network(inputs, actions_real).mean()
        actor_loss += self.args.action_l2 * (actions_real / self.args.action_max).pow(2).mean()
        # print('critic_loss is {}, actor_loss is {}'.format(critic_loss, actor_loss))
        # update the network
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        self._soft_update_target_network()
        self.train_step += 1
        if self.train_step % self.args.save_interval == 0:
            self.save_model(self.train_step)

    def save_model(self, train_step):
        num = str(train_step // self.args.save_interval)
        model_path = os.path.join(self.args.save_dir, self.args.env_name)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.actor_network.state_dict(), model_path + '/' + num + '_actor_params.pkl')
        torch.save(self.critic_network.state_dict(),  model_path + '/' + num + '_critic_params.pkl')





