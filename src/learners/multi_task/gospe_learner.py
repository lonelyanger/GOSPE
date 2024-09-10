
import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.multi_task.vdn import VDNMixer
from modules.mixers.multi_task.qattn import QMixer as MTAttnQMixer
import torch as th
from torch.optim import RMSprop, Adam
import torch.nn.functional as F


from gwd import gromov_wasserstein_discrepancy2


def graphloss(s_gph,k_gph):
    def regraph(graph):
        graph=graph.reshape(graph.shape[0],graph.shape[1],-1)
        graph=graph.reshape(-1,graph.shape[-1])
        return graph
    s_gph=regraph(s_gph)
    s_dist=th.cdist(s_gph,s_gph)
    
    k=2
    val,idx=th.sort(s_dist)
    val_k=val[:,k].reshape(-1,1).repeat(1,s_dist.shape[1])
    val_bool=th.where(s_dist>=val_k,0,1)
    val_bool=th.where((val_bool+val_bool.t())>0.1,1,0)

    k_gph=regraph(k_gph)
    k_dist=th.cdist(k_gph,k_gph)
    gloss=th.sum(val_bool*k_dist)/s_gph.shape[0]

    return gloss

def GWDloss(s_gph1,s_gph2,k_gph1,k_gph2,ot_hyperparams):
    # s_gph  bs*N*N
    def regraph(graph):
        graph=graph.reshape(-1,graph.shape[-2],graph.shape[-1])
        return graph
    trans,s_dist=gromov_wasserstein_discrepancy2(regraph(s_gph1), regraph(s_gph2), ot_hyperparams)
    k=2
    val,idx=th.sort(s_dist)
    val_k=val[:,k].reshape(-1,1).repeat(1,s_dist.shape[1])
    val_bool=th.where(s_dist>=val_k,0,1)
    # val_bool=th.where((val_bool+val_bool.t())>0.1,1,0)
    k_gph1=regraph(k_gph1)
    k_gph2=regraph(k_gph2)
    T,B,N,M=k_gph1.shape[0],k_gph2.shape[0],k_gph1.shape[1],k_gph2.shape[1]
    trans=trans[:,:,0:N,0:M]
    res=th.stack([th.matmul(t,k_gph2) for t in trans ],dim=0)
    res_f=[]
    for i in range(len(k_gph1)):
        res_f.append(th.cdist(k_gph1[i].reshape(1,-1),res[i].reshape(B,-1)))
    k_dist=th.cat(res_f,dim=0)

    gloss=th.sum(val_bool*k_dist)/(k_dist.shape[0]*k_dist.shape[1])
    return gloss


class GOSPELearner:
    def __init__(self, mac, logger, main_args):
        self.main_args = main_args
        self.mac = mac
        self.logger = logger

        # get some attributes from mac
        self.task2args = mac.task2args
        self.task2n_agents = mac.task2n_agents
        self.surrogate_decomposer = mac.surrogate_decomposer
        self.task2decomposer = mac.task2decomposer

        self.params = list(mac.parameters())

        self.last_target_update_episode = 0
        
        self.ot_hyperparams = {
        "ot_method": "proximal",
        "loss_type": "L2",
        "inner_iteration": 10,
        "outer_iteration": 10,
        "iter_bound": 1e-3,
        "sk_bound": 1e-3,
        "opt_trans":True,
        "GetTrans":True,
        "seqidx":th.inf
        }

        self.mixer = None
        if main_args.mixer is not None:
            if main_args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif main_args.mixer == "mt_qattn":
                self.mixer = MTAttnQMixer(self.surrogate_decomposer, main_args)
            else:
                raise ValueError(f"Mixer {main_args.mixer} not recognised.")
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        self._reset_optimizer()

        self.target_mac = copy.deepcopy(mac)

        # define attributes for each specific task
        self.task2train_info, self.task2encoder_params, self.task2encoder_optimiser = {}, {}, {}
        for task in self.task2args:
            task_args = self.task2args[task]
            self.task2train_info[task] = {}
            self.task2train_info[task]["log_stats_t"] = -task_args.learner_log_interval - 1

        self.c = main_args.c_step
        self.skill_dim = main_args.skill_dim
        self.beta = main_args.beta
        self.phi = main_args.coef_dist

        self.pretrain_steps = 0
        self.training_steps = 0

    def _reset_optimizer(self):
        if self.main_args.optim_type.lower() == "rmsprop":
            self.pre_optimiser = RMSprop(params=self.params, lr=self.main_args.lr, alpha=self.main_args.optim_alpha, eps=self.main_args.optim_eps, weight_decay=self.main_args.weight_decay)
            self.optimiser = RMSprop(params=self.params, lr=self.main_args.lr, alpha=self.main_args.optim_alpha, eps=self.main_args.optim_eps, weight_decay=self.main_args.weight_decay)
        elif self.main_args.optim_type.lower() == "adam":
            self.pre_optimiser = Adam(params=self.params, lr=self.main_args.lr, weight_decay=self.main_args.weight_decay)
            self.optimiser = Adam(params=self.params, lr=self.main_args.critic_lr, weight_decay=self.main_args.weight_decay)
        else:
            raise ValueError("Invalid optimiser type", self.main_args.optim_type)
        self.pre_optimiser.zero_grad()
        self.optimiser.zero_grad()

    def zero_grad(self):
        self.pre_optimiser.zero_grad()
        self.optimiser.zero_grad()

    def update(self, pretrain=True):
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.main_args.grad_norm_clip)
        if pretrain:
            self.pre_optimiser.step()
            self.pre_optimiser.zero_grad()
        else:
            self.optimiser.step()
            self.optimiser.zero_grad()

    def train_vae(self, batch: EpisodeBatch, t_env: int, episode_num: int, task: str):
        rewards = batch["reward"][:, :]
        actions = batch["actions"][:, :]
        terminated = batch["terminated"][:, :].float()
        mask = batch["filled"][:, :].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
    


        # Calculate estimated Q-Values
        mac_out = []
        self.mac.init_hidden(batch.batch_size, task)
        GMVAEloss=0
        skill_prob=[]
        for t in range(batch.max_seq_length):
            agent_outs,loss = self.mac.forward_skill(batch, t=t, task=task, actions=actions[:, t, :])
            # agent_outs is entity_dim
            mac_out.append(agent_outs)
            GMVAEloss=GMVAEloss+loss['total_loss']
            skill_prob.append(loss['c'])
        enc_loss=GMVAEloss
        mac_out = th.stack(mac_out, dim=1)  # Concat over time   bs,t,nagent,dim
        skill_prob=th.stack(skill_prob, dim=1)

       
        
        seq_skill_input=mac_out
        dec_loss = 0.   ### batch time agent skill
        self.mac.init_hidden(batch.batch_size, task)
        for t in range(batch.max_seq_length-self.c):
            seq_action_output = self.mac.forward_seq_action(batch, seq_skill_input[:, t, :, :], t, task=task)
            b, c, n, a = seq_action_output.size()
            dec_loss += (F.cross_entropy(seq_action_output.reshape(-1, a), actions[:, t:t + self.c].squeeze(-1).reshape(-1), reduction="sum") / mask[:, t:t + self.c].sum()) / n

        vae_loss = dec_loss / (batch.max_seq_length - self.c) + self.main_args.beta * enc_loss
        loss = vae_loss
      
        loss.backward()

        if t_env - self.task2train_info[task]["log_stats_t"] >= self.task2args[task].learner_log_interval:
            # self.logger.log_stat(f"pretrain/{task}/grad_norm", grad_norm.item(), t_env)
            self.logger.log_stat(f"pretrain/{task}/vae_loss", vae_loss.item(), t_env)
            # self.logger.log_stat(f"pretrain/{task}/dist_loss", dist_loss.item(), t_env)
            self.logger.log_stat(f"pretrain/{task}/kl_loss", enc_loss.item(), t_env)
            self.logger.log_stat(f"pretrain/{task}/dec_loss", dec_loss.item(), t_env)
            # self.logger.log_stat(f"pretrain/{task}/graph_loss", gloss.item(), t_env)

            for i in range(self.skill_dim):
                skill_dist = skill_prob.reshape(-1, self.skill_dim).mean(dim=0)
                self.logger.log_stat(f"pretrain/{task}/skill_class{i+1}", skill_dist[i].item(), t_env)
            
            self.task2train_info[task]["log_stats_t"] = t_env
            
    def test_vae(self, batch: EpisodeBatch, t_env: int, episode_num: int, task: str):
        return 
    
    def train_policy(self, batch: EpisodeBatch, t_env: int, episode_num: int, task: str):
        # Get the relevant quantities
        rewards = batch["reward"][:, :]
        actions = batch["actions"][:, :]
        terminated = batch["terminated"][:, :].float()
        mask = batch["filled"][:, :].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        # avail_actions = batch["avail_actions"]
        state_graph=batch["state_graph"]

        #### encode action
        with th.no_grad():
            new_actions = []
            z_out_state=[]
            w_out_state=[]
            self.mac.init_hidden(batch.batch_size, task)
            for t in range(batch.max_seq_length):
                z_outs,output = self.mac.forward_skill(batch, t=t, task=task, actions=actions[:, t, :])
     
                z_out_state.append(z_outs)
                label_c=output['c']
                w=output['w']
                w_out_state.append(w)
                label_action=label_c.max(dim=-1)[1].unsqueeze(-1)
                new_actions.append(label_action)
    
            actions = th.stack(new_actions, dim=1)
            z_out_state = th.stack(z_out_state, dim=1)
            w_out_state = th.stack(w_out_state, dim=1)
        ####

        #### representation
        mac_out_obs = []
        mac_out = []
        z_out_obs=[]
        w_out_obs=[]
        self.mac.init_hidden(batch.batch_size, task)
        for t in range(batch.max_seq_length):
            z_outs, agent_outs, pri_outs, w = self.mac.forward_both(batch, t=t, task=task,dist_skill=new_actions[t])

            z_out_obs.append(z_outs)
            mac_out.append(agent_outs)
            mac_out_obs.append(pri_outs)
            w_out_obs.append(w)
        z_out_obs=th.stack(z_out_obs, dim=1)
        w_out_obs=th.stack(w_out_obs, dim=1)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time
        _, _, n_agents, _ = mac_out.size()
        mac_out_obs = th.stack(mac_out_obs, dim=1)  # Concat over time
        dist_loss=F.mse_loss(z_out_obs,z_out_state) / mask.sum()/ n_agents

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :], dim=3, index=actions[:, :]).squeeze(3)  # Remove the last dim

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size, task)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward_qvalue(batch, t=t, task=task)
            target_mac_out.append(target_agent_outs)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out, dim=1)  # Concat across time


        # Max over target Q-Values
        if self.main_args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            # mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, :].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)

            cons_max_qvals = th.gather(mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        # Mix
        bs, seq_len = chosen_action_qvals.size(0), chosen_action_qvals.size(1)
     
        if self.mixer is not None:
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :],
                                             self.task2decomposer[task])
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, :],
                                                 self.task2decomposer[task])

            cons_max_qvals = self.mixer(cons_max_qvals, batch["state"][:, :],
                                        self.task2decomposer[task])

        # Calculate c-step Q-Learning targets
        cs_rewards = batch["reward"]
        for i in range(1, self.c):
            cs_rewards[:, :-self.c] += rewards[:, i:-(self.c - i)]
        # cs_rewards /= self.c

        targets = cs_rewards[:, :-self.c] + self.main_args.gamma * (1 - terminated[:, self.c - 1:-1]) * target_max_qvals[:, self.c:]

        # Td-error
        td_error = (chosen_action_qvals[:, :-self.c] - targets.detach())


        # mask = mask[:, :].expand_as(cons_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask[:, :-self.c]
        # masked_cons_error = cons_error * mask

        # Normal L2 loss, take mean over actual data
        td_loss = (masked_td_error ** 2).sum() / mask[:, :-self.c].sum()
  
        loss = td_loss + self.phi * dist_loss

        # Do RL Learning
        self.mac.agent.encoder.requires_grad_(False)
        self.mac.agent.state_encoder.requires_grad_(False)
        self.mac.agent.decoder.requires_grad_(False)
        # self.optimiser.zero_grad()
        loss.backward()
   

        # episode_num should be pulic
        if (t_env - self.last_target_update_episode) / self.main_args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = t_env

        if t_env - self.task2train_info[task]["log_stats_t"] >= self.task2args[task].learner_log_interval:
            self.logger.log_stat(f"{task}/loss", loss.item(), t_env)
            self.logger.log_stat(f"{task}/td_loss", td_loss.item(), t_env)

            mask_elems = mask.sum().item()
            self.logger.log_stat(f"{task}/td_error_abs", (masked_td_error.abs().sum().item() / mask_elems), t_env)
            self.logger.log_stat(f"{task}/q_taken_mean", (chosen_action_qvals * mask).sum().item() / (
                        mask_elems * self.task2args[task].n_agents), t_env)
            self.logger.log_stat(f"{task}/target_mean",
                                 (targets * mask[:, :-self.c]).sum().item() / (mask_elems * self.task2args[task].n_agents), t_env)
            self.task2train_info[task]["log_stats_t"] = t_env

    def pretrain(self, batch: EpisodeBatch, t_env: int, episode_num: int, task: str):
        if self.pretrain_steps == 0:
            self._reset_optimizer()
            for t in self.task2args:
                task_args = self.task2args[t]
                self.task2train_info[t]["log_stats_t"] = -task_args.learner_log_interval - 1
        
        self.train_vae(batch, t_env, episode_num, task)
        self.pretrain_steps += 1

    def pretrain_graph(self, batch: EpisodeBatch,  task: str, batch2: EpisodeBatch,task2: str): 
        self.ot_hyperparams["task1_nagt"]=self.task2n_agents[task]
        self.ot_hyperparams["task2_nagt"]=self.task2n_agents[task2]

        gloss=0
        actions = batch["actions"][:, :]
        state_graph1=batch["state_graph"]
        mac_out1 = []
        self.mac.init_hidden(batch.batch_size, task)
        for t in range(batch.max_seq_length):
            agent_outs,_ = self.mac.forward_skill(batch, t=t, task=task, actions=actions[:, t, :])
            mac_out1.append(agent_outs)
        mac_out1 = th.stack(mac_out1, dim=1)  
        skill_graph1=th.cdist(mac_out1,mac_out1)
        gloss+=graphloss(state_graph1,skill_graph1)

        actions2 = batch2["actions"][:, :]
        state_graph2=batch2["state_graph"]
        mac_out2 = []
        self.mac.init_hidden(batch2.batch_size, task2)
        for t in range(batch2.max_seq_length):
            agent_outs2,_ = self.mac.forward_skill(batch2, t=t, task=task2, actions=actions2[:, t, :])
            mac_out2.append(agent_outs2)
        mac_out2 = th.stack(mac_out2, dim=1)  
        skill_graph2=th.cdist(mac_out2,mac_out2)
        gloss+=graphloss(state_graph2,skill_graph2)

        gwdloss=GWDloss(state_graph1,state_graph2,mac_out1,mac_out2,self.ot_hyperparams)
       
        loss=gwdloss*self.main_args.omega+gloss*self.main_args.omega
        loss.backward()

      
    
    def test_pretrain(self, batch: EpisodeBatch, t_env: int, episode_num: int, task: str):
        self.test_vae(batch, t_env, episode_num, task)
    
    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int, task: str):
        if self.training_steps == 0:
            self._reset_optimizer()
            for t in self.task2args:
                task_args = self.task2args[t]
                self.task2train_info[t]["log_stats_t"] = -task_args.learner_log_interval - 1

        self.train_policy(batch, t_env, episode_num, task)
        self.training_steps += 1

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
       
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
