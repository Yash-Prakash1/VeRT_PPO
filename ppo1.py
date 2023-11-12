import time
from zmqRemoteApi import RemoteAPIClient
import numpy as np
import math
import copy
import json
# import matplotlib.pyplot as plt
import torch
import RL_net1
import random
from collections import deque, namedtuple
from vrep_ctrl import sim_run
import itertools
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal
import wandb
import matplotlib.pyplot as plt


def optimize_model(batch, reward):
    global best_pol_loss, best_val_loss, use_normalized_logging
    # flag = False
    if len(batch) < batch_size:
        return
    
    batch_adv_list = [m.adv for m in batch]
    for i in range(len(batch_adv_list)):
            if isinstance(batch_adv_list[i], tuple):
                batch_adv_list[i] = batch_adv_list[i][0]
    batch_adv = np.stack(batch_adv_list,axis=0)
    
    batch_adv = torch.tensor(batch_adv,dtype=torch.float32).to(device)

    # Advantage Normalization
    batch_adv = (batch_adv - torch.mean(batch_adv))/(torch.std(batch_adv)+1e-7)

    # Old Rewards
    old_reward_list = [m.old_reward for m in batch]
    # for i in range(len(old_reward_list)):
    #         if isinstance(batch_adv_list[i], tuple):
    #             batch_adv_list[i] = batch_adv_list[i][0]
    old_reward = torch.tensor(old_reward_list)

        # Variables to accumulate loss
    pol_loss_acc = []
    val_loss_acc = []
    
    # total_loss_acc = []

    batches = list(range(0,len(batch),batch_size)) 
    batches = np.random.permutation(batches)
    for _ in range(Policy_EPOCHS):
            for mb in batches:
                # print("mb is", mb)
                optimizer_policy.zero_grad()
                mini_batch = batch[mb:mb+batch_size]
                # minib_old_log_policy = old_log_policy[mb:mb+batch_size]
                # minib_old_log_p = old_log_policy[mb:mb+batch_size]
                minib_adv = batch_adv[mb:mb+batch_size]

                pol_loss, kl = Policy_loss(mini_batch, minib_adv, CLIP_EPS, device)
                # if kl > 1.5*target_kl:
                #     break
                if pol_loss < best_pol_loss:
                    best_pol_loss = pol_loss
                    print("On interation:", count, "Lowest Policy Loss:", best_pol_loss)
                
                # optimizer_value.zero_grad()
                pol_loss.backward()
                optimizer_policy.step()
                pol_loss_acc.append(float(pol_loss))

    for _ in range(Value_EPOCHS):
            for mb in batches:
                # print("mb is", mb)
                mini_batch = batch[mb:mb+batch_size]
                # minib_old_log_policy = old_log_policy[mb:mb+batch_size]
                # minib_old_log_p = old_log_policy[mb:mb+batch_size]
                minib_adv = batch_adv[mb:mb+batch_size]

                val_loss = Value_loss(mini_batch, critic, minib_adv, device)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    print("On interation:", count, "Lowest Value Loss:", best_val_loss)
                
                optimizer_value.zero_grad()          
                val_loss.backward()
                optimizer_value.step()
                # total_loss_acc.append(float(total_loss))
                val_loss_acc.append(float(val_loss))
    if np.mean(val_loss_acc)<20.0:
        use_normalized_logging = True

    if use_normalized_logging:
        wandb.log({"Value Loss": np.mean(val_loss_acc), "Policy Loss": np.mean(pol_loss_acc), "Value Loss Normalized": np.mean(val_loss_acc), "Average Run Reward": reward, "Learning Rate": optimizer_policy.param_groups[0]['lr'], "Old Reward Compare": torch.mean(old_reward)})    
    else:
        wandb.log({"Policy Loss": np.mean(pol_loss_acc), "Value Loss": np.mean(val_loss_acc), "Average Run Reward": reward, "Learning Rate": optimizer_policy.param_groups[0]['lr']})
    scheduler_policy.step()
    scheduler_value.step()

def steps(current_state, best_reward, best_cumulative_reward):
    memories = []
    best_state = current_state
    global end_sim_var
    s = 0
    cumulative_reward = 0.0
    
    while end_sim_var==False:
        # self.n_iter +=1
        s+=1
        action, log_prob, _ =set_action(current_state)
        # print("Step Number:", s) 
        state_value = float(critic(torch.tensor(current_state)))
        # print("Action is", action)
        action2sim=np.array([action[0].item(),action[0].item(),action[0].item(),action[0].item(),action[1].item()])
        # print("Action 2 sim:", action2sim)
        flag, observation, sim_time, orient_flag = sim_scene.step_sim(action2sim)
        observation=torch.tensor(observation,dtype=torch.float)
        
        if flag ==False:
        # Reward Shaping
            if np.array(sim_scene.sim.getObjectVelocity(sim_scene.body_ids[0]))[0][0] > 0.5:
                velocity_bonus = 2.0
            elif np.array(sim_scene.sim.getObjectVelocity(sim_scene.body_ids[0]))[0][0] > 0.75:
                velocity_bonus = 3.5
            elif np.array(sim_scene.sim.getObjectVelocity(sim_scene.body_ids[0]))[0][0] > 1.0:
                velocity_bonus = 5.0
            elif np.array(sim_scene.sim.getObjectVelocity(sim_scene.body_ids[0]))[0][0] > 3.0:
                velocity_bonus = 20.0
            elif np.array(sim_scene.sim.getObjectVelocity(sim_scene.body_ids[0]))[0][0] > 5.0:
                velocity_bonus = 50.0
            else:
                velocity_bonus = np.array(sim_scene.sim.getObjectVelocity(sim_scene.body_ids[0]))[0][0]
            # distance_penalty = -np.linalg.norm(np.array(sim_scene.sim.getObjectPosition(sim_scene.body_ids[0],-1)) - np.array(sim_scene.final_pos))
            # velocity_bonus = np.array(sim_scene.sim.getObjectVelocity(sim_scene.body_ids[0]))[0][0]

            immediate_reward =  velocity_bonus
            cumulative_reward = cumulative_reward*gamma + immediate_reward
            velocity = np.square(np.array(sim_scene.sim.getObjectVelocity(sim_scene.body_ids[0]))[0][0])

        if flag:
            # print("Flag is true")
            if sim_time >= n_time or orient_flag:
                immediate_reward -= 5.0
                
                
            else:
                immediate_reward += 20.0
            cumulative_reward = cumulative_reward*gamma + immediate_reward
            # last_value = float(critic(torch.tensor(current_state)))

                # sim_scene=sim_run(joint_names,body_names,scene_name) 
            end_sim_var = True
        next_state=copy.copy(observation)
        done= int(end_sim_var)
        memories.append(Memory(obs=current_state,action=action,new_obs=next_state,reward=torch.tensor([immediate_reward],dtype=torch.float),done=torch.tensor([done],dtype=torch.float), value=state_value, adv=None, log_prob=log_prob, returns=None, old_reward = velocity))
        # wandb.log({"Immediate Reward": immediate_reward})
            # print("Iter:", s, "Best Reward so far is:", reward)
        current_state = next_state
            
        if immediate_reward > best_reward:
            best_reward = immediate_reward
            best_state = current_state
            print("Iter:", s, "Best Reward so far is:", best_reward)
        # optimize_model(generalized_advantage_estimation(memories))
    # wandb.log({"Average Run Reward": cumulative_reward/s})
    # obs = [m.obs for m in memories]
    # for i in range(len(obs)):
    #     if isinstance(obs[i], tuple):
    #         obs[i] = obs[i][0]
    # obs = torch.stack(obs, axis=0)

    # act =  [m.action for m in memories]
    # act = torch.stack(act, axis=0)
    # _, log_prob, _ =set_action(state=obs, action=act)
    # memories.append(Memory(obs=memories.obs,action=memories.action,new_obs=memories.new_obs,reward=memories.reward,done=memories.done, value=memories.value, adv=None, log_prob=log_prob, returns=None))

    if cumulative_reward > best_cumulative_reward:
        best_cumulative_reward = cumulative_reward
        print("On Cycle number",count, "Best Cumulative Reward is ", (cumulative_reward), "Learning Rate:", optimizer_policy.param_groups[0]['lr'])
    optimize_model(generalized_advantage_estimation(memories),cumulative_reward)
    
    return generalized_advantage_estimation(memories), best_reward, end_sim_var, best_state, best_cumulative_reward

        

def generalized_advantage_estimation(memories):
    upd_memories = []
    returns = []
    # returnz = []
    # values = []
    gae = 0

    for t in reversed(range(len(memories))):
        if t == len(memories)-1:
            next_non_terminal = 1.0-memories[t].done
            next_value = memories[t].value
        else:
            next_non_terminal = 1 - memories[t].done
            next_value = memories[t+1].value
        delta = memories[t].reward + gamma*gae_lambda*next_value*next_non_terminal - memories[t].value
        gae = delta + gamma*gae_lambda*next_non_terminal*gae
        advantages = gae
        returns = advantages + memories[t].value

          
        # if memories[t].done:
        #     gae = memories[t].reward
        # else:
        #     delta = memories[t].reward + gamma*memories[t+1].value - memories[t].value
        #     gae = delta + gae*gamma*gae_lambda
        #     # gae = np.array(gae[0])
        #     # returns.insert(0, gae + memories[t].value)
        #     # returnz.append(gae+memories[t].value)
        #     values.insert(0,memories[t].value)
        upd_memories.append(Memory(obs=memories[t].obs,action=memories[t].action,new_obs = memories[t].new_obs, reward=memories[t].reward, done = memories[t].done, value = memories[t].value, adv = advantages, log_prob=memories[t].log_prob, returns=returns, old_reward=memories[t].old_reward))
    # returns = advantages + memories.value
    # upd_memories.append(Memory(obs=memories[t].obs,action=memories[t].action,new_obs = memories[t].new_obs, reward=gae + memories[t].value, done = memories[t].done, value = memories[t].value, adv = memories[t], log_prob=memories[t].log_prob))
    return upd_memories[::-1]

def log_policy_prob(mean,std,actions):

    act_log_softmax = -((mean-actions)**2)/(2*torch.exp(std).clamp(min=1e-4)) - torch.log(torch.sqrt(2*math.pi*torch.exp(std)))
    return act_log_softmax

def compute_log_policy_prob(batch,nn_policy, device):

    obs_list = [m.obs for m in batch]
    act_list = [m.action for m in batch]

    for i in range(len(obs_list)):
        if isinstance(obs_list[i], tuple):
            obs_list[i] = obs_list[i][0]
    obs_array = np.stack(obs_list,axis=0)
    obs_tensor = torch.tensor(obs_array,dtype=torch.float32).to(device)
    n_mean = nn_policy(obs_tensor)
    n_mean = n_mean.type(torch.DoubleTensor)
    logstd = actor.logstd.type(torch.DoubleTensor)

    for i in range(len(act_list)):
        if isinstance(act_list[i], tuple):
            act_list[i] = act_list[i][0]
    act_array = np.stack(act_list,axis=0)
    act_tensor = torch.tensor(act_array,dtype=torch.float32).to(device)

    return log_policy_prob(n_mean, logstd, act_tensor)

def Policy_loss(memories, adv, epsilon, device):

    obs_list = [m.obs for m in memories]
    for i in range(len(obs_list)):
        if isinstance(obs_list[i], tuple):
            obs_list[i] = obs_list[i][0]
    obs_array = np.stack(obs_list, axis=0)
    obs_tensor = torch.tensor(obs_array, dtype=torch.float32).to(device)

    act_list = [m.action for m in memories]
    act_array = np.stack(act_list, axis=0)
    act_tensor = torch.tensor(act_array, dtype=torch.float32).to(device)

    old_log_policy_list = [m.log_prob for m in memories]
    old_log_policy = torch.stack(old_log_policy_list,axis=0)

    _, new_log_policy, entropy = set_action(obs_tensor, act_tensor)

    rt_theta = torch.exp(new_log_policy - old_log_policy.detach())
    # adv = adv.unsqueeze(-t-1)

    # if sum(np.round(new_log_policy.detach().numpy(), decimals=4) == np.round(old_log_policy.detach().numpy(), decimals=4))[0] != sum(np.round(new_log_policy.detach().numpy(), decimals=4) == np.round(old_log_policy.detach().numpy(), decimals=4))[1]:
        # flag = True
        # print("STOP AND CHECK NOW!!")

    pg_loss = -torch.mean(torch.min(rt_theta*adv, torch.clamp(rt_theta, 1-epsilon, 1+epsilon)*adv))
    approx_kl = (old_log_policy - new_log_policy).mean().item()
    # entropy_loss = ent_coef*entropy
    Loss = pg_loss#+entropy_loss

    # rew_loss = -np.mean(rew_array)

    # total_loss = pg_loss + vl_loss_coef*vl_loss #+ et_loss_coef*
    # wandb.log({"Total Loss in Loop": total_loss, "Policy Loss in Loop": pg_loss, "Value Loss in Loop": vl_loss})

    return Loss, approx_kl#, policy_info

def Value_loss(memories, nn_value, adv, device):
    
    obs_list = [m.obs for m in memories]
    for i in range(len(obs_list)):
        if isinstance(obs_list[i], tuple):
            obs_list[i] = obs_list[i][0]
    obs_array = np.stack(obs_list, axis=0)
    obs_tensor = torch.tensor(obs_array, dtype=torch.float32).to(device)
    
    rew_list = [m.reward for m in memories]
    for i in range(len(rew_list)):
        if isinstance(rew_list[i], tuple):
            rew_list[i] = rew_list[i][0]
    rew_array = np.stack(rew_list, axis=0).squeeze(-1)
    rew_tensor = torch.tensor(rew_array, dtype=torch.float32).to(device)
    rewards = rew_tensor

    value = nn_value(obs_tensor)
    # returns = [m.returns for m in memories]
    # returns = torch.tensor(returns, dtype=torch.float32).to(device)
    returns = adv + value
    returns = returns.squeeze(-1)


    vl_loss = F.mse_loss(returns, rewards)

    return vl_loss
    

def set_action(state, action=None, log_prob=None, entropy=None):
    global steps_done
    steps_done+=1
    

    mean = actor(state)

    logstd = actor.logstd[0].expand_as(mean)
    std = torch.exp(logstd)

    probs = Normal(mean, std)
    if action is None:
        action = probs.sample()
        log_prob = probs.log_prob(action).sum(axis=-1)
    if action is not None:
        log_prob = probs.log_prob(action).sum(axis=-1)     
        entropy = probs.entropy().mean().item()
    


    return action, log_prob, entropy#, critic(state)


Memory = namedtuple('Memory', ['obs', 'action', 'new_obs', 'reward', 'done', 'value', 'adv', 'log_prob','returns', 'old_reward'], rename=False)



best_reward = -1e5
eps_start = 0.9
eps_end = 0.05
eps_decay = 100
CLIP_GRADIENT = 0.2
CLIP_EPS = 0.2
batch_size=20
tau = 0.005
gamma = 0.99
gae_lambda = 0.95
Policy_LR=1e-3
Value_LR = 5e-3
loss_out=[]
xpos_rec=[]
zpos_rec=[]
time_rec=[]
iters_rec=[]
steps_done=10000
run_nums=0
alpha = 0.2
sim_time_rec=[]
Max_Runs = 260
device = "cpu" 
Policy_EPOCHS = 10
Value_EPOCHS = 10
obs_dim = 19
act_dim = 2
n_steps = 2048
n_time = 10
best_cumulative_reward = -1e5
best_pol_loss = 1e5
best_val_loss = 1e5
vl_loss_coef = 1.0
et_loss_coef = 0.0
use_normalized_logging = False
rew_coef = 5
ent_coef = 0.01
target_kl = 0.05

actor = RL_net1.Actor(obs_dim, act_dim).to(device)
critic = RL_net1.Critic(obs_dim)
# ac.load_state_dict(torch.load('D:\CMU\Biorobotics Lab\MovingCoM-sac\PPO1'))
ac_target=copy.deepcopy(actor)
# print(ac_target)
optimizer_policy = optim.Adam(actor.parameters(), lr=Policy_LR)
optimizer_value = optim.Adam(critic.parameters(), lr=Value_LR)
scheduler_policy = torch.optim.lr_scheduler.StepLR(optimizer_policy,step_size=50, gamma=0.5)
scheduler_value = torch.optim.lr_scheduler.StepLR(optimizer_value,step_size=50, gamma=0.5)


q_params = itertools.chain(actor.parameters(), critic.parameters())


## Initialize the simulations ##
joint_names=['/Wheel_assem1','/Wheel_assem3','/Wheel_assem4','/Wheel_assem2','/Payload_x']
body_names=['/Frame1']
# scene_name="D:\CMU\Biorobotics Lab\MovingCoM-sac\wheel_vehicle_flat_x.ttt"
scene_name="D:\CMU\Biorobotics Lab\MovingCoM-sac\wheel_flat_terrain.ttt"
count=0
# change_track_links()

# Setting up WandB
# wandb login(key='95ed7ca5b0ac2b367863cf1ce5fa436edd52a1af')

wandb.init(
    name="Run18 (BLOCK REWARD WITH 0.75)",
    reinit=True,
    project='PPO_implementation_VeRT',
    config={
        "gamma": gamma,
        "gae lambda": gae_lambda,
        "Max_Runs": Max_Runs,
        "Policy_EPOCHS": Policy_EPOCHS,
        "Value_EPOCHS": Value_EPOCHS,
        "act_dim": act_dim,
        "obs_dim": obs_dim,
        "batch_size": batch_size,
        "Starting Policy Learning Rate": Policy_LR,
        "Time Cutoff for Copelliasim": "10 seconds",
        "Target KL": target_kl, 


    }

)
# device = torch.device("cuda:0")
# actor.to(device)
# actor.load_state_dict(torch.load("PPO14"))
while run_nums<Max_Runs:

    sim_scene=sim_run(joint_names,body_names,scene_name, n_time=n_time) 
    init_state=np.array(sim_scene.sim.getObjectPosition(sim_scene.body_ids[0],-1))
    end_sim_var=False
    # sim_boolparam_display = 0
    sim_scene.step_sim(np.zeros(5))
    
    
        
        
    initial_state=torch.tensor(sim_scene.record_state(),dtype=torch.float)    
    batch, reward, end_sim_var, state, cumulative_rew = steps(initial_state, best_reward, best_cumulative_reward)

    best_reward = reward
    best_cumulative_reward = cumulative_rew
    # optimize_model(batch)  
    
    
    if (count+1)%50==0:
        torch.save(actor.state_dict(), 'PPO18')
        # count=0
    else:
        count+=1
    if run_nums %10 == 0:
        print(run_nums)
    run_nums+=1
wandb.finish()
print('Run 18 Completed')
torch.save(actor.state_dict(), 'PPO Run18')