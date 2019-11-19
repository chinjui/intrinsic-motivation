import argparse
from collections import deque
import os
import random
import sys
import time
import yaml

import numpy as np
import gym
import torch
import torch.optim as optim

from models import ActorCritic, FwdDyn
from ppo import PPO
import gym_fetch
from gym_fetch.wrappers import make_vec_envs
import utils
import logger

parser = argparse.ArgumentParser(description='PPO')
parser.add_argument('--experiment-name', type=str, default='RandomAgent')
parser.add_argument('--env-id', type=str, default='FetchPush-v1')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--log-dir', type=str, default=None)
parser.add_argument('--clean-dir', action='store_true')
parser.add_argument('--log-interval', type=int, default=1)
parser.add_argument('--checkpoint-interval', type=int, default=20)
parser.add_argument('--eval-interval', type=int, default=100)
parser.add_argument('--add-intrinsic-reward', action='store_true')
parser.add_argument('--intrinsic-coef', type=float, default=1.0)
parser.add_argument('--max-intrinsic-reward', type=float, default=None)
parser.add_argument('--num-env-steps', type=int, default=int(1e7))
parser.add_argument('--num-processes', type=int, default=32)
parser.add_argument('--num-steps', type=int, default=2048)
parser.add_argument('--ppo-epochs', type=int, default=10)
parser.add_argument('--dyn-epochs', type=int, default=5)
parser.add_argument('--num-mini-batch', type=int, default=32)
parser.add_argument('--pi-lr', type=float, default=1e-4)
parser.add_argument('--v-lr', type=float, default=1e-3)
parser.add_argument('--dyn-lr', type=float, default=1e-3)
parser.add_argument('--closer-lr', type=float, default=1e-3)
parser.add_argument('--hidden-size', type=int, default=128)
parser.add_argument('--clip-param', type=float, default=0.3)
parser.add_argument('--value-coef', type=float, default=0.5)
parser.add_argument('--entropy-coef', type=float, default=0.01)
parser.add_argument('--grad-norm-max', type=float, default=5.0)
parser.add_argument('--dyn-grad-norm-max', type=float, default=5)
parser.add_argument('--gamma', type=float, default=0.9)
parser.add_argument('--use-gae', action='store_true')
parser.add_argument('--gae-lambda', type=float, default=0.95)
parser.add_argument('--draw-near-interval', type=int, default=3)
parser.add_argument('--share-optim', action='store_true')
parser.add_argument('--predict-delta-obs', action='store_true')
parser.add_argument('--use-linear-lr-decay', action='store_true')
parser.add_argument('--use-clipped-value-loss', action='store_true')
parser.add_argument('--use-tensorboard', action='store_true')
parser.add_argument('--cuda', action='store_false', default=True, help='enables CUDA training')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--render', action='store_true')

if __name__ == '__main__':
    # parse arguments
    args = parser.parse_args()

    # setup logging
    if args.log_dir is None:
        log_dir = utils.create_log_dirs("{}/{}".format(args.env_id, args.experiment_name),
                                        force_clean=args.clean_dir)
        args.__dict__['log_dir'] = log_dir
    else:
        log_dir = args.log_dir
    logger.configure(log_dir, ['stdout', 'log', 'csv'], tbX=args.use_tensorboard)

    # save parameters
    with open(os.path.join(log_dir,'params.yaml'), 'w') as f:
            yaml.dump(args.__dict__, f, default_flow_style=False)

    # set device and random seeds
    device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")
    torch.set_num_threads(1)
    utils.set_random_seeds(args.seed, args.cuda, args.debug)

    # setup environment
    envs_extr = make_vec_envs(env_id=args.env_id,
                         seed=args.seed,
                         num_processes=args.num_processes,
                         gamma=None,
                         log_dir=log_dir,
                         device=device,
                         obs_keys=['observation', 'desired_goal'],
                         allow_early_resets=False)
    envs_intr = make_vec_envs(env_id=args.env_id,
                         seed=args.seed,
                         num_processes=args.num_processes,
                         gamma=None,
                         log_dir=log_dir,
                         device=device,
                         obs_keys=['observation', 'desired_goal'],
                         allow_early_resets=False)

    # create agent
    agent_extr = PPO(log_dir,
                envs_extr.observation_space,
                envs_extr.action_space,
                actor_critic=ActorCritic,
                dynamics_model=FwdDyn,
                optimizer=optim.Adam,
                hidden_size=args.hidden_size,
                num_steps=args.num_steps,
                num_processes=args.num_processes,
                ppo_epochs=args.ppo_epochs,
                num_mini_batch=args.num_mini_batch,
                pi_lr=args.pi_lr,
                v_lr=args.v_lr,
                dyn_lr=args.dyn_lr,
                clip_param=args.clip_param,
                value_coef=args.value_coef,
                entropy_coef=args.entropy_coef,
                grad_norm_max=args.grad_norm_max,
                use_clipped_value_loss=True,
                use_tensorboard=args.use_tensorboard,
                add_intrinsic_reward=False, # No
                predict_delta_obs=args.predict_delta_obs,
                device=device,
                share_optim=args.share_optim,
                debug=args.debug)
    agent_intr = PPO(log_dir,
                envs_intr.observation_space,
                envs_intr.action_space,
                actor_critic=ActorCritic,
                dynamics_model=FwdDyn,
                optimizer=optim.Adam,
                hidden_size=args.hidden_size,
                num_steps=args.num_steps,
                num_processes=args.num_processes,
                ppo_epochs=args.ppo_epochs,
                num_mini_batch=args.num_mini_batch,
                pi_lr=args.pi_lr,
                v_lr=args.v_lr,
                dyn_lr=args.dyn_lr,
                clip_param=args.clip_param,
                value_coef=args.value_coef,
                entropy_coef=args.entropy_coef,
                grad_norm_max=args.grad_norm_max,
                use_clipped_value_loss=True,
                use_tensorboard=args.use_tensorboard,
                add_intrinsic_reward=True, # Yes
                predict_delta_obs=args.predict_delta_obs,
                device=device,
                share_optim=args.share_optim,
                debug=args.debug,
                use_extrinsic_reward=False)


    # optimizer to draw extrinsic and intrinsic policies closer
    closer_optim_extr = optim.Adam(agent_extr.actor_critic.policy.parameters(), args.closer_lr)
    closer_optim_intr = optim.Adam(agent_intr.actor_critic.policy.parameters(), args.closer_lr)

    # reset envs and initialize rollouts
    obs_extr = envs_extr.reset()
    obs_intr = envs_intr.reset()
    agent_extr.rollouts.obs[0].copy_(obs_extr[1])
    agent_extr.rollouts.to(device)
    agent_intr.rollouts.obs[0].copy_(obs_intr[1])
    agent_intr.rollouts.to(device)

    # start training
    agent_extr.train()
    agent_intr.train()
    start = time.time()

    num_updates = int(args.num_env_steps // args.num_processes // args.num_steps)
    print("Number of updates:", num_updates)

    for update in range(num_updates):
        need_record_video = update % 15 == 0

        # decrease learning rate linearly
        if args.use_linear_lr_decay:
            # extrinsic policy
            if args.share_optim:
                utils.update_linear_schedule(optimizer=agent_extr.optimizer,
                                             update=update,
                                             total_num_updates=num_updates,
                                             initial_lr=args.pi_lr)
            else:
                utils.update_linear_schedule(optimizer=agent_extr.policy_optimizer,
                                             update=update,
                                             total_num_updates=num_updates,
                                             initial_lr=args.pi_lr)

                utils.update_linear_schedule(optimizer=agent_extr.value_fn_optimizer,
                                             update=update,
                                             total_num_updates=num_updates,
                                             initial_lr=args.v_lr)

            # intrinsic policy
            if args.share_optim:
                utils.update_linear_schedule(optimizer=agent_intr.optimizer,
                                             update=update,
                                             total_num_updates=num_updates,
                                             initial_lr=args.pi_lr)
            else:
                utils.update_linear_schedule(optimizer=agent_intr.policy_optimizer,
                                             update=update,
                                             total_num_updates=num_updates,
                                             initial_lr=args.pi_lr)

                utils.update_linear_schedule(optimizer=agent_intr.value_fn_optimizer,
                                             update=update,
                                             total_num_updates=num_updates,
                                             initial_lr=args.v_lr)

        # linear decay the learning rate of draw closer optimizer
        utils.update_linear_schedule(optimizer=closer_optim_extr,
                                     update=update,
                                     total_num_updates=num_updates,
                                     initial_lr=args.closer_lr)
        utils.update_linear_schedule(optimizer=closer_optim_intr,
                                     update=update,
                                     total_num_updates=num_updates,
                                     initial_lr=args.closer_lr)


        extrinsic_rewards_extr = []
        episode_length_extr = []
        intrinsic_rewards_extr = []
        solved_episodes_extr = []
        extrinsic_rewards_intr = []
        episode_length_intr = []
        intrinsic_rewards_intr = []
        solved_episodes_intr = []
        obs_replay_buffer = deque()

        for step in range(args.num_steps):
            # render
            if args.render and need_record_video:
                frame = envs_extr.render()

            # select action
            value_extr, action_extr, action_log_probs_extr = agent_extr.select_action(step)
            value_intr, action_intr, action_log_probs_intr = agent_intr.select_action(step)

            # take a step in the environment
            obs_extr, reward_extr, done_extr, infos_extr = envs_extr.step(action_extr)
            obs_intr, reward_intr, done_intr, infos_intr = envs_intr.step(action_intr)

            # calculate intrinsic reward
            # intrinsic-policy part
            intrinsic_reward_intr = args.intrinsic_coef * agent_intr.compute_intrinsic_reward(step)
            if args.max_intrinsic_reward is not None:
                intrinsic_reward_intr = torch.clamp(agent_intr.compute_intrinsic_reward(step), 0.0, args.max_intrinsic_reward)
            intrinsic_rewards_intr.extend(list(intrinsic_reward_intr.cpu().numpy().reshape(-1)))
            # extrinsic-policy part
            intrinsic_reward_extr = torch.tensor(0).view(1, 1)
            intrinsic_rewards_extr.extend(list(intrinsic_reward_extr.cpu().numpy().reshape(-1)))

            # store experience
            agent_extr.store_rollout(obs_extr[1], action_extr, action_log_probs_extr,
                                value_extr, reward_extr, intrinsic_reward_extr,
                                done_extr)
            agent_intr.store_rollout(obs_intr[1], action_intr, action_log_probs_intr,
                                value_intr, reward_intr, intrinsic_reward_intr,
                                done_intr)

            # get final episode rewards
            for info in infos_extr:
                if 'episode' in info.keys():
                    extrinsic_rewards_extr.append(info['episode']['r'])
                    episode_length_extr.append(info['episode']['l'])
                    solved_episodes_extr.append(info['is_success'])
            for info in infos_intr:
                if 'episode' in info.keys():
                    extrinsic_rewards_intr.append(info['episode']['r'])
                    episode_length_intr.append(info['episode']['l'])
                    solved_episodes_intr.append(info['is_success'])

        # compute returns
        agent_extr.compute_returns(args.gamma, args.use_gae, args.gae_lambda)
        agent_intr.compute_returns(args.gamma, args.use_gae, args.gae_lambda)

        # draw extrinsic and intrinsic policy closer
        if update % args.draw_near_interval == 0:
          obs_from_both = torch.cat([agent_extr.rollouts.obs, agent_intr.rollouts.obs], dim=1).view(-1, *envs_extr.observation_space.shape)
          random_perm = torch.randperm(obs_from_both.shape[0])
          obs_from_both = obs_from_both[random_perm]
          bsize = 64
          divs_extr = []
          divs_intr = []
          for i in range(obs_from_both.shape[0] // bsize):
            obs_feed = obs_from_both[bsize*i:bsize*(i+1)]
            out_extr = agent_extr.actor_critic.select_action_distr(obs_feed)
            out_intr = agent_intr.actor_critic.select_action_distr(obs_feed)
            closer_optim_extr.zero_grad()
            closer_optim_intr.zero_grad()
            div_extr = torch.distributions.kl.kl_divergence(out_extr, out_intr).mean()
            div_intr = torch.distributions.kl.kl_divergence(out_intr, out_extr).mean()
            div_extr.backward(retain_graph=True)
            div_intr.backward()
            closer_optim_extr.step()
            closer_optim_intr.step()
            divs_extr.append(div_extr)
            divs_intr.append(div_intr)



        # update policy and value_fn, reset rollout storage
        tot_loss_extr, pi_loss_extr, v_loss_extr, dyn_loss_extr, entropy_extr, kl_extr, delta_p_extr, delta_v_extr = \
            agent_extr.update(obs_mean=obs_extr[2], obs_var=obs_extr[3])
        tot_loss_intr, pi_loss_intr, v_loss_intr, dyn_loss_intr, entropy_intr, kl_intr, delta_p_intr, delta_v_intr = \
            agent_intr.update(obs_mean=obs_intr[2], obs_var=obs_intr[3])

        # log data
        if update % args.log_interval == 0:
            current = time.time()
            elapsed = current - start
            total_steps = (update + 1) * args.num_processes * args.num_steps
            fps =int(total_steps / (current - start))

            logger.logkv('Time/Updates', update)
            logger.logkv('Time/Total Steps', total_steps)
            logger.logkv('Time/FPS', fps)
            logger.logkv('Time/Current', current)
            logger.logkv('Time/Elapsed', elapsed)
            logger.logkv('Time/Epoch', elapsed)

            # logs for extrinsic policy
            logger.logkv('ExtrinsicModel/Extrinsic/Mean', np.mean(extrinsic_rewards_extr))
            # logger.logkv('Extrinsic/Median', np.median(extrinsic_rewards))
            # logger.logkv('Extrinsic/Min', np.min(extrinsic_rewards))
            # logger.logkv('Extrinsic/Max', np.max(extrinsic_rewards))
            logger.logkv('ExtrinsicModel/Episodes/Solved', np.mean(solved_episodes_extr))
            # logger.logkv('Episodes/Length', np.mean(episode_length))
            logger.logkv('ExtrinsicModel/Intrinsic/Mean', np.mean(intrinsic_rewards_extr))
            # logger.logkv('Intrinsic/Median', np.median(intrinsic_rewards))
            # logger.logkv('Intrinsic/Min', np.min(intrinsic_rewards))
            # logger.logkv('Intrinsic/Max', np.max(intrinsic_rewards))
            # logger.logkv('Loss/Total', tot_loss)
            logger.logkv('ExtrinsicModel/Loss/Policy', pi_loss_extr)
            logger.logkv('ExtrinsicModel/Loss/Value', v_loss_extr)
            logger.logkv('ExtrinsicModel/Loss/Entropy', entropy_extr)
            logger.logkv('ExtrinsicModel/Loss/KL', kl_extr)
            # logger.logkv('Loss/DeltaPi', delta_p)
            # logger.logkv('Loss/DeltaV', delta_v)
            logger.logkv('ExtrinsicModel/Loss/Dynamics', dyn_loss_extr)
            logger.logkv('ExtrinsicModel/Value/Mean', np.mean(agent_extr.rollouts.value_preds.cpu().data.numpy()))
            # logger.logkv('Value/Median', np.median(agent.rollouts.value_preds.cpu().data.numpy()))
            # logger.logkv('Value/Min', np.min(agent.rollouts.value_preds.cpu().data.numpy()))
            # logger.logkv('Value/Max', np.max(agent.rollouts.value_preds.cpu().data.numpy()))

            # logs for intrinsic policy
            logger.logkv('IntrinsicModel/Extrinsic/Mean', np.mean(extrinsic_rewards_intr))
            # logger.logkv('Extrinsic/Median', np.median(extrinsic_rewards))
            # logger.logkv('Extrinsic/Min', np.min(extrinsic_rewards))
            # logger.logkv('Extrinsic/Max', np.max(extrinsic_rewards))
            logger.logkv('IntrinsicModel/Episodes/Solved', np.mean(solved_episodes_intr))
            # logger.logkv('Episodes/Length', np.mean(episode_length))
            logger.logkv('IntrinsicModel/Intrinsic/Mean', np.mean(intrinsic_rewards_intr))
            # logger.logkv('Intrinsic/Median', np.median(intrinsic_rewards))
            # logger.logkv('Intrinsic/Min', np.min(intrinsic_rewards))
            # logger.logkv('Intrinsic/Max', np.max(intrinsic_rewards))
            # logger.logkv('Loss/Total', tot_loss)
            logger.logkv('IntrinsicModel/Loss/Policy', pi_loss_intr)
            logger.logkv('IntrinsicModel/Loss/Value', v_loss_intr)
            logger.logkv('IntrinsicModel/Loss/Entropy', entropy_intr)
            logger.logkv('IntrinsicModel/Loss/KL', kl_intr)
            # logger.logkv('Loss/DeltaPi', delta_p)
            # logger.logkv('Loss/DeltaV', delta_v)
            logger.logkv('IntrinsicModel/Loss/Dynamics', dyn_loss_intr)
            logger.logkv('IntrinsicModel/Value/Mean', np.mean(agent_intr.rollouts.value_preds.cpu().data.numpy()))
            # logger.logkv('Value/Median', np.median(agent.rollouts.value_preds.cpu().data.numpy()))
            # logger.logkv('Value/Min', np.min(agent.rollouts.value_preds.cpu().data.numpy()))
            # logger.logkv('Value/Max', np.max(agent.rollouts.value_preds.cpu().data.numpy()))
            logger.logkv('KLDivergence_extr', np.mean(torch.stack(divs_extr).mean().cpu().data.numpy()))
            logger.logkv('KLDivergence_intr', np.mean(torch.stack(divs_intr).mean().cpu().data.numpy()))

            if args.use_tensorboard:
                # logs for extrinsic model
                logger.add_scalar('ExtrinsicModel/reward/mean', np.mean(extrinsic_rewards_extr), total_steps, elapsed)
                # logger.add_scalar('reward/median', np.median(extrinsic_rewards), total_steps, elapsed)
                # logger.add_scalar('reward/min', np.min(extrinsic_rewards), total_steps, elapsed)
                # logger.add_scalar('reward/max', np.max(extrinsic_rewards), total_steps, elapsed)
                logger.add_scalar('ExtrinsicModel/episode/solved', np.mean(solved_episodes_extr), total_steps, elapsed)
                # logger.add_scalar('episode/length', np.mean(episode_length), total_steps, elapsed)
                logger.add_scalar('ExtrinsicModel/intrinsic/mean', np.mean(intrinsic_rewards_extr), total_steps, elapsed)
                # logger.add_scalar('intrinsic/median', np.median(intrinsic_rewards), total_steps, elapsed)
                # logger.add_scalar('intrinsic/min', np.min(intrinsic_rewards), total_steps, elapsed)
                # logger.add_scalar('intrinsic/max', np.max(intrinsic_rewards), total_steps, elapsed)
                # logger.add_scalar('loss/total', tot_loss, total_steps, elapsed)
                logger.add_scalar('ExtrinsicModel/loss/policy', pi_loss_extr, total_steps, elapsed)
                logger.add_scalar('ExtrinsicModel/loss/value', v_loss_extr, total_steps, elapsed)
                logger.add_scalar('ExtrinsicModel/loss/entropy', entropy_extr, total_steps, elapsed)
                logger.add_scalar('ExtrinsicModel/loss/kl', kl_extr, total_steps, elapsed)
                # logger.add_scalar('loss/delta_p', delta_p, total_steps, elapsed)
                # logger.add_scalar('loss/delta_v', delta_v, total_steps, elapsed)
                logger.add_scalar('ExtrinsicModel/loss/dynamics', dyn_loss_extr, total_steps, elapsed)
                logger.add_scalar('ExtrinsicModel/value/mean', np.mean(agent_extr.rollouts.value_preds.cpu().data.numpy()), total_steps, elapsed)
                # logger.add_scalar('value/median', np.median(agent.rollouts.value_preds.cpu().data.numpy()), total_steps, elapsed)
                # logger.add_scalar('value/min', np.min(agent.rollouts.value_preds.cpu().data.numpy()), total_steps, elapsed)
                # logger.add_scalar('value/max', np.max(agent.rollouts.value_preds.cpu().data.numpy()), total_steps, elapsed)

                # logs for intrinsic model
                logger.add_scalar('IntrinsicModel/reward/mean', np.mean(extrinsic_rewards_intr), total_steps, elapsed)
                # logger.add_scalar('reward/median', np.median(extrinsic_rewards), total_steps, elapsed)
                # logger.add_scalar('reward/min', np.min(extrinsic_rewards), total_steps, elapsed)
                # logger.add_scalar('reward/max', np.max(extrinsic_rewards), total_steps, elapsed)
                logger.add_scalar('IntrinsicModel/episode/solved', np.mean(solved_episodes_intr), total_steps, elapsed)
                # logger.add_scalar('episode/length', np.mean(episode_length), total_steps, elapsed)
                logger.add_scalar('IntrinsicModel/intrinsic/mean', np.mean(intrinsic_rewards_intr), total_steps, elapsed)
                # logger.add_scalar('intrinsic/median', np.median(intrinsic_rewards), total_steps, elapsed)
                # logger.add_scalar('intrinsic/min', np.min(intrinsic_rewards), total_steps, elapsed)
                # logger.add_scalar('intrinsic/max', np.max(intrinsic_rewards), total_steps, elapsed)
                # logger.add_scalar('loss/total', tot_loss, total_steps, elapsed)
                logger.add_scalar('IntrinsicModel/loss/policy', pi_loss_intr, total_steps, elapsed)
                logger.add_scalar('IntrinsicModel/loss/value', v_loss_intr, total_steps, elapsed)
                logger.add_scalar('IntrinsicModel/loss/entropy', entropy_intr, total_steps, elapsed)
                logger.add_scalar('IntrinsicModel/loss/kl', kl_intr, total_steps, elapsed)
                # logger.add_scalar('loss/delta_p', delta_p, total_steps, elapsed)
                # logger.add_scalar('loss/delta_v', delta_v, total_steps, elapsed)
                logger.add_scalar('IntrinsicModel/loss/dynamics', dyn_loss_intr, total_steps, elapsed)
                logger.add_scalar('IntrinsicModel/value/mean', np.mean(agent_intr.rollouts.value_preds.cpu().data.numpy()), total_steps, elapsed)
                # logger.add_scalar('value/median', np.median(agent.rollouts.value_preds.cpu().data.numpy()), total_steps, elapsed)
                # logger.add_scalar('value/min', np.min(agent.rollouts.value_preds.cpu().data.numpy()), total_steps, elapsed)
                # logger.add_scalar('value/max', np.max(agent.rollouts.value_preds.cpu().data.numpy()), total_steps, elapsed)

                if args.debug:
                    # extrinsic
                    logger.add_histogram('debug/actions_extr', agent_extr.rollouts.actions.cpu().data.numpy(), total_steps)
                    logger.add_histogram('debug/observations_extr', agent_extr.rollouts.obs.cpu().data.numpy(), total_steps)
                    logger.logkv('Action/Mean_extr', np.mean(agent_extr.rollouts.actions.cpu().data.numpy()))
                    logger.logkv('Action/Median_extr', np.median(agent_extr.rollouts.actions.cpu().data.numpy()))
                    logger.logkv('Action/Min_extr', np.min(agent_extr.rollouts.actions.cpu().data.numpy()))
                    logger.logkv('Action/Max_extr', np.max(agent_extr.rollouts.actions.cpu().data.numpy()))

                    # intrinsic
                    logger.add_histogram('debug/actions_intr', agent_intr.rollouts.actions.cpu().data.numpy(), total_steps)
                    logger.add_histogram('debug/observations_intr', agent_intr.rollouts.obs.cpu().data.numpy(), total_steps)
                    logger.logkv('Action/Mean_intr', np.mean(agent_intr.rollouts.actions.cpu().data.numpy()))
                    logger.logkv('Action/Median_intr', np.median(agent_intr.rollouts.actions.cpu().data.numpy()))
                    logger.logkv('Action/Min_intr', np.min(agent_intr.rollouts.actions.cpu().data.numpy()))
                    logger.logkv('Action/Max_intr', np.max(agent_intr.rollouts.actions.cpu().data.numpy()))

                    # extrinsic
                    total_grad_norm = 0
                    total_weight_norm = 0
                    for name, param in agent_extr.actor_critic.named_parameters():
                        logger.add_histogram('debug/param/{}_extr'.format(name), param.cpu().data.numpy(), total_steps)
                        grad_norm = param.grad.data.norm(2)
                        weight_norm = param.data.norm(2)
                        total_grad_norm += grad_norm.item() ** 2
                        total_weight_norm += weight_norm.item() ** 2

                    total_grad_norm = total_grad_norm ** (1. / 2)
                    total_weight_norm = total_weight_norm ** (1. / 2)
                    logger.add_scalar('debug/param/grad_norm_extr', total_grad_norm, total_steps, elapsed)
                    logger.add_scalar('debug/param/weight_norm_extr', total_weight_norm, total_steps, elapsed)

                    # intrinsic
                    total_grad_norm = 0
                    total_weight_norm = 0
                    for name, param in agent_intr.actor_critic.named_parameters():
                        logger.add_histogram('debug/param/{}_intr'.format(name), param.cpu().data.numpy(), total_steps)
                        grad_norm = param.grad.data.norm(2)
                        weight_norm = param.data.norm(2)
                        total_grad_norm += grad_norm.item() ** 2
                        total_weight_norm += weight_norm.item() ** 2

                    total_grad_norm = total_grad_norm ** (1. / 2)
                    total_weight_norm = total_weight_norm ** (1. / 2)
                    logger.add_scalar('debug/param/grad_norm_intr', total_grad_norm, total_steps, elapsed)
                    logger.add_scalar('debug/param/weight_norm_intr', total_weight_norm, total_steps, elapsed)

            logger.dumpkvs()

            # checkpoint model
            if (update + 1) % args.checkpoint_interval == 0:
                agent_extr.save_checkpoint()
                agent_intr.save_checkpoint()
