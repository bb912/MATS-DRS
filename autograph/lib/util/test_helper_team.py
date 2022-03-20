import matplotlib.pyplot as plt
import matplotlib.cm
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from autograph.play.maze_nn_aut_adv_team_share import get_player_env, do_env_step
import ptan
import numpy as np


figsize = (12,12)
policy_img_shape = (5, 4)


# TODO:
# 1) Add a run function to step through environment while doing mcts batches. It has the option to print all the policies (the existing function should only show pi_cnn)
#    include an option to use pi_cnn, pi_tree, or pi_play? Maybe save this for later.
# 2) Add an option to use pi_cnn int he existing function instead of actionlist - it should have a maximum length parameter. Maybe save this for later.
# 3) Add new function to do an mcts batch from the current state and get policies.


def _get_policy_img(policy, value=None, maxaction=None):
    img = np.empty(policy_img_shape)
    img[:] = np.nan

    if policy is None:
        return img, None

    actionmap = {0: (0,1), # Move Up
                 1: (0,2), # Move Right
                 2: (1,1), # Move Down
                 3: (0,0), # Move Left
                 4: (2,0), # noop?
                 5: (2,1)} # interact

    for i, policyprob in enumerate(policy):
        img[actionmap[i]] = policyprob
    if value is not None:
        img[2,2] = value

    if maxaction is None:
        maxprob = np.argmax(policy)
        maxpos = actionmap[maxprob]
    else:
        maxpos = actionmap[maxaction]
    #plt.plot(x, y, 'wo')

    return img, maxpos


def plot_policies(ax, pi_cnns=None, V_cnns=None, pi_trees=None, pi_tree_scores=None, Qs=None, Ys=None, pi_plays=None):
    img_cnn = None
    maxpos_cnns = []
    img_pi_tree = None
    maxpos_pi_trees = []
    img_Q = None
    maxpos_Qs = []
    img_Y = None
    maxpos_Ys = []
    img_pi_play = None
    maxpos_pi_plays = []

    num_players = None

    def normalize_policies(policies):
        policies_scaled = [0] * len(policies)
        for i, policy in enumerate(policies):
            if policy is None:
                policies_scaled[i] = None
            elif sum(policy) == 0:
                policies_scaled[i] = [0] * len(policy)
            else:
                policies_scaled[i] = [policy_a / sum(policy) for policy_a in policy]

        return policies_scaled

    # We will get num_players from whichever list is given (assume all lists are of same length)
    if pi_cnns is not None:
        num_players = len(pi_cnns)
        (img_cnns, maxpos_cnns) = zip(*[_get_policy_img(pi_cnn, V_cnn) for (pi_cnn, V_cnn) in zip(pi_cnns, V_cnns)])
        img_cnn = np.concatenate(img_cnns, axis=0)

    if pi_trees is not None:
        num_players = len(pi_trees)

        #pi_tree_scores_scaled = [[pi_tree_score_a / sum(pi_tree_score) for pi_tree_score_a in pi_tree_score] if sum(pi_tree_score) > 0 else [0]*len(pi_tree_score) for pi_tree_score in pi_tree_scores]
        pi_tree_scores_scaled = normalize_policies(pi_tree_scores)

        (img_pi_trees, maxpos_pi_trees) = zip(*[_get_policy_img(pi_tree_score_scaled, maxaction=pi_tree) for (pi_tree_score_scaled, pi_tree) in zip(pi_tree_scores_scaled, pi_trees)])
        img_pi_tree = np.concatenate(img_pi_trees, axis=0)

    if Qs is not None:
        num_players = len(Qs)
        (img_Qs, maxpos_Qs) = zip(*[_get_policy_img(Q) for Q in Qs])
        img_Q = np.concatenate(img_Qs, axis=0)

    if Ys is not None:
        num_players = len(Ys)

        #Ys_scaled = [[Ya / sum(Y) for Ya in Y] if sum(Y) > 0 else [0]*len(Y) for Y in Ys]
        Ys_scaled = normalize_policies(Ys)

        (img_Ys, maxpos_Ys) = zip(*[_get_policy_img(Y_scaled) for Y_scaled in Ys_scaled])
        img_Y = np.concatenate(img_Ys, axis=0)

    if pi_plays is not None:
        num_players = len(pi_plays)
        (img_pi_plays, maxpos_pi_plays) = zip(*[_get_policy_img(pi_play) for pi_play in pi_plays])
        img_pi_play = np.concatenate(img_pi_plays, axis=0)

    assert num_players is not None, "At least one policy must be specified"

    if img_cnn is None:
        img_cnn = np.empty((num_players * policy_img_shape[0], policy_img_shape[1]))
        img_cnn[:] = np.nan

    if img_pi_tree is None:
        img_pi_tree = np.empty((num_players * policy_img_shape[0], policy_img_shape[1]))
        img_pi_tree[:] = np.nan

    if img_Q is None:
        img_Q = np.empty((num_players * policy_img_shape[0], policy_img_shape[1]))
        img_Q[:] = np.nan

    if img_Y is None:
        img_Y = np.empty((num_players * policy_img_shape[0], policy_img_shape[1]))
        img_Y[:] = np.nan

    if img_pi_play is None:
        img_pi_play = np.empty((num_players * policy_img_shape[0], policy_img_shape[1]))
        img_pi_play[:] = np.nan

    img = np.concatenate((img_cnn, img_pi_tree, img_pi_play, img_Q, img_Y), axis=1)
    img = img[0:img.shape[0]-1, 0:img.shape[1]]

    current_cmap = matplotlib.cm.get_cmap()
    current_cmap.set_bad(color='black')

    im = ax.matshow(img)

    # Plot maximum points
    sizey = policy_img_shape[0]
    offsetx = 0
    if pi_cnns is not None:
        for i, maxpos in enumerate(maxpos_cnns):
            if maxpos is not None:
                offsety = i * sizey
                ax.plot(offsetx + maxpos[1], offsety + maxpos[0], 'wo')
    offsetx += policy_img_shape[1]

    if pi_trees is not None:
        for i, maxpos in enumerate(maxpos_pi_trees):
            if maxpos is not None:
                offsety = i * sizey
                ax.plot(offsetx + maxpos[1], offsety + maxpos[0], 'wo')
    offsetx += policy_img_shape[1]

    if pi_plays is not None:
        for i, maxpos in enumerate(maxpos_pi_plays):
            if maxpos is not None:
                offsety = i * sizey
                ax.plot(offsetx + maxpos[1], offsety + maxpos[0], 'wo')
    offsetx += policy_img_shape[1]

    if Qs is not None:
        for i, maxpos in enumerate(maxpos_Qs):
            if maxpos is not None:
                offsety = i * sizey
                ax.plot(offsetx + maxpos[1], offsety + maxpos[0], 'wo')
    offsetx += policy_img_shape[1]

    if Ys is not None:
        for i, maxpos in enumerate(maxpos_Ys):
            if maxpos is not None:
                offsety = i * sizey
                ax.plot(offsetx + maxpos[1], offsety + maxpos[0], 'wo')
    offsetx += policy_img_shape[1]

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    im.set_clim(0, 1)


# This runs a partial episode.
# It should work essentially the same as run_episode_generic_team (in running.py), although some (hopefully minor) details may be different
#
# action_choice:
#  0 = use action list
#  1 = according to pi_play probability
#
# Here, each step consists of all players taking a turn. This may be different from the general terminology
# where each player's turn constitutes a step.
#
# pi_cnn_source:
#  0 = directly from cnn
#  1 = value used in pi_tree calculation
def doenvsteps(envs, actionlist=None, do_mcts_batch=None, action_choice=0, maxsteps=None, printsteps=False, dorender=False, get_pi_cnn=None, get_tree_and_play_policies=None, pi_cnn_source=0, figsize=figsize):
    if get_tree_and_play_policies is not None and do_mcts_batch is None:
        assert False, "do_mcts_batch is required to plot tree and play policies"

    pi_play_selector = ptan.actions.ProbabilityActionSelector()

    num_players = len(envs[0].get_agents())
    if maxsteps is not None:
        testlen = maxsteps
    elif actionlist is not None:
        testlen = max(len(agentactionlist) for agentactionlist in actionlist)
    else:
        assert False, "either maxsteps or actionlist must be given"

    # if dorender:
    #     print('***Initial state')
    #     envs[0].render()

    for stepnum in range(testlen):
        if printsteps:
            print('\n***Step {0}'.format(stepnum))
        if dorender:
            envs[-1].render()

        #policyinfo = [{}] * num_players
        pi_cnns = [None] * num_players
        V_cnns = [0] * num_players
        pi_trees = [None] * num_players
        pi_tree_scores = [None] * num_players
        Qs = [None] * num_players
        Ys = [None] * num_players
        pi_plays = [None] * num_players
        for player in range(num_players):
            for env in envs:
                assert env.get_turnn() == player

            if do_mcts_batch is not None:
                do_mcts_batch(envs)

            if get_pi_cnn is not None:
                #pi_cnns[player], V_cnns[player] = get_pi_cnn(envs)
                pi_cnns_from_cnn, V_cnns[player] = get_pi_cnn(envs)
                if pi_cnn_source == 0:
                    pi_cnns[player] = pi_cnns_from_cnn
            if get_tree_and_play_policies is not None:
                pi_trees[player], pi_tree_scores[player], Qs[player], Ys[player], pi_plays[player], pi_cnns_from_tree = get_tree_and_play_policies(envs)
                if pi_cnn_source == 1:
                    pi_cnns[player] = pi_cnns_from_tree # Note this may include added noise

            if action_choice == 0:
                assert actionlist is not None, "actionlist must be given if action_choice==0"
                playeractionlist = actionlist[player]
                if stepnum >= len(playeractionlist):
                    action = envs[0].noop_action()
                else:
                    action = playeractionlist[stepnum]
            elif action_choice == 1:
                assert get_tree_and_play_policies is not None, "get_tree_and_play_policies must be given if action_choice==1"
                action = pi_play_selector(np.array([pi_plays[player]]))[0]
            else:
                assert False, "Invalid action_choice"

            obs_return, rew, done, outer_info = do_env_step(envs, action)

            autstate = outer_info['automaton_states']
            if printsteps:
                print('Agent: {0}  Action: {1}  Reward: {2}  Aut State: {3}'.format(player, action, rew, autstate))

            if done:
                if printsteps:
                    print('Done!')
                break



        if get_pi_cnn is not None or get_tree_and_play_policies is not None:
            fig = plt.figure(figsize=figsize)
            ax = plt.gca()
            if get_pi_cnn is None:
                plot_policies(ax, pi_trees=pi_trees, pi_tree_scores=pi_tree_scores, Qs=Qs, Ys=Ys, pi_plays=pi_plays)
            elif get_tree_and_play_policies is None:
                plot_policies(ax, pi_cnns=pi_cnns, V_cnns=V_cnns)
            else:
                plot_policies(ax, pi_cnns=pi_cnns, V_cnns=V_cnns, pi_trees=pi_trees, pi_tree_scores=pi_tree_scores, Qs=Qs, Ys=Ys, pi_plays=pi_plays)
            plt.show(fig)

        if done:
            break

    if dorender:
        print('***Final state')
        envs[-1].render()

    finishedrun = stepnum == testlen-1

    return stepnum, done, player, rew, autstate, finishedrun


# If do_mcts_batch is specified, a batch will be run for each player before getting the policies
def get_all_player_policies(envs, do_mcts_batch=None, get_pi_cnn=None, get_tree_and_play_policies=None, pi_cnn_source=0):
    num_players = len(envs[0].get_agents())
    noop = envs[0].noop_action()

    # if get_pi_cnn is None:
    #     pi_cnns = None
    #     V_cnns = None
    # else:
    pi_cnns = [None] * num_players
    V_cnns = [0] * num_players

    # if get_tree_and_play_policies is None:
    #     pi_trees = None
    #     pi_tree_scores = None
    #     Qs = None
    #     Ys = None
    #     pi_plays = None
    # else:
    pi_trees = [None] * num_players
    pi_tree_scores = [None] * num_players
    Qs = [None] * num_players
    Ys = [None] * num_players
    pi_plays = [None] * num_players

    for i in range(num_players):
        if do_mcts_batch is not None:
            do_mcts_batch(envs)

        if get_pi_cnn is not None:
            pi_cnns_from_cnn, V_cnns[i] = get_pi_cnn(envs)
            if pi_cnn_source == 0:
                pi_cnns[i] = pi_cnns_from_cnn

        if get_tree_and_play_policies is not None:
            pi_trees[i], pi_tree_scores[i], Qs[i], Ys[i], pi_plays[i], pi_cnns_from_tree = get_tree_and_play_policies(envs)
            if pi_cnn_source == 1:
                pi_cnns[i] = pi_cnns_from_tree  # Note this may include added noise

        for env in envs:
            env.step(noop)

    return pi_cnns, V_cnns, pi_trees, pi_tree_scores, Qs, Ys, pi_plays


def get_and_plot_policies(envs, do_mcts_batch=None, get_pi_cnn=None, get_tree_and_play_policies=None, figsize=figsize, pi_cnn_source=0):
    pi_cnns, V_cnns, pi_trees, pi_tree_scores, Qs, Ys, pi_plays = get_all_player_policies(envs, do_mcts_batch=do_mcts_batch, get_pi_cnn=get_pi_cnn, get_tree_and_play_policies=get_tree_and_play_policies, pi_cnn_source=pi_cnn_source)

    fig = plt.figure(figsize=figsize)
    ax = plt.gca()
    plot_policies(ax, pi_cnns, V_cnns, pi_trees, pi_tree_scores, Qs, Ys, pi_plays)
    plt.show(fig)
