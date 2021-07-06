import pandas as pd
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import seaborn as sns
from utils import *

sns.set()
sns.despine()
sns.set_context("paper", rc={"font.size":18,"axes.labelsize":18,"xtick.labelsize": 16,"ytick.labelsize": 16,"legend.fontsize": 16})
sns.set_style('white', {'axes.edgecolor': "0.5","pdf.fonttype": 42})
plt.gcf().subplots_adjust(bottom=0.15)

provide_recs = False
recommendation_time = 500
multi_CE = False
single_ce = not multi_CE
rand_prob = False
episodes = 10000

#  Specify Agent type here
agent1_opt_crit = 'ESR'
agent2_opt_crit = 'SER'

# specify games to be plotted in the for-loop condition below
# ['game1', 'game2noM', 'game2noR', 'game3', 'game4']:
for game in ['game1', 'game2noM', 'game2noR', 'game4','game5']:
    for opt_init in [False]: #[True, False]:
        path_data = f'data/{game}'
        path_plots = f'plots/{game}'

        if opt_init:
            path_data += '/opt_init'
            path_plots += '/opt_init'
        else:
            path_data += '/zero_init'
            path_plots += '/zero_init'

        if rand_prob:
            path_data += '/opt_rand'
            path_plots += '/opt_rand'
        else:
            path_data += '/opt_eq'
            path_plots += '/opt_eq'

        path_data += f'/row{agent1_opt_crit}_col{agent2_opt_crit}'
        path_plots += f'/row{agent1_opt_crit}_col{agent2_opt_crit}'

        print(path_plots)
        mkdir_p(path_plots)

        info = ''
        if provide_recs:
            info += f'CE_{recommendation_time}_'
            if single_ce:
                info += 'single'
            if multi_CE:
                info += 'multi'
        else:
            info += 'NE'

        df1 = pd.read_csv(f'{path_data}/agent1_{info}.csv')
        df2 = pd.read_csv(f'{path_data}/agent2_{info}.csv')
        df1 = df1.iloc[::5, :]
        df2 = df2.iloc[::5, :]

        ax = sns.lineplot(x='Episode', y='Payoff', linewidth=2.0, data=df1, ci='sd', label='Agent 1')
        ax = sns.lineplot(x='Episode', y='Payoff', linewidth=2.0, data=df2, ci='sd', label='Agent 2')
        ax.set(ylabel='Scalarised payoff')
        ax.set_ylim(1, 18)
        ax.set_xlim(0, episodes)
        plot_name = f"{path_plots}/{game}_returns_{info}"

        plt.savefig(plot_name + ".pdf")
        plt.clf()

        # Plot the action probabilities for Agent 1
        df1 = pd.read_csv(f'{path_data}/agent1_probs_{info}.csv')
        df1 = df1.iloc[::5, :]


        if game == 'game2noM':
            label2 = 'R'
            label1 = 'L'
        else:
            label2 = 'M'
            label1 = 'L'

        if game == 'chicken':
            label1 = "S"
            label2 = "D"


        ax = sns.lineplot(x='Episode', y='Action 1', linewidth=2.0, data=df1, ci='sd', label=label1)
        ax = sns.lineplot(x='Episode', y='Action 2', linewidth=2.0, data=df1, ci='sd', label=label2)

        if game in ['game1', 'game4']:
            ax = sns.lineplot(x='Episode', y='Action 3', linewidth=2.0, data=df1, ci='sd', label='R')
        ax.set(ylabel='Action probability')
        # if provide_recs:
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(0, episodes)
        plot_name = f"{path_plots}/{game}_{agent1_opt_crit}_A1_{info}"

        plt.savefig(plot_name + ".pdf")
        plt.clf()

        # Plot the action probabilities for Agent 2
        df1 = pd.read_csv(f'{path_data}/agent2_probs_{info}.csv')
        df1 = df1.iloc[::5, :]

        ax = sns.lineplot(x='Episode', y='Action 1', linewidth=2.0, data=df1, ci='sd', label=label1)
        ax = sns.lineplot(x='Episode', y='Action 2', linewidth=2.0, data=df1, ci='sd', label=label2)
        if game in ['game1', 'game4']:
            ax = sns.lineplot(x='Episode', y='Action 3', linewidth=2.0, data=df1, ci='sd', label='R')
        ax.set(ylabel='Action probability')
        #if provide_recs:
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(0, episodes)
        plot_name = f"{path_plots}/{game}_{agent2_opt_crit}_A2_{info}"

        plt.savefig(plot_name + ".pdf")
        plt.clf()

        # Plot distribution over the joint action space, for the last 1k episodes, over all trials
        df = pd.read_csv(f'{path_data}/states_{info}.csv', header=None)

        if game == 'game2noM':
            x_axis_labels = ["L", "R"]
            y_axis_labels = ["L", "R"]

        if game == 'chicken':
            x_axis_labels = ["S", "D"]
            y_axis_labels = ["S", "D"]

        if game in ['game2noR', 'game3']:
            x_axis_labels = ["L", "M"]
            y_axis_labels = ["L", "M"]

        if game in ['game1', 'game4']:
            x_axis_labels = ["L", "M", "R"]
            y_axis_labels = ["L", "M", "R"]

        ax = sns.heatmap(df, annot=True, cmap="YlGnBu", vmin=0, vmax=1, xticklabels=x_axis_labels, yticklabels=y_axis_labels)
        plot_name = f"{path_plots}/{game}_states_{info}"

        plt.savefig(plot_name + ".pdf")
        plt.clf()

