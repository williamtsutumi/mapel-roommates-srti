import mapel.roommates as mapel
import random
import math

size = 5
num_agents = 14 


experiment = mapel.prepare_online_roommates_experiment()


experiment.add_instance(culture_id='id',
                        label='ID',
                        num_agents=num_agents,
                        color='red', marker='+', ms = 125)
experiment.add_instance(culture_id='symmetric',
                        label='MA',
                        num_agents=num_agents,
                        color='orange', marker='+', ms = 125)
experiment.add_instance(culture_id='asymmetric',
                        label='MD',
                        num_agents=num_agents,
                        color='blue', marker='+', ms = 125)
experiment.add_instance(culture_id='chaos',
                        label='CH',
                        num_agents=num_agents,
                        color='black', marker='+', ms = 125)
    

sr_families = [
    { 'culture': 'expectation', 'label': 'Expectation_0.2', 'params': {'std': .2}, 'color': 'red', 'marker': 'o' },
    { 'culture': 'expectation', 'label': 'Expectation_0.4', 'params': {'std': .4}, 'color': 'red', 'marker': 'o' },

    { 'culture': 'ic', 'label': 'Impartial', 'params': {}, 'color': 'magenta', 'marker': 'o' },

    { 'culture': 'group_ic', 'label': 'Group_ic_0.25', 'params': {'proportion': 0.25}, 'color': 'darkviolet', 'marker': 'o' },
    { 'culture': 'group_ic', 'label': 'Group_ic_0.50', 'params': {'proportion': 0.50}, 'color': 'darkviolet', 'marker': 'o' },

    { 'culture': 'attributes', 'label': 'Attributes_2D', 'params': {'dim': 2}, 'color': 'yellow', 'marker': 'o' },
    { 'culture': 'attributes', 'label': 'Attributes_5D', 'params': {'dim': 5}, 'color': 'yellow', 'marker': 'o' },

    { 'culture': 'fame', 'label': 'Fame_0.2', 'params': {'radius': .2}, 'color': 'lime', 'marker': 'o' },
    { 'culture': 'fame', 'label': 'Fame_0.4', 'params': {'radius': .4}, 'color': 'lime', 'marker': 'o' },

    { 'culture': 'reverse_euclidean', 'label': 'Reverse_euclidean_0.05', 'params': {'proportion': .05}, 'color': 'sienna', 'marker': 'o' },
    { 'culture': 'reverse_euclidean', 'label': 'Reverse_euclidean_0.15', 'params': {'proportion': .15}, 'color': 'sienna', 'marker': 'o' },
    { 'culture': 'reverse_euclidean', 'label': 'Reverse_euclidean_0.25', 'params': {'proportion': .25}, 'color': 'sienna', 'marker': 'o' },

    { 'culture': 'mallows_euclidean', 'label': 'Mallows_euclidean_0.2', 'params': {'phi': .2}, 'color': 'cyan', 'marker': 'o' },
    { 'culture': 'mallows_euclidean', 'label': 'Mallows_euclidean_0.4', 'params': {'phi': .4}, 'color': 'cyan', 'marker': 'o' },
    { 'culture': 'mallows_euclidean', 'label': 'Mallows_euclidean_0.6', 'params': {'phi': .6}, 'color': 'cyan', 'marker': 'o' },

    { 'culture': 'norm-mallows', 'label': 'Norm_mallows_0.2', 'params': {'normphi': .2}, 'color': 'green', 'marker': 'o' },
    { 'culture': 'norm-mallows', 'label': 'Norm_mallows_0.4', 'params': {'normphi': .4}, 'color': 'green', 'marker': 'o' },
    { 'culture': 'norm-mallows', 'label': 'Norm_mallows_0.6', 'params': {'normphi': .6}, 'color': 'green', 'marker': 'o' },
    { 'culture': 'norm-mallows', 'label': 'Norm_mallows_0.8', 'params': {'normphi': .8}, 'color': 'green', 'marker': 'o' },

    { 'culture': 'malasym', 'label': 'Mallows_md_0.2', 'params': {'normphi': .2}, 'color': 'blue', 'marker': 'o' },
    { 'culture': 'malasym', 'label': 'Mallows_md_0.4', 'params': {'normphi': .4}, 'color': 'blue', 'marker': 'o' },
    { 'culture': 'malasym', 'label': 'Mallows_md_0.6', 'params': {'normphi': .6}, 'color': 'blue', 'marker': 'o' },
] # 22 cultures

for family in srti_families:
    experiment.add_family(culture_id=family['culture'],
                        family_id=family['label'],
                        label=family['label'],
                        params=family['params'],
                        num_agents=num_agents, size=size,
                        color=family['color'], marker=family['marker'])

textual = ['ID', 'MD', 'MA', 'CH']
figsize = (8,8)

saveas = 'experiment'
experiment.compute_distances(distance_id='mutual_attraction', num_threads=5)
experiment.embed_2d(embedding_id='kk')

experiment.print_map_2d(show=True, textual=textual, saveas=saveas, legend_pos=(0,0), figsize=figsize)


experiment.compute_feature(feature_id='dist_from_id_1')
experiment.compute_feature(feature_id='mutuality')
experiment.compute_feature(feature_id='avg_num_of_bps_for_rand_matching')
experiment.compute_feature(feature_id='num_of_bps_min_weight')

experiment.print_map_2d_colored_by_feature(show=True, textual=textual, rounding=0, feature_id='dist_from_id_1', saveas=saveas+'_rankdistortion', figsize=figsize)
experiment.print_map_2d_colored_by_feature(show=True, textual=textual, rounding=0, feature_id='mutuality', saveas=saveas+'_mutuality', figsize=figsize)
experiment.print_map_2d_colored_by_feature(show=True, textual=textual, rounding=0, feature_id='avg_num_of_bps_for_rand_matching', saveas=saveas+'_avgbps', figsize=figsize)
experiment.print_map_2d_colored_by_feature(show=True, textual=textual, rounding=0, feature_id='num_of_bps_min_weight', saveas=saveas+'_numbps_minweight', figsize=figsize, scale='log')
