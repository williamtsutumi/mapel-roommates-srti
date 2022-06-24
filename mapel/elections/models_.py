#!/usr/bin/env python

from collections import Counter
from typing import Union

import numpy as np

import mapel.elections.models.euclidean as euclidean
import mapel.elections.models.group_separable as group_separable
import mapel.elections.models.guardians as guardians
import mapel.elections.models.impartial as impartial
import mapel.elections.models.mallows as mallows
import mapel.elections.models.single_crossing as single_crossing
import mapel.elections.models.single_peaked as single_peaked
import mapel.elections.models.urn_model as urn_model
from mapel.main._glossary import *
from mapel.elections.models.preflib import generate_preflib_votes
import mapel.elections.models.field_experiment as fe
import mapel.elections.models.didi as didi


def generate_approval_votes(model_id: str = None, num_candidates: int = None,
                            num_voters: int = None, params: dict = None) -> Union[list, np.ndarray]:

    main_models = {'impartial_culture': impartial.generate_approval_ic_votes,
                   'ic': impartial.generate_approval_ic_votes,
                   'id': impartial.generate_approval_id_votes,
                   'resampling': mallows.generate_approval_resampling_votes,
                   'noise_model': mallows.generate_approval_noise_model_votes,
                   'urn': urn_model.generate_approval_urn_votes,
                   'urn_model': urn_model.generate_approval_urn_votes,
                   'euclidean': euclidean.generate_approval_euclidean_votes,
                   'disjoint_resampling': mallows.generate_approval_disjoint_resamplin_votes,
                   'vcr': euclidean.generate_approval_vcr_votes,
                   'truncated_mallows': mallows.generate_approval_truncated_mallows_votes,
                   'truncated_urn': urn_model.generate_approval_truncated_urn_votes,
                   'moving_resampling': mallows.generate_approval_moving_resampling_votes,
                   'simplex_resampling': mallows.generate_approval_simplex_resampling_votes,
                   'anti_pjr': mallows.approval_anti_pjr_votes,
                   'partylist': mallows.approval_partylist_votes,
                   'field': fe.generate_approval_field_votes,
                   }

    if model_id in main_models:
        return main_models.get(model_id)(num_voters=num_voters, num_candidates=num_candidates,
                                         params=params)
    elif model_id in ['approval_full']:
        return impartial.generate_approval_full_votes(num_voters=num_voters,
                                                      num_candidates=num_candidates)
    elif model_id in ['approval_empty']:
        return impartial.generate_approval_empty_votes(num_voters=num_voters)

    elif model_id in APPROVAL_FAKE_MODELS:
        return [model_id, num_candidates, num_voters, params]
    else:
        print("No such election model_id!", model_id)
        return []


def generate_ordinal_votes(model_id: str = None, num_candidates: int = None, num_voters: int = None,
                           params: dict = None) -> Union[list, np.ndarray]:

    if model_id in LIST_OF_PREFLIB_MODELS:
        return generate_preflib_votes(model_id=model_id, num_candidates=num_candidates,
                                      num_voters=num_voters, params=params)

    naked_models = {'impartial_culture': impartial.generate_ordinal_ic_votes,
                    'iac': impartial.generate_impartial_anonymous_culture_election,
                    'conitzer': single_peaked.generate_ordinal_sp_conitzer_votes,
                    'spoc_conitzer': single_peaked.generate_ordinal_spoc_conitzer_votes,
                    'walsh': single_peaked.generate_ordinal_sp_walsh_votes,
                    'real_identity': guardians.generate_real_identity_votes,
                    'real_uniformity': guardians.generate_real_uniformity_votes,
                    'real_antagonism': guardians.generate_real_antagonism_votes,
                    'real_stratification': guardians.generate_real_stratification_votes}

    euclidean_models = {'1d_interval': euclidean.generate_ordinal_euclidean_votes,
                        '1d_gaussian': euclidean.generate_ordinal_euclidean_votes,
                        '1d_one_sided_triangle': euclidean.generate_ordinal_euclidean_votes,
                        '1d_full_triangle': euclidean.generate_ordinal_euclidean_votes,
                        '1d_two_party': euclidean.generate_ordinal_euclidean_votes,
                        '2d_disc': euclidean.generate_ordinal_euclidean_votes,
                        '2d_square': euclidean.generate_ordinal_euclidean_votes,
                        '2d_gaussian': euclidean.generate_ordinal_euclidean_votes,
                        '3d_gaussian': euclidean.generate_ordinal_euclidean_votes,
                        '3d_cube': euclidean.generate_ordinal_euclidean_votes,
                        '4d_cube': euclidean.generate_ordinal_euclidean_votes,
                        '5d_cube': euclidean.generate_ordinal_euclidean_votes,
                        '10d_cube': euclidean.generate_ordinal_euclidean_votes,
                        '2d_sphere': euclidean.generate_ordinal_euclidean_votes,
                        '3d_sphere': euclidean.generate_ordinal_euclidean_votes,
                        '4d_sphere': euclidean.generate_ordinal_euclidean_votes,
                        '5d_sphere': euclidean.generate_ordinal_euclidean_votes,
                        '4d_ball': euclidean.generate_ordinal_euclidean_votes,
                        '5d_ball': euclidean.generate_ordinal_euclidean_votes,
                        '2d_grid': euclidean.generate_elections_2d_grid,
                        'euclidean': euclidean.generate_ordinal_euclidean_votes}

    party_models = {'1d_gaussian_party': euclidean.generate_1d_gaussian_party,
                    '2d_gaussian_party': euclidean.generate_2d_gaussian_party,
                    'walsh_party': single_peaked.generate_sp_party,
                    'conitzer_party': single_peaked.generate_sp_party,
                    'mallows_party': mallows.generate_mallows_party,
                    'ic_party': impartial.generate_ic_party
                    }

    single_param_models = {'urn_model': urn_model.generate_urn_votes,
                            'urn': urn_model.generate_urn_votes,
                           'group-separable':
                               group_separable.generate_ordinal_group_separable_votes,
                           'single-crossing': single_crossing.generate_ordinal_single_crossing_votes,
                           'weighted_stratification': impartial.generate_weighted_stratification_votes,
                           'didi': didi.generate_didi_votes}

    double_param_models = {'mallows': mallows.generate_mallows_votes,
                           'norm-mallows': mallows.generate_mallows_votes,
                           'norm-mallows_with_walls':  mallows.generate_norm_mallows_with_walls_votes,
                           'norm-mallows_mixture': mallows.generate_norm_mallows_mixture_votes,
                           'walsh_mallows': single_peaked.generate_walsh_mallows_votes,
                           'conitzer_mallows': single_peaked.generate_conitzer_mallows_votes,}

    if model_id in naked_models:
        votes = naked_models.get(model_id)(num_voters=num_voters,
                                           num_candidates=num_candidates)

    elif model_id in euclidean_models:
        votes = euclidean_models.get(model_id)(num_voters=num_voters,
                                               num_candidates=num_candidates,
                                               model=model_id, params=params)

    elif model_id in party_models:
        votes = party_models.get(model_id)(num_voters=num_voters,
                                           num_candidates=num_candidates,
                                           model=model_id, params=params)

    elif model_id in single_param_models:
        votes = single_param_models.get(model_id)(num_voters=num_voters,
                                                  num_candidates=num_candidates,
                                                  params=params)

    elif model_id in double_param_models:
        votes = double_param_models.get(model_id)(num_voters, num_candidates, params)

    elif model_id in LIST_OF_FAKE_MODELS:
        votes = [model_id, num_candidates, num_voters, params]
    else:
        votes = []
        print("No such election model_id!", model_id)

    if model_id not in LIST_OF_FAKE_MODELS:
        votes = [[int(x) for x in row] for row in votes]

    return np.array(votes)


def store_votes_in_a_file(election, model_id, num_candidates, num_voters,
                          params, path, ballot, votes=None):
    """ Store votes in a file """
    if votes is None:
        votes = election.votes

    with open(path, 'w') as file_:
        if model_id in NICE_NAME:
            file_.write("# " + NICE_NAME[model_id] + " " + str(params) + "\n")
        else:
            file_.write("# " + model_id + " " + str(params) + "\n")

        file_.write(str(num_candidates) + "\n")

        for i in range(num_candidates):
            file_.write(str(i) + ', c' + str(i) + "\n")

        c = Counter(map(tuple, votes))
        counted_votes = [[count, list(row)] for row, count in c.items()]
        counted_votes = sorted(counted_votes, reverse=True)

        file_.write(str(num_voters) + ', ' + str(num_voters) + ', ' +
                    str(len(counted_votes)) + "\n")

        if ballot == 'approval':
            for i in range(len(counted_votes)):
                file_.write(str(counted_votes[i][0]) + ', {')
                for j in range(len(counted_votes[i][1])):
                    file_.write(str(int(counted_votes[i][1][j])))
                    if j < len(counted_votes[i][1]) - 1:
                        file_.write(", ")
                file_.write("}\n")

        elif ballot == 'ordinal':
            for i in range(len(counted_votes)):
                file_.write(str(counted_votes[i][0]) + ', ')
                for j in range(len(counted_votes[i][1])):
                    file_.write(str(int(counted_votes[i][1][j])))
                    if j < len(counted_votes[i][1]) - 1:
                        file_.write(", ")
                file_.write("\n")


# # # # # # # # # # # # # # # #
# LAST CLEANUP ON: 16.05.2022 #
# # # # # # # # # # # # # # # #
