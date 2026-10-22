#
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileCopyrightText: Copyright © 2026 University of Bern
#
# SPDX-FileContributor: Lisa Fournier <lisa.fournier@idiap.ch>
# SPDX-FileContributor: Lisa Fournier <lisa.fournier@unibe.ch>
#
# SPDX-License-Identifier: GPL-3.0-only
#
import os
import pickle  # nosec

import numpy as np


def load_and_combine_integrated_CLIP(path_to_integrated_CLIP, 
                                     prefix="", 
                                     cols=None,
                                     RBP_list=None):
    """Load and combine the integrated CLIP for further analysis.

    Args:
        path_to_integrated_CLIP (string): path to the integrated CLIP (.pkl files).
        iso_type (str, optional): Either proximal or distal. Defaults to 'proximal'.
        cols: List of columns to keep from the df_results. Allows to save a significant amount
        of memory.

    Returns:
        dict: Returns a dictionnary with the results from the integration for each processed RBP.
        If iso_type is proximal, returns results when RBP cross-links to the proximal isoform. If iso_type
        is distal, returns results when RBP cross-links to the distal isoform.
        RBP names in keys and pd.DataFrame in values.
    """

    df_results = {}
    for filename in os.listdir(path_to_integrated_CLIP):
        if filename.endswith(prefix + ".pkl"):
            if RBP_list is not None and not any(rbp in filename for rbp in RBP_list):
                continue
            if os.path.getsize(os.path.join(path_to_integrated_CLIP, filename)) > 0:
                with open(os.path.join(path_to_integrated_CLIP, filename), "rb") as f:
                    df_result = pickle.load(f)  # nosec
                    if cols is None:
                        df_results[filename.split("_result")[0]] = df_result  # nosec
                    else:
                        df_results[filename.split("_result")[0]] = df_result[
                            cols
                        ]  # nosec
    return df_results


def is_auto_cross_linked(
    df_results, rbp_name, nb_nucleotides_downstream=-000, region_length=50
):
    """Determines whether a given rbp auto-binds to its own isoform.

    Args:
        df_results (dict): dictionnary with RBP names in keys and pd.DataFrame in values. Output of the function
        *load_and_combine_integrated_CLIP*. Choose if you provide the results of the integration for
        cross-linking on the proximal isoforms or on the distal ones.
        rbp_name (str): Name on the RBP you want to check the auto-cross links for.
        nb_nucleotides_upstream (int, optional): Number of nucleotide upstream of the PAS we analyze. Defaults to 1000.
        region_length (int, optional): Length of the regions we analyze. Defaults to 50.

    Returns:
        dict: dictionnary with all replicates for the given RBP in keys, and regions of auto cross-links in values.
    """
    rbp_replicates = [key for key in df_results.keys() if key.startswith(rbp_name)]

    regions_auto_cross_linked_per_replicates = {}
    for rbp_replicate in rbp_replicates:
        df_results[rbp_replicate]["auto_cross_linked"] = df_results[rbp_replicate][
            "cross_link"
        ].apply(
            lambda list_cross_links: np.any(
                [rbp.startswith(rbp_name) for rbp in list_cross_links]
            )
        )
        regions = list(
            df_results[rbp_replicate][
                df_results[rbp_replicate]["auto_cross_linked"] is True
            ].index
        )
        regions_auto_cross_linked = []
        for region in regions:
            nb = int(region.split("Region")[1])
            region_interval = (
                "["
                + str(nb_nucleotides_downstream + region_length * (nb - 1))
                + ":"
                + str(nb_nucleotides_downstream + region_length * nb)
                + "]"
            )
            regions_auto_cross_linked.append((region, region_interval))
        regions_auto_cross_linked_per_replicates[
            rbp_replicate
        ] = regions_auto_cross_linked
    return regions_auto_cross_linked_per_replicates
