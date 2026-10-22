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
import pandas as pd
from gtfparse import read_gtf
from scipy.stats import norm


def create_regions_around_PAS(
    start: int,
    end: int,
    strand: str,
    nt_after: int = 200,
    nt_before: int = -1000,
    length_region: int = 50,
):
    """Create intervals corresponding to regions of the wanted size towards the polyadenylation site
    (PAS) of a given isoform.

    Args:
        start: start nucleotide position
        end: end nucleotide position
        strand: strand for the given isoform. Can be + or -.
        nt_after: Number of nucleotides downstream the PAS we want to analyze (5' -> 3').
        nt_before: Number of nucleotides upstream the PAS we want to analyze (5' -> 3').
        length_region: Length of the regions to ananlyze.

    Returns:
        pd.arrays.IntervalArray: Returns an array of Intervals corresponding to the regions surrounding the PAS.
    """
    intervals = []
    nb_regions = int(np.floor((nt_after - nt_before) / length_region))
    if strand == "+":
        for i in range(nb_regions):
            intervals.append(
                (
                    nt_before + end + i * length_region,
                    nt_before + end + (i + 1) * length_region,
                )
            )
    if strand == "-":
        for i in range(nb_regions):
            intervals.append(
                (
                    -nt_before + start - (i + 1) * length_region,
                    -nt_before + start - i * length_region,
                )
            )
    return intervals


def is_cross_linked_per_region(
    df_transcripts: pd.DataFrame,
    RBP_regions: dict,
    regions_col: str = "regions",
    strand_col: str = "strand",
    chrom_col: str = "seqname",
):
    """Add to df_transcripts information on whether the RBP cross-link for each region surrounding the isoform.

    Args:
        df_transcripts: DataFrame containing transcripts information.
        Basically a .gtf file transformed into a DataFrame.
        The DataFrame contains isoforms in rows and a column for the regions around the PAS.
        The regions are pd.arrays.IntervalArray outputed by create_regions_around_PAS.
        RBP_regions (dict of pd.arrays.IntervalArray): Intervals representing the binding regions of
        a RBP in values, and strands in keys.
        regions_col: column name of the DataFrane df_transcripts containing the regions information
        (pd.arrays.IntervalArray).
        strand_col: column name bearing the strand information

    Returns:
        pd.DataFrame: The updated df_transcripts.
    """

    nb_regions = len(df_transcripts.head(1)["regions"].values[0])
    for i in range(nb_regions):
        df_transcripts["Region" + str(i + 1)] = df_transcripts.apply(
            lambda x: np.logical_and(
                RBP_regions[x[strand_col]][x[chrom_col]] > x[regions_col][i][0],
                RBP_regions[x[strand_col]][x[chrom_col]] <= x[regions_col][i][1],
            ).any(),
            axis=1,
        )

    return df_transcripts


# old version of the fisher table -- updated on 17.01.2024
def get_fisher_table_old(
    cross_link_list: list,
    no_cross_link_list: list,
    significant_isoforms_group: list,
    significant_isoforms_all: list = [],
):
    """Returns contigency table used for fisher count test testing for the association between cross-link events and
    significant shifts.

    Args:
        cross_link_list: list of isoforms exhibiting a cross-link event with the RBP
        no_cross_link_list: list of isoforms exhibiting no cross-link event with the RBP
        significant_isoforms_group: group of isoforms being part of a pair exhibiting
        significant proximal/distal shift between two conditions.
        This is the group that will be compared to the isoforms exhibiting no cross-link event.
        significant_isoforms_all: All isoforms exhibiting a significant proximal/distal shift.
        This argument is useful in case you want to analyze the group *significant_isoform_group* only,
        even if other isoforms exhibit a significant shift.
        In this case, put the list of all isoforms exhibiting a shift, and the ones that are not of
        interested will be removed.

    Returns:
        np.array: 2D contingency table
                [[number of isoforms exhibiting a cross link event with the RBP and a significant shift,
                number of isoforms exhibiting a cross link event with the RBP but no significant shift],
                [number of isoforms exhibiting no cross link event with the RBP and a significant shift,
                number of isoforms exhibiting no cross link event with the RBP but no significant shift]]
    """
    if len(significant_isoforms_all) > 0:
        significant_iso_to_remove = [
            iso
            for iso in significant_isoforms_all
            if iso not in significant_isoforms_group
        ]
        cross_link_list = [
            iso for iso in cross_link_list if iso not in significant_iso_to_remove
        ]
        no_cross_link_list = [
            iso for iso in no_cross_link_list if iso not in significant_iso_to_remove
        ]
    cross_link_sig_shift = len(
        list(set(cross_link_list) & set(significant_isoforms_group))
    )
    cross_link_no_sig_shift = len(cross_link_list) - cross_link_sig_shift
    no_cross_link_sig_shift = len(
        list(set(no_cross_link_list) & set(significant_isoforms_group))
    )
    no_cross_link_no_sig_shift = len(no_cross_link_list) - no_cross_link_sig_shift
    return np.array(
        [
            [cross_link_sig_shift, cross_link_no_sig_shift],
            [no_cross_link_sig_shift, no_cross_link_no_sig_shift],
        ]
    )


def get_fisher_table(
    cross_link_list: list,
    significant_isoforms_group: list,
    not_significant_isoforms_group: list,
):
    """Returns contigency table used for fisher count test testing for the association between cross-link events and
    significant shifts.

    Args:
        cross_link_list: list of isoforms exhibiting a cross-link event with the RBP
        no_cross_link_list: list of isoforms exhibiting no cross-link event with the RBP
        significant_isoforms_group: group of isoforms being part of a pair exhibiting
        significant proximal/distal shift between two conditions.
        This is the group that will be compared to the isoforms exhibiting no cross-link event.

    Returns:
        np.array: 2D contingency table
                [[number of isoforms exhibiting a cross link event with the RBP and a significant shift,
                number of isoforms exhibiting a cross link event with the RBP but no significant shift],
                [number of isoforms exhibiting no cross link event with the RBP and a significant shift,
                number of isoforms exhibiting no cross link event with the RBP but no significant shift]]
    """
    cross_link_sig_shift = len(
        [iso for iso in significant_isoforms_group if iso in cross_link_list]
    )
    no_cross_link_sig_shift = len(significant_isoforms_group) - cross_link_sig_shift
    cross_link_no_sig_shift = len(
        [iso for iso in not_significant_isoforms_group if iso in cross_link_list]
    )
    no_cross_link_no_sig_shift = (
        len(not_significant_isoforms_group) - cross_link_no_sig_shift
    )

    return np.array(
        [
            [cross_link_sig_shift, cross_link_no_sig_shift],
            [no_cross_link_sig_shift, no_cross_link_no_sig_shift],
        ]
    )


def add_cross_link_column(
    df_transcripts: pd.DataFrame,
    clip: pd.DataFrame,
    RBP_name: str,
    regions_col: str,
    start_col: str = "start",
    end_col: str = "end",
    strand_col: str = "strand",
):
    """From a DataFrame of transcripts with a column of pd.arrays.IntervalArrays depicting the regions around the PAS
    (output of *create_regions_around_PAS*), create a new column per region with True meaning there is a cross-link
    with this RBP in this region, and False meaning no cross-link with this RBP in this region.
    The columns will be named "RBP_name_Region(i)" with i, the number of the region with 1 being the region the most

    Args:
        df_transcripts: Table containing the transcripts is rows and their start, end and
        strand information in column.
        These correspond to the transcripts you want to check cross-links on.
        clip: Clip data for a given RBP. Basically a .gff file converted into a pd.DataFrame.
        RBP_name: Name of the given RBP.
        regions_col: Name of the column containing the regions information
        (created with *create_regions_around_PAS*)
        start_col: Name of the column containing the start information. Defaults to 'start'.
        end_col: Name of the column containing the end information Defaults to 'end'.
        strand_col: Name of the column containing the strand information. Defaults to 'strand'.

    Returns:
        pd.DataFrame: The updated dataframe of transcripts with the columns describing the cross-link events
        with the RBP for each region.
    """
    clip["link_regions"] = clip.apply(
        lambda row: pd.Interval(row[start_col], row[end_col]), axis=1
    )
    RBP_regions = {}
    for strand in clip[strand_col].value_counts().index:
        RBP_regions[strand] = pd.arrays.IntervalArray(
            clip[clip[strand_col] == strand]["link_regions"]
        )

    df_transcripts = is_cross_linked_per_region(
        df_transcripts, RBP_regions, regions_col
    )

    rename_cols = {}
    for col in df_transcripts.columns:
        if col.startswith("Region"):
            rename_cols[col] = RBP_name + "_" + col

    df_transcripts = df_transcripts.rename(columns=rename_cols)
    return df_transcripts


def get_cross_link_matrix(
    df_transcripts: pd.DataFrame,
    clip_files: list,
    RBP_unique_names: list,
    transcripts_to_keep: pd.DataFrame = None,
    df_transcripts_ID: str = "uniqueID",
    transcripts_to_keep_ID: str = "uniqueIDs",
    nt_after: int = -50,
    nt_before: int = -250,
    length_region: int = 200,
    path_to_save: str = None,
    prefix: str = "",
    start_col: str = "start",
    end_col: str = "end",
    strand_col: str = "strand",
):
    """From a FataFrame of transcripts with their start, end and strand information and
    a list of CLIP data files (.gff) along with their corresponding names,
    this function will create a matrix with transcripts in rows and in columns the cross-link information for each
    found RBP and each defined region.

    Args:
        df_transcripts: DataFrame with transcripts in rows and their respective start, end and strand
        informations in columns.
        clip_files: list of the clip filepaths to process.
        RBP_unique_names: list of unique RBP names decribing the clip_files.
        transcripts_to_keep: a DataFrame of specific transcripts to analyze,
        which does not necessarily contain the start, end and strand information. If provided, the
        final matrix will contain only this transcripts. If None, all transcripts from df_transcripts will be
        analyzed.
        df_transcripts_ID: Name of the column in df_transcripts containing the IDs of the transcripts.
        transcripts_to_keep_ID: Name of the column if transcripts_to_keep containing
        the IDs of the transcripts.
        nt_after: Number of nucleotides after (downstream) the PAS to analyze.
        nt_before: Number of nucleotides before (upstream) the PAS to analyze.
        length_region: Length of the regions to analyze. Defaults to 200.
        path_to_save: Path to save the matrix. If None, will be saved in the current directory.
        start_col: Name of the column in df_transcripts bearing the start information.
        end_col: Name of the column in df_transcripts bearing the end information.
        strand_col: Name of the column in df_transcripts bearing the strabd information.
    Raises:
        ValueError: If the number of clip_files doesn't match the number of unique RBP names.
    """

    if transcripts_to_keep is not None:
        df_transcripts = df_transcripts[
            df_transcripts[df_transcripts_ID].isin(
                transcripts_to_keep[transcripts_to_keep_ID].values
            )
        ]

    df_transcripts["regions"] = df_transcripts.apply(
        lambda row: create_regions_around_PAS(
            row[start_col],
            row[end_col],
            row[strand_col],
            nt_after=nt_after,
            nt_before=nt_before,
            length_region=length_region,
        ),
        axis=1,
    )

    # clip_files, RBP_unique_names = get_all_clip_files(path_to_clip, corrupted=corrupted)

    if not (len(clip_files) == len(RBP_unique_names)):
        raise ValueError(
            "List of files and list of RBP names should be of the same length."
        )

    for clip_path, RBP_name in zip(clip_files, RBP_unique_names):
        clip = read_gtf(clip_path)
        df_transcripts = add_cross_link_column(
            df_transcripts, clip, RBP_name, regions_col="regions"
        )

    cross_link_matrix = df_transcripts.set_index(df_transcripts_ID)[
        [col for col in df_transcripts.columns if "Region" in col]
    ].astype(int)

    print("Done calculating the matrix!")
    if path_to_save is not None:
        with open(
            os.path.join(path_to_save, prefix + "_" + "matrix_cross_links.pkl"), "wb"
        ) as f:
            pickle.dump(cross_link_matrix, f)
    else:
        with open(
            os.path.join("./", prefix + "_" + "matrix_cross_links.pkl"), "wb"
        ) as f:
            pickle.dump(cross_link_matrix, f)


def get_bootstrap_distribution_one_region(
    table_of_pairs: pd.DataFrame,
    sig_pairs: pd.DataFrame,
    cross_link_isoforms: list,
    iso_type: str = "proximal",
    nb_boostrap=20000,
    sig_threshold=0.05,
):
    """Compute a boostrapped distribution acroos all pairs (from table of pairs)
    of the frequency of cross link events on the isoform iso_type in the
    population of isoforms using the complete list of cross-linked isoforms.
    Compute the frequency of cross link events in the sig_pairs, and return the p-value
    associated with the null hypothesis of this frequency belonging to the bootstraped
    distribution of frequencies.
    Args:
        table_of_pairs: Table of all analysed pairs. Should contain columns "proximal_id" and "distal_id".
        sig_pairs: Table of significant pairs (either towards a proximal or a distal shift).
        Should contain columns "proximal_id" and "distal_id".
        cross_link_isoforms: List of isoforms that are known to be cross-linked by a given RBP in
        a given region.
        iso_type: Type of isoform on which we study the cross-link events. Default to proximal.
        nb_bootstrap: Number of bootstraped values to compute to create the null distribution.
    Returns:
        proportions: The list of values from the bootstraped distribution.
        p-value: p-value associated with the null hypothesis of the frequency of cross-link
        events in the significant pairs belonging to the null bootstraped distribution of frequencies.
    """

    proportions = []
    for i in range(nb_boostrap):
        sub_table = table_of_pairs.sample(len(sig_pairs))
        nb_cross_links = len(
            sub_table[sub_table[f"{iso_type}_id"].isin(cross_link_isoforms)]
        )
        proportions.append(nb_cross_links / len(sig_pairs))
    sig_prop = len(
        sig_pairs[sig_pairs[f"{iso_type}_id"].isin(cross_link_isoforms)]
    ) / len(sig_pairs)

    # Z-test
    z_score = (sig_prop - np.array(proportions).mean()) / np.array(proportions).std()

    # Calculate the p-value
    p_value = 2 * (1 - norm.cdf(abs(z_score)))

    if sig_prop < np.array(proportions).mean():
        regulation = "negative"
    else:
        regulation = "positive"

    if p_value < sig_threshold:
        significance = "sig"
    else:
        significance = "unsig"

    return proportions, p_value, significance, regulation
