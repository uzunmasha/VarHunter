from Bio import AlignIO
import os
import multiprocessing
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from scipy.optimize import differential_evolution

def find_global_maximum(values:list) -> tuple:
    """
    Find global maximum using interpolid function
    and return tuple of extremum coordinates.

    Args:
        y_values (list): list ofmissmatches percentage in each window 
    """
    x_int = np.arange(1, len(values)+1)

    # Интерполяция функции f(x)
    interp_f = interp1d(x_int, values)
    # Интерполяционная функция
    def interpolated_f(x):
        return interp_f(x)
    # Определяем функцию потерь для минимизации
    def loss(x):
        return -interpolated_f(x)

    result = differential_evolution(loss, [(1, len(values))])
    return round(result.x.item()), round(-result.fun, 4)

def extremum_singularity(values:list):
    """
    Check for singularity of the integer value
    of the extremum on the coordinate y (missmatches percentage)

    Args:
        values (list): list ofmissmatches percentage in each window 
    """
    extremums = []
    for _ in range(25):
        extremums.append(find_global_maximum(values)[0])

    extr_lst = list(set(extremums))
    count = 0
    for j in range(1, len(extr_lst)):
        if extr_lst[j-1] + 1 == extr_lst[j] or extr_lst[j-1] + 2 == extr_lst[j]:
            count += 1
    return count == (len(extr_lst)-1)

def count_differences(values:list) -> list:
    """
    Counts differences between near function values.

    Args:
        values (list): list ofmissmatches percentage in each window 
    """
    diff = []
    for i in range(1, len(values)):
        diff.append(round(np.abs(values[i] - values[i-1]), 4))
    return diff

def max_dist_near_extr(diff:list, extr_x:int) -> bool:
    """
    Сheck that the maximum distance lies
    in the neighborhood of the global extremum.

    Args:
        diff (list): list of differences between near function values
        extr_x (float): extremum value by Ox
    """
    max_diff = np.max(diff)
    max_diff_index = diff.index(max_diff)+2
    closeness = [extr_x-3, extr_x+3]
    if closeness[0] < max_diff_index < closeness[1]:
        return True
    return False

def less_threshhold(values:list, threshhold:float, n:int) -> bool:
    """
    Checks that the given number of points 
    is less than some threshold. 

    Args:
        values (list): list ofmissmatches percentage in each window 
        threshhold (float): limit for value cutoff
        n (int): the number of points to be less than threshhold
    """
    count = 0
    for value in values:
        if value < threshhold:
            count += 1 
    return count >= n

def more_threshhold(values:list, threshhold:int, n:int) -> bool:
    """
    Checks that the given number of points 
    is greater than some threshold. 

    Args:
        values (list): list ofmissmatches percentage in each window 
        threshhold (float): limit for value cutoff
        n (int): the number of points to be greater than threshhold
    """
    count = 0
    for value in values:
        if value > threshhold:
            count += 1 
    return count >= n

def extract_alignment(dir_name:str, pair_name:str) -> list:
    """
    Searches for an alignment for a particular pair of genes
    and returns the aligned sequences as a list of two strings.

    Args:
        dir_name (str): name of directory with alignments
        pair_name (str): genes pair name 
    """
    sequences = []
    with open(os.path.join(f'{dir_name}', pair_name), mode='r') as file:
        seq = ''
        for line in file:
            line = line.strip()
            if line.startswith('>'):
                if seq:
                    sequences.append(seq)
                    seq = ''
            else:
                seq += line
        if seq:
            sequences.append(seq)
    return sequences

def not_long_gap(values:list, dir_name:str, pair_name:str) -> bool:
    """
    Сhecks areas with a high percentage of mismatches
    for gaps throughout the entire length of the window(s).

    Args
        values (list): list ofmissmatches percentage in each window 
        dir_name (str): name of directory with alignments
        pair_name (str): genes pair name 
    """
    seq_1, seq_2 = extract_alignment(dir_name, pair_name)
    window_size = len(seq_1) // 20 

    def count_end_gaps(seq:str, values:list, n:int) -> list:
        seq_gaps = []
        for i, value in enumerate(values):
            if value >= 95:
                seq_gaps.append(seq[n*i:n*(i+1)].count('-'))
        return seq_gaps
    
    seq_1_gaps = count_end_gaps(seq_1, values, window_size)
    seq_2_gaps = count_end_gaps(seq_2, values, window_size)

    if seq_1_gaps > seq_2_gaps:
        freq_gaps = [freq / window_size for freq in seq_1_gaps]
        return False in set(map(lambda x: x >= 0.95, freq_gaps))
    freq_gaps = [freq / window_size for freq in seq_2_gaps]
    return False in set(map(lambda x: x >= 0.95, freq_gaps))

def check_gene_pair(dir_name:str, pair_name:str, values:list) -> bool:
    """
    Testing each pair of genes against the conditions
    under which we can detect a particular pattern.

    Args:
        dir_name (str): name of directory with alignments
        pair_name (str): genes pair name 
        values (list): list ofmissmatches percentage in each window 
    """
    indexes = list(range(1, len(values) + 1))
    extremum_idx = find_global_maximum(values)[0]
    extremum = find_global_maximum(values)[1]
    differences = count_differences(values)
    
    if any(value > 95 for value in values):
        return extremum_singularity(values) and max_dist_near_extr(differences, extremum_idx) and \
        less_threshhold(values, 20, 14) and more_threshhold(values, 40, 2) and not_long_gap(values, dir_name, pair_name)
    else:
        return extremum_singularity(values) and max_dist_near_extr(differences, extremum_idx) and \
        less_threshhold(values, 20, 14) and more_threshhold(values, 40, 2)
    
def extract_all_pairs(gene_pair_name:str, df:pd.DataFrame) -> list:
    """
    Search for all possible gene pairs for two genomes.

    Args:
        gene_pair_name (str): name of file with alignment (name of genes pairs)
        df (pd.DataFrame): table with mismatch percentages for all pairwise alignments
    """
    first = gene_pair_name[len('aligned_'):len('aligned_')+15]
    second = gene_pair_name[len('aligned_STRP.0423.00004.0001i_01903-'):len('aligned_STRP.0423.00004.0001i_01903-')+15]
    gene_pairs = df.loc[df['File'].str.contains(first)].loc[df['File'].str.contains(second)].File.to_list()
    return gene_pairs

def find_similar_extr(extrs:dict) -> tuple:
    """
    Search for extrema with the same coordinates.

    Args:
        extrs (dict): 'gene_pair_name : (extr_x, extr_y)'
    """
    matching_pairs = []
    for name, extr in extrs.items():
        for other_name, other_extr in extrs.items():
            if name != other_name and extr[0] == other_extr[0] and abs(extr[1] - other_extr[1]) <= 1:
                matching_pairs.append((name, other_name))
    return matching_pairs

def filter_dataframe_by_names(df, gene_pairs_local):
    filtered_df = df[df['File'].isin(gene_pairs_local)]
    return filtered_df

def find_variation(df:pd.DataFrame, genes_pair_name:str, dir_name:str):
    """
    Search for variation within genomic pairs, 
    which should involve 4 genes (3 from each genome)

    Args:
        df (pd.DataFrame): table with mismatch percentages for all pairwise alignments
        genes_pair_name (str): gene pair for whose genomes we are looking for variation
        dir_name (str): name of directory with alignments

    """
    gene_pairs_local = filter_dataframe_by_names(df, extract_all_pairs(genes_pair_name, df))
    gene_pairs_local['check'] = gene_pairs_local.apply(lambda x: check_gene_pair(dir_name, x[0], x[1:]), axis= 1)
    gene_pairs_local = gene_pairs_local[gene_pairs_local['check'] == True]
    extremums = {}
    for gene_pair in gene_pairs_local.File:
        values = list(df.query("File == @gene_pair").drop('File', axis=1).iloc[0])
        extremums[gene_pair] = find_global_maximum(values)
    
    res = find_similar_extr(extremums)
    n = round( len(res) / 2)
    return res[:n]

def find_variation_in_df(input_df:pd.DataFrame, start_pos:int, stop_pos:int, dir_name:str) -> pd.DataFrame:
    input_df_test = input_df.iloc[start_pos:stop_pos, :]
    input_df_test['check'] = input_df_test.apply(lambda x: check_gene_pair(dir_name, x[0], x[1:]), axis=1)
    input_df_test_true = input_df_test[input_df_test['check'] == True]

    genes_pairs_names = input_df_test_true.File.to_list()
    
    output_df = pd.DataFrame(columns=['genome_A','genome_B', 'gene_1A', 'gene_1B', 'gene_2A', 'gene_2B'])
    if len(input_df_test_true) != 0:
        for genes_pair in genes_pairs_names:
            phase_variation = find_variation(input_df, genes_pair, dir_name)

            if len(phase_variation) >= 2:
                for pair in phase_variation:
                    row_to_append = pd.DataFrame([
                        {
                            'genome_A':pair[0][8:23], 
                            'gene_1A':pair[0][30:35],
                            'gene_1B':pair[0][58:63],
                            'genome_B':pair[0][36:51],
                            'gene_2A':pair[1][30:35],
                            'gene_2B':pair[1][58:63]
                        }
                    ])
                    output_df = pd.concat([output_df, row_to_append], ignore_index=True)
            new_output_df =  output_df[output_df.index % 4 == 0]

        return new_output_df.drop_duplicates(inplace=False)
    else:
        return output_df

# in_df = pd.read_csv('Wolbachia_1_mismatches.csv').fillna(0)
# start = 0
# stop = 100
# d = 'Wolbachia_1_pairs_aligned_short'

in_df = pd.read_csv('Streptococcus_pneumoniae_mismatches.csv').fillna(0)
start = 45000
stop = 46000
d = 'Streptococcus_pneumoniae_pairs_aligned'


res_df = find_variation_in_df(in_df, start, stop, d)
# res_df.to_csv('test_version_45000-46000.csv')
print(res_df)
# aligned_STRP.0423.00004.0001i_01903-STRP.0423.00022.0001i_00884.fasta.best.fas
# aligned_WOLB.0323.00001.0001i_00126-WOLB.0323.00002.0001i_00068.fasta.best.fas