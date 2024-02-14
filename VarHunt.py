from Bio import AlignIO
import os
import pandas as pd

def process_info_files(genomes_csv, lst_info_file):
    """
    Process information files to extract relevant data for further analysis.

    Args:
        genomes_csv (str): Path to the CSV file containing genome information.
        lst_info_file (str): Path to the LSTINFO file containing additional information.

    Returns:
        pandas.DataFrame: Processed DataFrame containing necessary information.
    """
    genome_names = pd.read_csv(genomes_csv, sep='\t', header=None).rename(columns={0:'GCF', 1:'species', 2:'strain'})
    genome_names['strain'] = genome_names['strain'].str[len('strain='):]
    genome_names['strain'] = genome_names['strain'].fillna(0)

    lst_info = pd.read_csv(lst_info_file, sep='\t')
    lst_info['GCF'] = lst_info['orig_name'].str.extract(r'(GCF\_\d{9}\.\d{1})', expand=True)

    data_1 = lst_info[['gembase_name', 'GCF']]
    data_2 = pd.merge(data_1, genome_names, on='GCF')

    return data_2

def convert_fasta_to_dataframe(file_name):
    """
    Convert a FASTA file to a pandas DataFrame.

    Args:
        file_name (str): Path to the input FASTA file.

    Returns:
        pandas.DataFrame: DataFrame containing gene sequences and related information.
    """
    dic = {}
    cur_scaf = ''
    cur_seq = []

    for line in open(file_name):
        if line.startswith(">") and cur_scaf == '':
            cur_scaf = line.split(' ')[0][1:]
        elif line.startswith(">") and cur_scaf != '':
            dic[cur_scaf] = ''.join(cur_seq)
            cur_scaf = line.split(' ')[0][1:]
            cur_seq = []
        else:
            cur_seq.append(line.rstrip())

    dic[cur_scaf] = ''.join(cur_seq)

    names = dic.keys()
    seqs = dic.values()

    genes_dict = {'name': names, 'gene': seqs}
    genes_df = pd.DataFrame.from_dict(genes_dict)
    genes_df['gembase_name'] = genes_df['name'].str.extract(r'(STRP\.0423\.\d{5})', expand=True)

    return genes_df