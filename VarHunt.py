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

def process_dataframes(genes_df, data_2):
    """
    Process DataFrames containing gene sequences and additional information.

    Args:
        genes_df (pandas.DataFrame): DataFrame containing gene sequences.
        data_2 (pandas.DataFrame): DataFrame containing additional information.

    Returns:
        pandas.DataFrame: Processed DataFrame with merged data.
    """
    input_table = pd.merge(genes_df, data_2, on='gembase_name')
    input_table = input_table[['name', 'GCF', 'species', 'strain', 'gene']]
    input_table = input_table[~input_table['species'].str.contains(' sp.')]

    return input_table

def create_gene_pairs_folders(condition_rows, selection_condition):
    """
    Create folders and save gene pairs based on specified conditions.

    Args:
        condition_rows (pandas.DataFrame): DataFrame containing gene pairs.
        selection_condition (str): Selection condition for filtering gene pairs.
    """
    # Создаем папку для сохранения пар
    output_folder = f"{selection_condition.replace(' ', '_')}_pairs"
    os.makedirs(output_folder, exist_ok=True)

    # Получаем количество строк в DataFrame
    num_rows = len(condition_rows)

    # Проходим по каждой строке в DataFrame
    for i in range(num_rows):
        for j in range(i + 1, num_rows):
            # Получаем значения из пары
            name1, name2 = condition_rows.iloc[i]['name'], condition_rows.iloc[j]['name']
            gcf1, gcf2 = condition_rows.iloc[i]['GCF'], condition_rows.iloc[j]['GCF']

            # Проверяем, что значения из пары не совпадают и GCF у пары разные
            if name1 != name2 and gcf1 != gcf2:
                # Форматируем текст и создаем имя файла
                row1 = condition_rows.iloc[i]
                row2 = condition_rows.iloc[j]
                
                # Заменяем пробелы на нижнее подчеркивание
                species1 = row1['species'].replace(' ', '_')
                species2 = row2['species'].replace(' ', '_')

                formatted_text = f">{row1['name']}_{row1['GCF']}_{species1}_{row1['strain']}\n{row1['gene']}\n"
                formatted_text += f">{row2['name']}_{row2['GCF']}_{species2}_{row2['strain']}\n{row2['gene']}\n"
                
                # Создаем имя файла
                filename = f"{output_folder}/{row1['name']}-{row2['name']}.fasta"

                # Сохраняем пару в файл
                with open(filename, 'w') as file:
                    file.write(formatted_text)