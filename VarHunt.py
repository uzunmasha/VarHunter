from Bio import AlignIO
import os
import multiprocessing
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from scipy.optimize import differential_evolution

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


def align_sequences(input_folder, output_folder):
    """
    Align gene sequences using the PRANK algorithm with codon-based alignment.

    Args:
        input_folder (str): Path to the folder containing input FASTA files.
        output_folder (str): Path to the folder for saving aligned sequences.
    """
    # Проверяем, существует ли уже папка с выравниваниями
    if os.path.exists(output_folder):
        print(f"Folder {output_folder} already exists. Skipping alignment.")
        # Если папка существует, переходим сразу к поиску мисмэтчей
        find_mismatches(output_folder)
        return

    os.makedirs(output_folder, exist_ok=True)
    
    # Определяем количество доступных процессоров
    num_processors = multiprocessing.cpu_count()
    
    # Создаем пул процессов
    pool = multiprocessing.Pool(processes=num_processors)
    
    # Получаем список файлов в папке
    files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.fasta')]
    
    # Запускаем выравнивание для каждого файла в пуле процессов
    for filename in files:
        input_path = filename
        output_path = os.path.join(output_folder, f"aligned_{os.path.basename(filename)}")

        command = f"prank -d={input_path} -o={output_path} -codon"
        os.system(command)
    
    # Завершаем работу пула процессов
    pool.close()
    pool.join()
    
    # После каждого выравнивания вызываем функцию для анализа мисмэтчей
    find_mismatches(output_folder)
            
def find_mismatches(output_folder):
    mismatches_data = []

    # Проходим по всем файлам в папке output_folder
    for filename in os.listdir(output_folder):
        if filename.endswith('.fas'):
            aligned_file = os.path.join(output_folder, filename)
            
            # Открываем файл с выравниванием
            alignment = AlignIO.read(aligned_file, 'fasta')
            
            # Получаем длину выравнивания
            alignment_length = alignment.get_alignment_length()

            # Разделяем длину на 20 сегментов
            segment_length = alignment_length // 20

            # Проходим по выравниванию с окном длиной segment_length
            for i in range(0, alignment_length, segment_length):
                # Инициализируем счетчик мисматчей для текущего сегмента
                mismatch_count = 0

                # Проходим по каждой позиции в сегменте и сравниваем символы в выравниваниях
                for j in range(segment_length):
                    # Проверяем, что индексы не выходят за пределы длины выравнивания
                    if i + j < alignment_length:
                        # Получаем символы из каждого выравнивания
                        symbol1 = alignment[0, i + j]
                        symbol2 = alignment[1, i + j]
                        
                        # Если символы не совпадают, увеличиваем счетчик мисматчей
                        if symbol1 != symbol2:
                            mismatch_count += 1
                
                # Вычисляем количество мисмэтчей в текущем сегменте, поделенное на длину сегмента
                mismatch_ratio = mismatch_count / segment_length * 100

                # Добавляем данные о мисматчах для текущего сегмента в список
                mismatches_data.append({'File': filename,
                                        'Segment': i // segment_length + 1,
                                        'Mismatch_Count': mismatch_ratio})
    
    # Создаем DataFrame из списка словарей
    df = pd.DataFrame(mismatches_data)
    
    # Переворачиваем таблицу так, чтобы сегменты были столбцами
    df = df.pivot(index='File', columns='Segment', values='Mismatch_Count')
    
    return df


def mismatch_imaging(file_name:str):
    df = pd.read_csv(file_name)
    df.name = file_name[:-len('_mismatches.csv')]

    condition = f'{df.name}'
    folder_name = f'{condition}_plots'
    os.makedirs(folder_name, exist_ok=True)

    # Найдем максимальное значение на всем диапазоне графиков для установки общего предела оси y
    max_value = df.iloc[:, 1:].max().max()

    for index, row in df.iterrows():
        # Получим название из первого столбца
        title = row.iloc[0]

        # Получим значения точек из остальных столбцов
        values = row.iloc[1:].values

        # Сгенерируем список с номерами точек
        indexes = list(range(1, len(values) + 1))

        # Создаем график
        plt.plot(indexes, values, marker='o')

        # Подписали оси и заголовки графика
        plt.xlabel('Number of windows size=20')
        plt.ylabel('Mismatch Ratio % (Mismatch Count / Segment Length)')
        plt.title(f'{title}')

        # Установим общий предел для оси y
        plt.ylim(0, max_value)
        plt.xticks(range(1, 20))

        filename = f'{folder_name}/{title[len('aligned_'):-len('.fasta')]}.png'
        plt.savefig(filename)

        # Очистим текущую фигуру
        plt.clf()

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


def detect_phase_variations(genomes_csv, lst_info_file, fasta_file, selection_condition):
    """
    Detect phase variations in gene sequences based on specified conditions.

    Args:
        genomes_csv (str): Path to the CSV file containing genome information.
        lst_info_file (str): Path to the LSTINFO file containing additional information.
        fasta_file (str): Path to the FASTA file containing gene sequences.
        selection_condition (str or list of str): Selection condition(s) for filtering gene pairs.
    """
    data_2 = process_info_files(genomes_csv, lst_info_file)
    genes_df = convert_fasta_to_dataframe(fasta_file)
    input_table = process_dataframes(genes_df, data_2)

    if isinstance(selection_condition, list):
        for condition in selection_condition:
            condition_rows = input_table[input_table['species'].str.contains(condition, na=False)]
            create_gene_pairs_folders(condition_rows, condition)
            input_folder = f"{condition.replace(' ', '_')}_pairs"
            output_folder = f"{condition.replace(' ', '_')}_pairs_aligned"
            align_sequences(input_folder, output_folder)
            mismatches_df = find_mismatches(output_folder)
            mismatches_df.to_csv(f'{condition}_mismatches.csv', index=True)
            mismatch_imaging(f'{condition}_mismatches.csv')
            output_df = find_variation_in_df(mismatches_df, 64000, 64500, output_folder)
            output_df.to_csv(f'{condition.replace(' ', '_')}_phva.csv')
    else:
        condition_rows = input_table[input_table['species'].str.contains(selection_condition, na=False)]
        create_gene_pairs_folders(condition_rows, selection_condition)
        input_folder = f"{selection_condition.replace(' ', '_')}_pairs"
        output_folder = f"{selection_condition.replace(' ', '_')}_pairs_aligned"
        align_sequences(input_folder, output_folder)
        mismatches_df = find_mismatches(output_folder)
        mismatches_df.to_csv(f'{selection_condition}_mismatches.csv', index=True)
        mismatch_imaging(f'{selection_condition}_mismatches.csv')
        output_df = find_variation_in_df(mismatches_df, 64000, 64500, output_folder)
        output_df.to_csv(f'{condition.replace(' ', '_')}_phva.csv')

# Входные данные:
genomes_csv_path = 'raw_data/genomesNames.csv'
lst_info_file_path = 'raw_data/LSTINFO-LSTINFO-NA-filtered-0.0001_0.6.lst'
fasta_file_path = 'raw_data/HxxHxHHits820SeqsGenes.fasta'
selection_condition = ['Streptococcus pneumoniae']
detect_phase_variations(genomes_csv_path, lst_info_file_path, fasta_file_path, selection_condition)

