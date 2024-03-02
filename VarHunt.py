from Bio import AlignIO
import os
import pandas as pd
import matplotlib.pyplot as plt

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
        if filename.endswith('.fasta'):
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

                # Добавляем данные о мисматчах для текущего сегмента в список
                mismatches_data.append({'File': filename,
                                        'Segment': i // segment_length + 1,
                                        'Mismatch_Count': mismatch_count})
    
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
    os.mkdir(folder_name)

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
        plt.ylabel('Number of mismatches')
        plt.title(f'{title}')

        filename = f'{folder_name}/{title[len('aligned_'):-len('.fasta')]}.png'
        plt.savefig(filename)

        # Очистим текущую фигуру
        plt.clf()


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
    else:
        condition_rows = input_table[input_table['species'].str.contains(selection_condition, na=False)]
        create_gene_pairs_folders(condition_rows, selection_condition)
        input_folder = f"{selection_condition.replace(' ', '_')}_pairs"
        output_folder = f"{selection_condition.replace(' ', '_')}_pairs_aligned"
        align_sequences(input_folder, output_folder)
        mismatches_df = find_mismatches(output_folder)
        mismatches_df.to_csv(f'{selection_condition}_mismatches.csv', index=True)
        mismatch_imaging(f'{selection_condition}_mismatches.csv')

# Входные данные:
genomes_csv_path = 'raw_data/genomesNames.csv'
lst_info_file_path = 'raw_data/LSTINFO-LSTINFO-NA-filtered-0.0001_0.6.lst'
fasta_file_path = 'raw_data/HxxHxHHits820SeqsGenes.fasta'
selection_condition = ['Streptococcus porcinus', 'Streptococcus gordonii']
detect_phase_variations(genomes_csv_path, lst_info_file_path, fasta_file_path, selection_condition)
