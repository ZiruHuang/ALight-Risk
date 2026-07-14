from abnumber import Chain
import os
import pandas as pd
import json
## 正样本
### 统一序列名称
seq_list = pd.read_table("1.Data/Pos/1.Pos_seq.tsv",index_col= None)
seq_list['NewID']=[f'AL_{i+1}' for i in range(len(seq_list))]
columns = seq_list.columns.tolist()
# 将新列名移动到第一个位置
columns.insert(0, columns.pop(columns.index('NewID')))
# 重新排列 DataFrame 的列顺序
seq_list = seq_list[columns]
seq_list.to_csv("1.Data/Pos/1.Pos_seq_NewID.tsv", sep="\t", index=False, header=True)
### 提取Fv序列
def get_processed_seq(seq):
    try:
        chain = Chain(seq, scheme='imgt')  # 创建Chain对象
        processed_seq = chain.seq  # 获取处理后的序列
        # 如果未能识别出FV序列或为空，返回None
        if not processed_seq or len(processed_seq) == 0:
            return None
        return processed_seq
    except Exception as e:
        # 出现异常时返回None
        print(f"无法识别FV序列: {seq} - {e}")
        return None
seq_list['Processed Seq'] = seq_list['Seq'].apply(get_processed_seq)
seq_list.to_csv('1.Data/Pos/2.Pos_seq_fv.csv', index=False)
# 将处理后的序列保存为FASTA文件
with open("1.Data/Pos/2.Pos_seq_fv.fasta", "w") as fasta_file:
    for index, row in seq_list.iterrows():
        seq_id = row['NewID']  # 第一列作为序列名称
        processed_seq = row['Processed Seq']  # 第三列作为序列
        fasta_file.write(f">{seq_id}\n{processed_seq}\n")
### 1.1.3  删除含有特殊字符的序列
from Bio import SeqIO

valid_aas = set("ACDEFGHIKLMNPQRSTVWY")

def is_valid_sequence(seq):
    return set(seq).issubset(valid_aas)

clean_records = []
for record in SeqIO.parse("1.Data/Pos/2.Pos_seq_fv.fasta", "fasta"):
    if is_valid_sequence(str(record.seq)):
        clean_records.append(record)

SeqIO.write(clean_records, "1.Data/Pos/3.cleaned-Pos_seq_fv.fasta", "fasta")

## 负样本
import pandas as pd
import os
import glob

# 指定路径
path = "./1.Data/Neg/OAS_data/csv/"

# 批量读取CSV文件
files = glob.glob(os.path.join(path, "*.csv"))

LK_type = []

for file_path in files:
    # 读取CSV文件中的指定列
    try:
        tmp = pd.read_csv(file_path)['v_call_light'].dropna()
        # 将每个文件的序列加入列表
        LK_type.extend(tmp.tolist())
    except Exception as e:
        print(f"读取文件 {file_path} 时出错: {e}")
        
light_chain = []

for file_path in files:
    # 读取CSV文件中的指定列
    try:
        tmp = pd.read_csv(file_path)['sequence_alignment_aa_light'].dropna()
        # 将每个文件的序列加入列表
        light_chain.extend(tmp.tolist())
    except Exception as e:
        print(f"读取文件 {file_path} 时出错: {e}")

# 创建ID列
ids = [f"OAS{i+1}" for i in range(len(light_chain))]

# 构建最终的DataFrame
Neg_seq_table = pd.DataFrame({'ID': ids,"Seq":light_chain, 'LK': LK_type})

# 保存为TSV文件
Neg_seq_table.to_csv("1.Data/Neg/1.Neg_OAS_table.txt", sep="\t", index=False, header=True)

Neg_seq_table['Processed Seq'] = Neg_seq_table['Seq'].apply(get_processed_seq)
Neg_seq_table.to_csv('1.Data/Neg/2.OAS_seq_fv.csv', index=False)
# 将处理后的序列保存为FASTA文件
with open("1.Data/Neg/2.OAS_seq_fv.fasta", "w") as fasta_file:
    for index, row in Neg_seq_table.iterrows():
        seq_id = row['ID']  # 第一列作为序列名称
        processed_seq = row['Processed Seq']  # 第三列作为序列
        fasta_file.write(f">{seq_id}\n{processed_seq}\n")

### 1.2.2  AL-Base nonAL序列
seq_list = pd.read_table("1.Data/Neg/1.Neg_ALBase-NonAL_table.txt",index_col= None)
seq_list['NewID']=[f'nonAL_{i+1}' for i in range(len(seq_list))]
columns = seq_list.columns.tolist()
# 将新列名移动到第一个位置
columns.insert(0, columns.pop(columns.index('NewID')))
# 重新排列 DataFrame 的列顺序
seq_list = seq_list[columns]
seq_list['Processed Seq'] = seq_list['Seq'].apply(get_processed_seq)
seq_list.to_csv('1.Data/Neg/2.ALBase-NonAL_seq_fv.csv', index=False)
seq_list = seq_list.dropna(subset=['Processed Seq'])
# 将处理后的序列保存为FASTA文件
with open("1.Data/Neg/2.ALBase-NonAL_seq_fv.fasta", "w") as fasta_file:
    for index, row in seq_list.iterrows():
        seq_id = row['NewID']  # 第一列作为序列名称
        processed_seq = row['Processed Seq']  # 第三列作为序列
        fasta_file.write(f">{seq_id}\n{processed_seq}\n")

