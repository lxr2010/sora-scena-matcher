import pandas as pd

# 定义输入和输出文件名
input_file = 'match_result.csv'
old_file = 'old_match_result.csv'

try:
    # 读取CSV文件
    df_match = pd.read_csv(input_file)
    df_old = pd.read_csv(old_file)
except FileNotFoundError as e:
    print(f"错误：找不到文件 {e.filename}。请确保文件在正确的路径下。")
    exit()

# 确保必要的列存在
required_columns_match = ['RemakeVoiceID', 'OldScriptId', 'OldVoiceFilename']
required_columns_old = ['RemakeVoiceID', 'OldScriptId', 'OldVoiceFilename']
if not all(col in df_match.columns for col in required_columns_match):
    print(f"错误：'{input_file}' 必须包含以下列：{required_columns_match}")
    exit()
if not all(col in df_old.columns for col in required_columns_old):
    print(f"错误：'{old_file}' 必须包含以下列：{required_columns_old}")
    exit()

# 定义所有应为整数的列
numeric_cols = ['RemakeVoiceID', 'RemakeScenaScriptLineno', 'RemakeScenaScriptAddStructLineno', 'RemakeScenaScriptTranslationLineno', 'RemakeScenaScriptTranslationAddStructLineno', 'OldScriptId']

# --- 1. 从 old_match_result.csv 创建映射 ---
# 将ID列转换为可空整数类型
df_old['RemakeVoiceID'] = pd.to_numeric(df_old['RemakeVoiceID'], errors='coerce').astype('Int64')
df_old['OldScriptId'] = pd.to_numeric(df_old['OldScriptId'], errors='coerce').astype('Int64')

# 移除 RemakeVoiceID 或 OldScriptId 为空的行
df_old.dropna(subset=['RemakeVoiceID', 'OldScriptId', 'OldVoiceFilename'], inplace=True)

# 按 RemakeVoiceID 分组，并取第一个非空的 OldScriptId 和 OldVoiceFilename
old_id_map = df_old.groupby('RemakeVoiceID')['OldScriptId'].first()
old_voice_map = df_old.groupby('RemakeVoiceID')['OldVoiceFilename'].first()

# --- 2. 处理 match_result.csv ---
# 将所有数字列转换为可空整数类型，以正确处理空值
for col in numeric_cols:
    if col in df_match.columns:
        df_match[col] = pd.to_numeric(df_match[col], errors='coerce').astype('Int64')

# --- 3. 应用第一个功能：更新 RemakeVoiceID < 50000 的项 ---
# 创建辅助列来存放来自 old.csv 的数据
df_match['New_OldScriptId'] = df_match['RemakeVoiceID'].map(old_id_map)
df_match['New_OldVoiceFilename'] = df_match['RemakeVoiceID'].map(old_voice_map)

# 条件：RemakeVoiceID < 50000 且从 old.csv 找到了一个新的 OldVoiceFilename
condition_update = (df_match['RemakeVoiceID'] < 50000) & (df_match['New_OldVoiceFilename'].notna())

# 使用 .loc 根据条件更新 OldScriptId 和 OldVoiceFilename
df_match.loc[condition_update, 'OldScriptId'] = df_match.loc[condition_update, 'New_OldScriptId']
df_match.loc[condition_update, 'OldVoiceFilename'] = df_match.loc[condition_update, 'New_OldVoiceFilename']

# --- 4. 应用第二个功能：清理 RemakeVoiceID > 50000 的项 ---
# 创建一个 old_match_result.csv 中所有 OldVoiceFilename 的集合
old_voice_filenames_set = set(df_old['OldVoiceFilename'].dropna().unique())

# 条件：RemakeVoiceID > 50000 且 OldVoiceFilename 存在于上面的集合中
condition_clear = (df_match['RemakeVoiceID'] > 50000) & (df_match['OldVoiceFilename'].isin(old_voice_filenames_set))

# 需要清空的列
columns_to_clear = ['OldVoiceFilename', 'OldScriptId', 'OldVoiceText']
for col in columns_to_clear:
    if col in df_match.columns:
        # 对于数字列，使用 pd.NA；对于文本列，使用 ''
        value_to_set = pd.NA if col in numeric_cols else ''
        df_match.loc[condition_clear, col] = value_to_set

# 同时，将这些行的 MatchType 设置为 'conflict'
if 'MatchType' in df_match.columns:
    df_match.loc[condition_clear, 'MatchType'] = 'conflict'

# --- 5. 清理和准备输出 ---
# 删除辅助列
final_df = df_match.drop(columns=['New_OldScriptId', 'New_OldVoiceFilename'])

# 定义一个函数，将数字转为整数的字符串，将 pd.NA 转为空字符串
def to_str_or_empty(val):
    if pd.isna(val):
        return ''
    return str(int(val))

# 在导出前，对所有数字列应用此函数，以防止它们被写成浮点数
for col in numeric_cols:
    if col in final_df.columns:
        final_df[col] = final_df[col].apply(to_str_or_empty)

# --- 6. 按要求输出文件 ---
import os

# 筛选出 RemakeVoiceID > 50000 的项
df_to_output = final_df[pd.to_numeric(final_df['RemakeVoiceID']) > 50000]

# 创建输出目录
output_dir = 'output_scripts'
os.makedirs(output_dir, exist_ok=True)

# 按 'RemakeScenaScriptFilename' 分组
grouped = df_to_output.groupby('RemakeScenaScriptFilename')

# 为每个分组保存一个CSV文件
for filename, group_df in grouped:
    # 构建输出文件路径
    output_path = os.path.join(output_dir, f"match_result_{filename}.csv")
    # 保存文件
    group_df.to_csv(output_path, index=False)

print(f"处理完成！结果已按脚本文件名保存到 '{output_dir}' 目录中。")
