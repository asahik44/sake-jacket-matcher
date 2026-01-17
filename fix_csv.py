import pandas as pd
import re

# 1. 元のファイルを読み込む
input_file = 'additional_situation_genre_map.csv'
output_file = 'clean_additional_data.csv'

print(f"読み込み中: {input_file}")
with open(input_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 2. データを抽出・整形
data = []
for line in lines:
    line = line.strip()
    if not line: continue
    
    # 余分な記号（CSVの引用符など）を除去
    if line.startswith('"') and line.endswith('"'):
        line = line[1:-1]
    line = line.replace('""', '"')
    
    # ("キーワード", "ジャンル") のパターンを探す
    match = re.search(r'\("([^"]+)",\s*"([^"]+)"\)', line)
    if match:
        text, label = match.groups()
        data.append({"text": text, "label": label})

# 3. きれいなCSVとして保存
if data:
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    print(f"完了！ '{output_file}' を作成しました。")
    print(f"データ件数: {len(df)}件")
    print("データ例:")
    print(df.head())
else:
    print("データが見つかりませんでした。ファイルの形式を確認してください。")