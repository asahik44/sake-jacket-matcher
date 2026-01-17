import pickle
import pandas as pd

# ==========================================
# 設定: AIに教えたい「言葉とジャンルの結びつき」
# ==========================================
# ここで「洋酒」「米焼酎」などの課題を解決するデータを仕込みます
MANUAL_KNOWLEDGE = [
    # --- 洋酒の定義 ---
    ("洋酒好き", "ウイスキー"),
    ("洋酒 おすすめ", "ブランデー"),
    ("洋酒のプレゼント", "ウイスキー"),
    ("洋酒", "ジン・クラフトジン"),
    
    # --- 焼酎の区別 ---
    ("米焼酎 おすすめ", "米焼酎"),
    ("いも焼酎が好き", "芋焼酎"),
    ("麦焼酎のボトル", "麦焼酎"),
    ("クセのない焼酎", "甲類焼酎"), 
    
    # --- 味や雰囲気からの推測 ---
    ("さっぱりしたお酒", "サワーの素・割材"),
    ("乾杯用のお酒", "ビール"),
    ("お祝いの泡", "スパークリングワイン"),
    ("甘いお酒", "梅酒"),
    ("デザートワイン", "リキュール"),
    ("沖縄のお酒", "泡盛"),
]

def augment_text(text):
    # 同じ意味で言い回しを変えてデータを水増しする
    return [text, f"{text}を教えて", f"{text}が飲みたい", f"{text}のおすすめ"]

def main():
    print("データベースを読み込んでいます...")
    try:
        with open('sake_database.pkl', 'rb') as f:
            db_data = pickle.load(f)
    except FileNotFoundError:
        print("エラー: sake_database.pkl が見つかりません")
        return

    data_rows = []

    # 1. 実データ（商品名）からの抽出
    print(f"商品データの抽出中... (総数: {len(db_data)}件)")
    for item in db_data:
        name = item['name']
        genre = item.get('genre', '')
        
        # ジャンルがない、または「その他」「ノンアルコール」は除外
        if not genre or genre in ["その他", "ノンアルコール"]:
            continue
            
        data_rows.append({"text": name, "label": genre})

    # 2. 知識データの追加
    print("知識データを追加中...")
    for phrase, target_genre in MANUAL_KNOWLEDGE:
        # そのジャンルがDB内に存在するか確認（存在しないジャンルは学習できないため）
        exists = any(row['label'] == target_genre for row in data_rows)
        if exists:
            augmented_phrases = augment_text(phrase)
            for p in augmented_phrases:
                # 重要なキーワードなので、5回繰り返してAIに強く記憶させる
                for _ in range(5): 
                    data_rows.append({"text": p, "label": target_genre})

    # 3. CSVに保存
    df = pd.DataFrame(data_rows)
    
    # 重複削除
    df = df.drop_duplicates(subset=["text"])
    
    csv_filename = "genre_dataset.csv"
    df.to_csv(csv_filename, index=False)
    
    print("-" * 30)
    print(f"完了！ '{csv_filename}' を作成しました。")
    print(f"データ件数: {len(df)}")
    print(f"ユニークなジャンル数: {len(df['label'].unique())}")
    print("\n含まれるジャンル一覧:")
    print(df['label'].unique())

if __name__ == "__main__":
    main()