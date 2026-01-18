import streamlit as st
import streamlit.components.v1 as components
import pickle
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import BertTokenizer, BertForSequenceClassification
import torch.nn.functional as F
import os 

# ==========================================
# ★Google Analytics設定
# ==========================================
def inject_ga():
    if "GA_ID" in st.secrets:
        GA_ID = st.secrets["GA_ID"]
        ga_code = f"""
        <script async src="https://www.googletagmanager.com/gtag/js?id={GA_ID}"></script>
        <script>
            window.dataLayer = window.dataLayer || [];
            function gtag(){{dataLayer.push(arguments);}}
            gtag('js', new Date());
            gtag('config', '{GA_ID}');
        </script>
        """
        components.html(ga_code, height=0)

# ==========================================
# ★設定エリア
# ==========================================
DEBUG_MODE = False  # Trueにすると裏側のスコアなどが見えます
APP_TITLE = "Sake Jacket Matcher"

GENRE_ORDER = [
    "ビール", "海外ビール", "地ビール・クラフトビール",
    "ウイスキー", "ワイン", "赤ワイン", "白ワイン", "スパークリングワイン", "シャンパン",
    "日本酒", "焼酎", "芋焼酎", "麦焼酎", "米焼酎",
    "サワーの素・割材", "リキュール", "ジン・クラフトジン", "梅酒",
    "ノンアルコール"
]

# ==========================================
# アプリ設定
# ==========================================
st.set_page_config(page_title=APP_TITLE, layout="wide")
inject_ga()

# ★フィルタボタンを残しつつ白枠を消すCSS
st.markdown("""
<style>
    /* 1. ヘッダー（上のバー）は「表示」させる！ */
    /* これがないとフィルターボタン（＞）が消えてしまいます */
    header {
        visibility: visible !important;
        background-color: transparent !important;
    }
    
    /* 2. 通常のフッターを消す */
    footer {
        visibility: hidden !important;
        display: none !important;
    }
    
    /* 3. 虹色の線だけ消す */
    div[data-testid="stDecoration"] {
        visibility: hidden;
        display: none;
    }

    /* 4. Streamlit Cloud特有の「白枠（ViewerBadge）」を強制的に消す */
    div[class*="viewerBadge"] {
        visibility: hidden !important;
        display: none !important;
    }
    .viewerBadge_container__1QSob {
        display: none !important;
    }

    /* 画像サイズの調整 */
    div[data-testid="stImage"] img { height: 200px; object-fit: contain; width: 100%; }
</style>
""", unsafe_allow_html=True)

# --- 定義 ---
BROAD_CATEGORIES = {
    "洋酒": ["ウイスキー", "ブランデー", "ジン・クラフトジン", "ウォッカ", "ラム", "テキーラ", "リキュール", "赤ワイン", "白ワイン"],
    "焼酎": ["芋焼酎", "麦焼酎", "米焼酎", "黒糖焼酎", "泡盛"],
    "ウィスキー": ["ウイスキー"], "ウヰスキー": ["ウイスキー"], "WHISKY": ["ウイスキー"],
    "ワイン": ["赤ワイン", "白ワイン", "ロゼワイン", "スパークリングワイン", "シャンパン"], "泡": ["スパークリングワイン", "シャンパン"],
    "ビール": ["ビール", "海外ビール", "地ビール・クラフトビール"],
    "サワー": ["サワーの素・割材", "リキュール"],
    "日本酒": ["日本酒"],
}

# --- モデル読み込み (頑丈版) ---
@st.cache_resource
def load_all_models():
    # 1. データベース読み込み (必須)
    try:
        with open('sake_database.pkl', 'rb') as f:
            db_data = pickle.load(f)
    except FileNotFoundError:
        st.error("データベース(sake_database.pkl)が見つかりません。")
        return None

    # 2. CLIPモデル読み込み (必須・自動DL)
    try:
        clip_model = SentenceTransformer('sentence-transformers/clip-ViT-B-32-multilingual-v1')
        all_vectors = np.concatenate([item['vector'] for item in db_data], axis=0)
    except Exception as e:
        st.error(f"CLIPモデル読み込みエラー: {e}")
        return None
    
    # ジャンルソート
    raw_genres = list(set([item.get('genre', 'その他') for item in db_data]))
    sorted_genres = sorted(raw_genres, key=lambda x: GENRE_ORDER.index(x) if x in GENRE_ORDER else 999)

    # 3. カスタムモデル読み込み (任意: なければスキップ)
    intent_tk, intent_md, genre_tk, genre_md = None, None, None, None
    has_logic_model = False

    try:
        intent_path = "./my_intent_model"
        genre_path = "./my_genre_model"
        
        if os.path.exists(intent_path) and os.path.exists(genre_path):
            intent_tk = BertTokenizer.from_pretrained(intent_path)
            intent_md = BertForSequenceClassification.from_pretrained(intent_path)
            genre_tk = BertTokenizer.from_pretrained(genre_path)
            genre_md = BertForSequenceClassification.from_pretrained(genre_path)
            has_logic_model = True
    except Exception:
        pass # モデルがない場合は無視して進む

    return {
        "db": db_data,
        "clip": clip_model,
        "vectors": all_vectors,
        "genres": sorted_genres,
        "intent_tk": intent_tk,
        "intent_md": intent_md,
        "genre_tk": genre_tk,
        "genre_md": genre_md,
        "has_logic_model": has_logic_model
    }

models = load_all_models()
if not models:
    st.stop()

# --- アルゴリズム関数 ---

# Intent / Genre 推論
def predict_intent(text):
    if not models["has_logic_model"]: return False, 0.0
    inputs = models["intent_tk"](text, return_tensors="pt", padding=True, truncation=True, max_length=64)
    with torch.no_grad(): outputs = models["intent_md"](**inputs)
    probs = F.softmax(outputs.logits, dim=-1)
    return probs[0][1].item() > 0.5, probs[0][1].item()

def predict_genre_probs(text):
    if not models["has_logic_model"]: return {}
    inputs = models["genre_tk"](text, return_tensors="pt", padding=True, truncation=True, max_length=64)
    with torch.no_grad(): outputs = models["genre_md"](**inputs)
    probs = F.softmax(outputs.logits, dim=-1)[0]
    return {models["genre_md"].config.id2label[i]: prob.item() for i, prob in enumerate(probs)}

# MMR (Maximal Marginal Relevance) 並び替えロジック
def mmr_sort(query_vec, candidate_vectors, candidate_items, top_k=12, diversity=0.4):
    """
    diversity: 0に近いほど類似度重視（今まで通り）、1に近いほど多様性重視（バラバラになる）
    推奨値: 0.3 ~ 0.5
    """
    # クエリとの類似度計算
    # candidate_vectorsはnumpy配列なのでtorch tensorに変換して計算
    cand_tensor = torch.tensor(candidate_vectors)
    query_tensor = torch.tensor(query_vec).unsqueeze(0)
    
    sims_to_query = util.cos_sim(query_tensor, cand_tensor)[0]
    
    # MMRによる選抜ループ
    selected_indices = []
    candidate_indices = list(range(len(candidate_items)))
    
    # 候補が少なすぎる場合は単純ソートで返す
    if len(candidate_items) <= top_k:
        sorted_indices = torch.argsort(sims_to_query, descending=True).tolist()
        return [candidate_items[i] for i in sorted_indices], [sims_to_query[i].item() for i in sorted_indices]

    for _ in range(min(len(candidate_items), top_k)):
        best_mmr_score = -float('inf')
        best_idx = -1
        
        for idx in candidate_indices:
            # クエリとの類似度
            similarity_to_query = sims_to_query[idx].item()
            
            # すでに選んだものとの類似度（最大値）
            if selected_indices:
                # 選ばれたアイテムのベクトルを取得
                selected_vecs = cand_tensor[selected_indices]
                current_vec = cand_tensor[idx].unsqueeze(0)
                sim_to_selected = util.cos_sim(current_vec, selected_vecs)
                max_similarity_to_selected = torch.max(sim_to_selected).item()
            else:
                max_similarity_to_selected = 0
            
            # MMRスコア = (1-λ)*クエリ類似度 - λ*選定済みとの類似度
            mmr_score = (1 - diversity) * similarity_to_query - diversity * max_similarity_to_selected
            
            if mmr_score > best_mmr_score:
                best_mmr_score = mmr_score
                best_idx = idx
        
        selected_indices.append(best_idx)
        candidate_indices.remove(best_idx)
    
    # 結果の構築
    results = [candidate_items[i] for i in selected_indices]
    result_scores = [sims_to_query[i].item() for i in selected_indices]
    
    return results, result_scores

# --- 検索エンジン本体 ---
def search_engine(original_query, selected_genres, min_p, max_p, mode="visual", logic_mode="A"):
    ai_message = ""
    search_genres = []
    
    # ★ プロンプトエンジニアリング (検証用 C/Dモード)
    # クエリを加工して「お酒の見た目を探している」ことを強調する
    if mode == "visual" and ("C" in logic_mode or "D" in logic_mode):
        # 日本語と英語を混ぜてCLIPに伝わりやすくするテクニック
        query_for_clip = f"「{original_query}」という雰囲気の、お酒のボトルデザイン。 Package design of sake bottle with the vibe of {original_query}."
        if DEBUG_MODE: st.toast(f"Modified Query: {query_for_clip}")
    else:
        query_for_clip = original_query

    # ジャンル絞り込みロジック
    if selected_genres:
        search_genres = selected_genres
    elif mode == "logic" and models["has_logic_model"]:
        # (既存のLogicモード処理...)
        target_genres = []
        for broad_key, children in BROAD_CATEGORIES.items():
            if broad_key in original_query: target_genres.extend(children)
        for g in models["genres"]:
            if g in original_query and g not in target_genres: target_genres.append(g)
        
        if target_genres:
            search_genres = list(set(target_genres))
            ai_message = "キーワードからジャンルを絞り込みました"
        else:
            is_nonal, nonal_conf = predict_intent(original_query)
            if is_nonal:
                search_genres = ["ノンアルコール"]
                ai_message = "ノンアルコール商品から探します"
            else:
                genre_probs = predict_genre_probs(original_query)
                sorted_genres = sorted(genre_probs.items(), key=lambda x: x[1], reverse=True)
                candidates = [sorted_genres[0][0]]
                for g, p in sorted_genres[1:]:
                    if p > 0.15: candidates.append(g)
                search_genres = candidates
                ai_message = f"AI推論: {search_genres[0]} などが合いそうです"

    elif mode == "visual" or not models["has_logic_model"]:
        search_genres = [] 
        ai_message = ""

    # ベクトル化
    query_vec = models["clip"].encode(query_for_clip, convert_to_tensor=True).cpu().numpy()
    
    # フィルタリング
    valid_indices = []
    for i, item in enumerate(models["db"]):
        if search_genres and item.get('genre') not in search_genres: continue
        if not (min_p <= item['price'] <= max_p): continue
        valid_indices.append(i)
        
    if not valid_indices: return [], ai_message
    
    # 候補データのベクトル抽出
    target_vectors = models["vectors"][valid_indices]
    candidate_items = [models["db"][i] for i in valid_indices]

    # ★ ランキングロジック分岐 (検証用 B/Dモード)
    if mode == "visual" and ("B" in logic_mode or "D" in logic_mode):
        # MMR (多様性重視)
        # diversity=0.4 くらいがバランス良し
        results, raw_scores = mmr_sort(query_vec, target_vectors, candidate_items, top_k=12, diversity=0.4)
    else:
        # 既存 (単純な類似度順)
        scores = util.cos_sim(query_vec, target_vectors)[0]
        sorted_args = torch.argsort(scores, descending=True)
        
        results = []
        raw_scores = []
        for i in range(min(12, len(sorted_args))):
            idx = sorted_args[i].item()
            results.append(candidate_items[idx])
            raw_scores.append(scores[idx].item())

    # 結果データの整形 (スコア付与など)
    final_results = []
    for item, raw_score in zip(results, raw_scores):
        # visualモードまたはモデルなしなら係数高め
        display_score = min(raw_score * 5.0, 0.99) if (mode == "visual" or not models["has_logic_model"]) else min(raw_score * 3.5, 0.99)
        item['match_score'] = display_score
        final_results.append(item)
        
    return final_results, ai_message

# --- UI構築 ---
st.title(f"🍾 {APP_TITLE}")

# サイドバー
st.sidebar.header("Search Mode")

if models["has_logic_model"]:
    mode_options = ("ジャケ買い (感性)", "AIソムリエ (知識)")
else:
    mode_options = ("ジャケ買い (感性)",) 

mode_select = st.sidebar.radio("検索モード", mode_options, index=0)
mode_key = "visual" if "ジャケ買い" in mode_select else "logic"

st.sidebar.divider()
st.sidebar.header("Filters")
user_genres = st.sidebar.multiselect("ジャンル固定", options=models["genres"])
price_range = st.sidebar.slider("価格帯", 0, 30000, (0, 30000), 500, format="¥%d")

# ★★★ 検証用メニュー (ここを追加！) ★★★
st.sidebar.divider()
st.sidebar.markdown("### 🧪 開発者メニュー")
logic_mode = st.sidebar.selectbox(
    "検索アルゴリズム検証",
    [
        "A: 通常 (Baseline)",
        "B: MMR (多様性重視)",
        "C: Prompt (言葉を補正)",
        "D: MMR + Prompt (最強?)"
    ],
    index=0
)
# ★★★★★★★★★★★★★★★★★★★★★★★★

if DEBUG_MODE: st.sidebar.warning("🔧 デバッグモード ON")

# メインエリア
col1, col2 = st.columns([3, 1], vertical_alignment="bottom")
with col1:
    placeholder = "例：サイバーパンクな夜、森の中で読書、初恋の味..." if mode_key == "visual" else "例：魚料理に合うワイン、BBQ..."
    query = st.text_input("どんな雰囲気のお酒がいい？", placeholder=placeholder).strip()
with col2:
    search_btn = st.button("Digる", type="primary", use_container_width=True)

if query or search_btn:
    if search_btn:
        # GA送信 (A/Bテスト用にLogicモードも一緒に送ると分析しやすいかも)
        components.html(f"<script>gtag('event', 'search', {{'search_term': '{query}', 'logic_mode': '{logic_mode}'}});</script>", height=0)

    st.divider()
    
    # 検索実行 (logic_modeを渡す)
    results, message = search_engine(query, user_genres, price_range[0], price_range[1], mode=mode_key, logic_mode=logic_mode)
    
    if message: st.caption(message)
    
    if results:
        cols_count = 3 if mode_key == "visual" else 4
        cols = st.columns(cols_count)
        
        for i, item in enumerate(results):
            with cols[i % cols_count]:
                with st.container(height=450, border=True): 
                    if item.get('image_url'): st.image(item['image_url'], use_container_width=True)
                    else: st.text("No Image")
                    
                    match_percent = int(item['match_score'] * 100)
                    price_str = f"¥{item['price']:,}"
                    
                    if mode_key == "visual":
                        st.progress(match_percent / 100, text=f"Match: {match_percent}%")
                        short_name = item['name'] if len(item['name']) < 35 else item['name'][:34] + "…"
                        st.write(f"**{short_name}**")
                        st.caption(f"{item.get('genre')} | {price_str}")
                    else:
                        st.caption(f"🏷 {item.get('genre')}")
                        short_name = item['name'] if len(item['name']) < 25 else item['name'][:24] + "…"
                        st.write(f"**{short_name}**")
                        st.write(f"{price_str}")
                        if DEBUG_MODE: st.caption(f"Score: {match_percent}%")

                    st.link_button("楽天で見る ➤", item['url'], use_container_width=True)
    else:
        st.warning("Not found... 条件を変えてDigり直してください💿")