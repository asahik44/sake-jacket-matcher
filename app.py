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
DEBUG_MODE = False
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

st.markdown("""
<style>
    /* 画像サイズの調整 */
    div[data-testid="stImage"] img { height: 200px; object-fit: contain; width: 100%; }
        
    /* ★ハンバーガーメニューとフッターを隠す */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
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
        "has_logic_model": has_logic_model # フラグを追加
    }

models = load_all_models()
if not models:
    st.stop()

# --- 推論ロジック ---
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

# 検索エンジン
def search_engine(query, selected_genres, min_p, max_p, mode="visual"):
    ai_message = ""
    search_genres = []
    
    if selected_genres:
        search_genres = selected_genres
    elif mode == "logic" and models["has_logic_model"]:
        target_genres = []
        for broad_key, children in BROAD_CATEGORIES.items():
            if broad_key in query: target_genres.extend(children)
        for g in models["genres"]:
            if g in query and g not in target_genres: target_genres.append(g)
                
        if target_genres:
            search_genres = list(set(target_genres))
            ai_message = f"キーワードから {len(search_genres)}ジャンル に絞りました" if DEBUG_MODE else "キーワードからジャンルを絞り込みました"
        else:
            is_nonal, nonal_conf = predict_intent(query)
            if is_nonal:
                search_genres = ["ノンアルコール"]
                msg = f"Logic: ノンアルコール検知 ({int(nonal_conf*100)}%)" if DEBUG_MODE else "ノンアルコール商品から探します"
                ai_message = msg
            else:
                genre_probs = predict_genre_probs(query)
                sorted_genres = sorted(genre_probs.items(), key=lambda x: x[1], reverse=True)
                candidates = [sorted_genres[0][0]]
                for g, p in sorted_genres[1:]:
                    if p > 0.15: candidates.append(g)
                search_genres = candidates
                msg = f"Logic: {search_genres[0]} などを推論" if DEBUG_MODE else f"AI推論: {search_genres[0]} などが合いそうです"
                ai_message = msg

    elif mode == "visual" or not models["has_logic_model"]:
        search_genres = [] 
        ai_message = "Free Vibe: ジャケットの雰囲気だけで全ジャンルから探します"

    query_vec = models["clip"].encode(query, convert_to_tensor=True).cpu().numpy()
    
    valid_indices = []
    for i, item in enumerate(models["db"]):
        if search_genres and item.get('genre') not in search_genres: continue
        if not (min_p <= item['price'] <= max_p): continue
        valid_indices.append(i)
        
    if not valid_indices: return [], ai_message
        
    target_vectors = models["vectors"][valid_indices]
    scores = util.cos_sim(query_vec, target_vectors)[0]
    sorted_args = torch.argsort(scores, descending=True)
    
    results = []
    for i in range(min(12, len(sorted_args))):
        idx = sorted_args[i].item()
        original_idx = valid_indices[idx]
        item = models["db"][original_idx]
        
        raw = scores[idx].item()
        # visualモードまたはモデルなしなら係数高め
        display_score = min(raw * 5.0, 0.99) if (mode == "visual" or not models["has_logic_model"]) else min(raw * 3.5, 0.99)
            
        item['match_score'] = display_score
        results.append(item)
    return results, ai_message

# --- UI構築 ---
st.title(f"🍾 {APP_TITLE}")

# サイドバー
st.sidebar.header("Search Mode")

# ★モデルの有無でモード選択肢を変える
if models["has_logic_model"]:
    mode_options = ("ジャケ買い (感性)", "AIソムリエ (知識)")
else:
    mode_options = ("ジャケ買い (感性)",) # モデルがないときはこれ一択

mode_select = st.sidebar.radio("検索モード", mode_options, index=0)
mode_key = "visual" if "ジャケ買い" in mode_select else "logic"

st.sidebar.divider()
st.sidebar.header("Filters")
user_genres = st.sidebar.multiselect("ジャンル固定", options=models["genres"])
price_range = st.sidebar.slider("価格帯", 0, 30000, (0, 30000), 500, format="¥%d")

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
        components.html(f"<script>gtag('event', 'search', {{'search_term': '{query}'}});</script>", height=0)

    st.divider()
    results, message = search_engine(query, user_genres, price_range[0], price_range[1], mode=mode_key)
    
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