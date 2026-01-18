import streamlit as st
import streamlit.components.v1 as components
import pickle
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import BertTokenizer, BertForSequenceClassification
import torch.nn.functional as F
import os
import traceback

# ==========================================
# ★設定エリア
# ==========================================
DEBUG_MODE = True  # デバッグ強制ON
APP_TITLE = "Sake Jacket Matcher"

GENRE_ORDER = [
    "ビール", "海外ビール", "地ビール・クラフトビール",
    "ウイスキー", "ワイン", "赤ワイン", "白ワイン", "スパークリングワイン", "シャンパン",
    "日本酒", "焼酎", "芋焼酎", "麦焼酎", "米焼酎",
    "サワーの素・割材", "リキュール", "ジン・クラフトジン", "梅酒",
    "ノンアルコール"
]

st.set_page_config(page_title=APP_TITLE, layout="wide")

# GAタグ注入 (エラー回避版)
def inject_ga():
    try:
        if "GA_ID" in st.secrets:
            GA_ID = st.secrets["GA_ID"]
            ga_code = f"""<script async src="https://www.googletagmanager.com/gtag/js?id={GA_ID}"></script><script>window.dataLayer = window.dataLayer || [];function gtag(){{dataLayer.push(arguments);}}gtag('js', new Date());gtag('config', '{GA_ID}');</script>"""
            components.html(ga_code, height=0)
    except Exception:
        pass

inject_ga()

st.markdown("""
<style>
    header {visibility: visible !important; background-color: transparent !important;}
    footer {visibility: hidden !important; display: none !important;}
    div[data-testid="stDecoration"] {visibility: hidden; display: none;}
    div[class*="viewerBadge"] {visibility: hidden !important; display: none !important;}
    .viewerBadge_container__1QSob {display: none !important;}
    div[data-testid="stImage"] img { height: 200px; object-fit: contain; width: 100%; }
</style>
""", unsafe_allow_html=True)

# --- モデル読み込み ---
@st.cache_resource
def load_all_models():
    try:
        with open('sake_database.pkl', 'rb') as f:
            db_data = pickle.load(f)
    except FileNotFoundError:
        st.error("データベース(sake_database.pkl)が見つかりません。")
        return None

    try:
        clip_model = SentenceTransformer('sentence-transformers/clip-ViT-B-32-multilingual-v1')
        all_vectors = np.concatenate([item['vector'] for item in db_data], axis=0)
    except Exception as e:
        st.error(f"CLIPモデル読み込みエラー: {e}")
        return None
    
    raw_genres = list(set([item.get('genre', 'その他') for item in db_data]))
    sorted_genres = sorted(raw_genres, key=lambda x: GENRE_ORDER.index(x) if x in GENRE_ORDER else 999)

    intent_tk, intent_md, genre_tk, genre_md = None, None, None, None
    has_logic_model = False
    try:
        if os.path.exists("./my_intent_model") and os.path.exists("./my_genre_model"):
            intent_tk = BertTokenizer.from_pretrained("./my_intent_model")
            intent_md = BertForSequenceClassification.from_pretrained("./my_intent_model")
            genre_tk = BertTokenizer.from_pretrained("./my_genre_model")
            genre_md = BertForSequenceClassification.from_pretrained("./my_genre_model")
            has_logic_model = True
    except Exception:
        pass 

    return {"db": db_data, "clip": clip_model, "vectors": all_vectors, "genres": sorted_genres, 
            "intent_tk": intent_tk, "intent_md": intent_md, "genre_tk": genre_tk, "genre_md": genre_md, 
            "has_logic_model": has_logic_model}

models = load_all_models()
if not models: st.stop()

# --- アルゴリズム関数 ---
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

def mmr_sort(query_vec, candidate_vectors, candidate_items, top_k=12, diversity=0.4):
    try:
        query_tensor = torch.tensor(query_vec).float().cpu()
        if query_tensor.dim() == 1: query_tensor = query_tensor.unsqueeze(0)
        cand_tensor = torch.tensor(candidate_vectors).float().cpu()
        
        sims_to_query = util.cos_sim(query_tensor, cand_tensor)[0]
        
        selected_indices = []
        candidate_indices = list(range(len(candidate_items)))
        
        if len(candidate_items) <= top_k:
            sorted_indices = torch.argsort(sims_to_query, descending=True).tolist()
            return [candidate_items[i] for i in sorted_indices], [sims_to_query[i].item() for i in sorted_indices]

        for _ in range(min(len(candidate_items), top_k)):
            best_mmr_score = -float('inf')
            best_idx = -1
            for idx in candidate_indices:
                similarity_to_query = sims_to_query[idx].item()
                if selected_indices:
                    selected_vecs = cand_tensor[selected_indices]
                    current_vec = cand_tensor[idx].unsqueeze(0)
                    sim_to_selected = util.cos_sim(current_vec, selected_vecs)
                    max_similarity_to_selected = torch.max(sim_to_selected).item()
                else:
                    max_similarity_to_selected = 0
                mmr_score = (1 - diversity) * similarity_to_query - diversity * max_similarity_to_selected
                if mmr_score > best_mmr_score:
                    best_mmr_score = mmr_score
                    best_idx = idx
            selected_indices.append(best_idx)
            candidate_indices.remove(best_idx)
        return [candidate_items[i] for i in selected_indices], [sims_to_query[i].item() for i in selected_indices]
    except Exception as e:
        st.error(f"MMR Error: {e}")
        return [], []

# --- 検索エンジン本体 (実況中継版) ---
def search_engine(original_query, selected_genres, min_p, max_p, mode="visual", logic_mode="A"):
    ai_message = ""
    search_genres = []
    st.write("🏃‍♂️ [STEP 1] 検索開始") # 実況1
    
    try:
        if mode == "visual" and ("C" in logic_mode or "D" in logic_mode):
            query_for_clip = f"「{original_query}」という雰囲気のお酒のボトルデザイン。 Package design of sake bottle with the vibe of {original_query}."
        else:
            query_for_clip = original_query

        if selected_genres:
            search_genres = selected_genres
        elif mode == "logic" and models["has_logic_model"]:
            # Logic省略...
            pass
        elif mode == "visual" or not models["has_logic_model"]:
            search_genres = [] 
            ai_message = ""

        # ベクトル化
        query_vec = models["clip"].encode(query_for_clip, convert_to_tensor=True).float().cpu().numpy()
        if query_vec.ndim == 1: query_vec = query_vec[None, :] 
        
        st.write(f"🏃‍♂️ [STEP 2] ベクトル化完了 Shape: {query_vec.shape}") # 実況2
        
        # フィルタリング
        valid_indices = []
        for i, item in enumerate(models["db"]):
            if search_genres and item.get('genre') not in search_genres: continue
            if not (min_p <= item['price'] <= max_p): continue
            valid_indices.append(i)
            
        st.write(f"🏃‍♂️ [STEP 3] 候補抽出完了 件数: {len(valid_indices)}") # 実況3

        if not valid_indices: 
            st.warning("⚠️ 候補が0件でした")
            return [], ai_message
        
        target_vectors = models["vectors"][valid_indices]
        candidate_items = [models["db"][i] for i in valid_indices]

        # ランキング計算
        st.write(f"🏃‍♂️ [STEP 4] ランキング計算開始 Mode: {logic_mode}") # 実況4

        if mode == "visual" and ("B" in logic_mode or "D" in logic_mode):
            results, raw_scores = mmr_sort(query_vec, target_vectors, candidate_items, top_k=12, diversity=0.4)
        else:
            # Baseline
            q_tensor = torch.tensor(query_vec).float().cpu()
            t_tensor = torch.tensor(target_vectors).float().cpu()
            
            # cos_sim の戻り値は (1, N)
            scores = util.cos_sim(q_tensor, t_tensor)
            st.write(f"🏃‍♂️ [DEBUG] Score Shape: {scores.shape}") # 追加デバッグ
            
            # (N,) に直す
            scores = scores[0] 
            
            sorted_args = torch.argsort(scores, descending=True)
            
            results = []
            raw_scores = []
            for i in range(min(12, len(sorted_args))):
                idx = sorted_args[i].item()
                results.append(candidate_items[idx])
                raw_scores.append(scores[idx].item())

        st.write("🏃‍♂️ [STEP 5] 計算完了") # 実況5

        final_results = []
        for item, raw_score in zip(results, raw_scores):
            display_score = min(raw_score * 5.0, 0.99)
            item['match_score'] = display_score
            final_results.append(item)
            
        return final_results, ai_message

    except Exception as e:
        st.error(f"🚨 システムエラー発生: {e}")
        st.code(traceback.format_exc())
        return [], "システムエラー"

# --- UI構築 ---
st.title(f"🍾 {APP_TITLE}")
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

st.sidebar.divider()
st.sidebar.markdown("### 🧪 開発者メニュー")
logic_mode = st.sidebar.selectbox("検索アルゴリズム検証", ["A: 通常 (Baseline)", "B: MMR (多様性重視)", "C: Prompt (言葉を補正)", "D: MMR + Prompt (最強?)"], index=0)
if DEBUG_MODE: st.sidebar.warning("🔧 デバッグモード ON")

col1, col2 = st.columns([3, 1], vertical_alignment="bottom")
with col1:
    placeholder = "例：サイバーパンクな夜..." 
    query = st.text_input("どんな雰囲気のお酒がいい？", placeholder=placeholder).strip()
with col2:
    search_btn = st.button("Digる", type="primary", use_container_width=True)

if query or search_btn:
    st.divider()
    # 検索実行
    results, message = search_engine(query, user_genres, price_range[0], price_range[1], mode=mode_key, logic_mode=logic_mode)
    
    if message: st.caption(message)
    
    if results:
        cols = st.columns(3)
        for i, item in enumerate(results):
            with cols[i % 3]:
                with st.container(height=450, border=True): 
                    if item.get('image_url'): st.image(item['image_url'], use_container_width=True)
                    else: st.text("No Image")
                    st.write(f"**{item['name'][:30]}**")
                    st.link_button("楽天で見る ➤", item['url'], use_container_width=True)
    else:
        # 結果が空だった場合の表示
        if message != "システムエラー":
            st.warning("⚠️ 結果が見つかりませんでした (Not Found)")