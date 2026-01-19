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
import gc
import time 

# ==========================================
# ★設定エリア
# ==========================================
DEBUG_MODE = False
APP_TITLE = "Sake Jacket Matcher"
APP_VERSION = "ver 1.0.4" # ★バージョン更新 検索ワード検知機能追加
USE_LOGIC_MODEL = False

GENRE_ORDER = [
    "ビール", "海外ビール", "地ビール・クラフトビール",
    "ウイスキー", "ワイン", "赤ワイン", "白ワイン", "スパークリングワイン", "シャンパン",
    "日本酒", "焼酎", "芋焼酎", "麦焼酎", "米焼酎",
    "リキュール", "ジン・クラフトジン", "梅酒",
    "ノンアルコール","サワーの素・割材"
]

st.set_page_config(
    page_title="Sake Jacket Matcher | AIで直感的にジャケ買い", 
    layout="wide",
    page_icon="https://sake-jaket.herahin.net/sake_favicon.png"
)
st.sidebar.caption(f"App Version: {APP_VERSION}")

def inject_ga():
    try:
        if "GA_ID" in st.secrets:
            GA_ID = st.secrets["GA_ID"]
        elif "GA_ID" in os.environ:
            GA_ID = os.environ["GA_ID"]
        else:
            return

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
        raw_vectors = np.concatenate([item['vector'] for item in db_data], axis=0)
        all_vectors_tensor = torch.tensor(raw_vectors).float().cpu()
        del raw_vectors
        gc.collect()
    except Exception as e:
        st.error(f"CLIPモデル読み込みエラー: {e}")
        return None
    
    raw_genres = list(set([item.get('genre', 'その他') for item in db_data]))
    sorted_genres = sorted(raw_genres, key=lambda x: GENRE_ORDER.index(x) if x in GENRE_ORDER else 999)

    if USE_LOGIC_MODEL: 
        try:
            if os.path.exists("./my_intent_model") and os.path.exists("./my_genre_model"):
                intent_tk = BertTokenizer.from_pretrained("./my_intent_model")
                intent_md = BertForSequenceClassification.from_pretrained("./my_intent_model")
                genre_tk = BertTokenizer.from_pretrained("./my_genre_model")
                genre_md = BertForSequenceClassification.from_pretrained("./my_genre_model")
                has_logic_model = True
        except Exception:
            pass

    intent_tk, intent_md, genre_tk, genre_md = None, None, None, None
    has_logic_model = False
    
    result = {
        "db": db_data,
        "clip": clip_model,
        "vectors": all_vectors_tensor,
        "genres": sorted_genres,
        "intent_tk": intent_tk, 
        "intent_md": intent_md, 
        "genre_tk": genre_tk, 
        "genre_md": genre_md, 
        "has_logic_model": has_logic_model
    }
    gc.collect()
    return result

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

# MMRロジック
def mmr_sort(query_vec, candidate_vectors_tensor, candidate_items, top_k=12, diversity=0.4):
    try:
        PRE_FILTER_K = 300 
        
        query_tensor = torch.tensor(query_vec).float().cpu()
        if query_tensor.dim() == 1: query_tensor = query_tensor.unsqueeze(0)
        
        all_sims = util.cos_sim(query_tensor, candidate_vectors_tensor)[0]
        
        if len(candidate_items) > PRE_FILTER_K:
            top_indices = torch.argsort(all_sims, descending=True)[:PRE_FILTER_K]
            candidate_vectors_tensor = candidate_vectors_tensor[top_indices]
            candidate_items = [candidate_items[i] for i in top_indices.tolist()]
            sims_to_query = all_sims[top_indices]
        else:
            sims_to_query = all_sims

        selected_indices = []
        candidate_indices = list(range(len(candidate_items)))
        
        for _ in range(min(len(candidate_items), top_k)):
            best_mmr_score = -float('inf')
            best_idx = -1
            
            for idx in candidate_indices:
                similarity_to_query = sims_to_query[idx].item()
                
                if selected_indices:
                    selected_vecs = candidate_vectors_tensor[selected_indices]
                    current_vec = candidate_vectors_tensor[idx].unsqueeze(0)
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

# --- 検索エンジン本体 ---
def search_engine(original_query, selected_genres, min_p, max_p, mode="visual", logic_mode="A", progress_bar=None, status_text=None):
    ai_message = ""
    search_genres = []
    
    if progress_bar: progress_bar.progress(10)
    if status_text: status_text.text("🤔 キーワードを解析中...")
    
    try:
        if mode == "visual" and ("C" in logic_mode or "D" in logic_mode):
            query_for_clip = f"「{original_query}」という雰囲気のお酒のボトルデザイン。 Package design of sake bottle with the vibe of {original_query}."
        else:
            query_for_clip = original_query

        if progress_bar: progress_bar.progress(30)
        if status_text: status_text.text("🎨 イメージをベクトルに変換中...")

        if selected_genres:
            search_genres = selected_genres
        elif mode == "logic" and models["has_logic_model"]:
            pass
        elif mode == "visual" or not models["has_logic_model"]:
            search_genres = [] 
            ai_message = ""

        query_vec = models["clip"].encode(query_for_clip, convert_to_tensor=True).float().cpu().numpy()
        if query_vec.ndim == 1: query_vec = query_vec[None, :] 
        
        if progress_bar: progress_bar.progress(50)
        if status_text: status_text.text("🍷 データベースから候補を抽出中...")

        valid_indices = []
        for i, item in enumerate(models["db"]):
            if search_genres and item.get('genre') not in search_genres: continue
            if not (min_p <= item['price'] <= max_p): continue
            valid_indices.append(i)
            
        if not valid_indices: 
            return [], ai_message
        
        target_vectors_tensor = models["vectors"][valid_indices]
        candidate_items = [models["db"][i] for i in valid_indices]

        if progress_bar: progress_bar.progress(70)
        if status_text: status_text.text(f"🚀 {len(candidate_items)}件の中からベストマッチを選定中...")

        # ランキング計算
        if mode == "visual" and ("B" in logic_mode or "D" in logic_mode):
            results, raw_scores = mmr_sort(query_vec, target_vectors_tensor, candidate_items, top_k=12, diversity=0.4)
        else:
            q_tensor = torch.tensor(query_vec).float().cpu()
            scores = util.cos_sim(q_tensor, target_vectors_tensor)
            scores = scores[0] 
            sorted_args = torch.argsort(scores, descending=True)
            results = []
            raw_scores = []
            for i in range(min(12, len(sorted_args))):
                idx = sorted_args[i].item()
                results.append(candidate_items[idx])
                raw_scores.append(scores[idx].item())

        if progress_bar: progress_bar.progress(100)
        if status_text: status_text.text("✨ 完了！")
        time.sleep(0.5) 

        # スコア正規化
        if raw_scores:
            max_s = max(raw_scores)
            min_s = min(raw_scores)
            normalized_scores = []
            
            if max_s == min_s:
                normalized_scores = [0.99] * len(raw_scores)
            else:
                for s in raw_scores:
                    norm = (s - min_s) / (max_s - min_s)
                    scaled = 0.70 + (norm * 0.29)
                    normalized_scores.append(scaled)
        else:
            normalized_scores = []

        final_results = []
        for item, score in zip(results, normalized_scores):
            item['match_score'] = score
            final_results.append(item)

        # ★最後にスコアが高い順に並び替える
        final_results.sort(key=lambda x: x['match_score'], reverse=True)
            
        return final_results, ai_message

    except Exception as e:
        st.error(f"🚨 システムエラー: {e}")
        st.code(traceback.format_exc())
        return [], "システムエラー"

# --- UI構築 ---
st.title(f"🍾 {APP_TITLE}")
st.caption(f"Released: {APP_VERSION}") 

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

if DEBUG_MODE:
    st.sidebar.divider()
    st.sidebar.markdown("### 🧪 開発者メニュー")
    logic_mode = st.sidebar.selectbox("検索アルゴリズム検証", ["A: 通常 (Baseline)", "B: MMR (多様性重視)", "C: Prompt (言葉を補正)", "D: MMR + Prompt (最強?)"], index=1)
    st.sidebar.warning("🔧 デバッグモード ON")
else:
    logic_mode = "B: MMR (多様性重視)"

col1, col2 = st.columns([3, 1], vertical_alignment="bottom")
with col1:
    placeholder = "例：サイバーパンクな夜,森の中で読書,初恋の味..." 
    query = st.text_input("どんな雰囲気のお酒がいい？", placeholder=placeholder).strip()
with col2:
    search_btn = st.button("Digる", type="primary", use_container_width=True)

if query or search_btn:
    # ★URLに検索ワードを記録する
    st.query_params.from_dict({"q": query})

    st.divider()
    
    status_text = st.empty()
    progress_bar = st.progress(0)
    
    with st.spinner('AIが脳みそフル回転中...'):
        results, message = search_engine(query, user_genres, price_range[0], price_range[1], mode=mode_key, logic_mode=logic_mode, progress_bar=progress_bar, status_text=status_text)
    
    time.sleep(0.2)
    progress_bar.empty()
    status_text.empty()

    if message: st.caption(message)
    
    if results:
        cols = st.columns(3)
        for i, item in enumerate(results):
            with cols[i % 3]:
                with st.container(height=450, border=True): 
                    if item.get('image_url'): st.image(item['image_url'], use_container_width=True)
                    else: st.text("No Image")
                    
                    if mode_key == "visual":
                        st.progress(item['match_score'], text=f"Match: {int(item['match_score']*100)}%")
                    
                    st.write(f"**{item['name'][:30]}**")
                    price_str = f"¥{item['price']:,}"
                    st.caption(f"🏷 {item.get('genre')} | 💰 {price_str}")
                    st.link_button("楽天で見る ➤", item['url'], use_container_width=True)
    else:
        if message != "システムエラー":
            st.warning("⚠️ 結果が見つかりませんでした")