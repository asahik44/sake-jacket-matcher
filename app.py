import os
import streamlit as st
import streamlit.components.v1 as components
import pickle
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import BertTokenizer, BertForSequenceClassification
import torch.nn.functional as F
import traceback
import gc
import time
import json
import datetime
import uuid
from google.cloud import bigquery
from google.oauth2 import service_account

# ==========================================
# ★設定エリア
# ==========================================
DEBUG_MODE = False
APP_TITLE = "Sake Jacket Matcher"
APP_VERSION = "ver 1.2.6" # ★セッションID対応版
USE_LOGIC_MODEL = False

# ★BigQueryの設定
BQ_TABLE_ID = "sake-app-logs.sake_app_logs.search_logs" 

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

# ==========================================
# ★セッションIDの生成（ユーザー識別用）
# ==========================================
if "session_id" not in st.session_state:
    # まだIDがない場合（アクセスした瞬間）、ランダムなUUIDを発行して保存
    st.session_state.session_id = str(uuid.uuid4())

# デバッグ用：サイドバーにIDを表示（本番では消してもOK）
if DEBUG_MODE:
    st.sidebar.text(f"Session ID: {st.session_state.session_id}")


# --- BigQueryログ送信関数（修正版） ---
def log_to_bigquery(query_text, genres, min_p, max_p):
    """
    検索ログをBigQueryに送信する関数（環境変数読み込み版）
    """
    if not query_text: return 
    
    # ★変更点: st.secrets ではなく os.environ から直接読む
    # ファイルの先頭で import os しているので、ここはそのまま os.environ が使えます
    json_str = os.environ.get("GCP_JSON")

    if not json_str:
        # 環境変数にもない場合は、念のため st.secrets も見てみる（バックアップ）
        try:
            if "GCP_JSON" in st.secrets:
                json_str = st.secrets["GCP_JSON"]
        except Exception:
            pass
    
    # それでもなければエラー表示して終了
    if not json_str:
        if DEBUG_MODE: st.sidebar.error("⚠️ Secret 'GCP_JSON' not found in env.")
        return

    try:
        # 文字列のJSONを辞書データに変換
        key_dict = json.loads(json_str)
        
        creds = service_account.Credentials.from_service_account_info(key_dict)
        client = bigquery.Client(credentials=creds, project=key_dict["project_id"])

        rows_to_insert = [{
            "timestamp": datetime.datetime.now().isoformat(),
            "session_id": st.session_state.session_id,
            "query": query_text,
            "genres": ",".join(genres) if genres else "All",
            "min_price": min_p,
            "max_price": max_p
        }]

        errors = client.insert_rows_json(BQ_TABLE_ID, rows_to_insert)
        
        if errors:
            if DEBUG_MODE: st.sidebar.error(f"BQ Error: {errors}")
            print(f"BQ Insert Error: {errors}")
        else:
            if DEBUG_MODE: st.sidebar.success("Log saved!")
            print(f"Log saved: {query_text} (ID: {st.session_state.session_id})")

    except Exception as e:
        print(f"BigQuery Connection Error: {e}")
        if DEBUG_MODE: st.sidebar.error(f"BQ Exception: {e}")

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
def mmr_sort(query_vec, candidate_vectors_tensor, candidate_items, top_k=12, diversity=0.8):
    try:
        PRE_FILTER_K = 2000 
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
        else:
            search_genres = [] 

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

        if mode == "visual" and ("B" in logic_mode or "D" in logic_mode):
            results, raw_scores = mmr_sort(query_vec, target_vectors_tensor, candidate_items, top_k=12, diversity=0.8)
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
mode_key = "visual"

st.sidebar.divider()
st.sidebar.header("Filters")
user_genres = st.sidebar.multiselect("ジャンル固定", options=models["genres"])
price_range = st.sidebar.slider("価格帯", 0, 100000, (0, 100000), 1000, format="¥%d")

logic_mode = "B: MMR (多様性重視)"

col1, col2 = st.columns([3, 1], vertical_alignment="bottom")
with col1:
    placeholder = "例：サイバーパンクな夜,森の中で読書,初恋の味..." 
    query = st.text_input("どんな雰囲気のお酒がいい？", placeholder=placeholder).strip()
with col2:
    search_btn = st.button("Digる", type="primary", use_container_width=True)

if query or search_btn:
    st.query_params.from_dict({"q": query})

    # ★修正：重複送信防止（時間 ＋ 検索ワードの一致チェック）
    if "last_log_time" not in st.session_state:
        st.session_state.last_log_time = 0.0
    if "last_logged_query" not in st.session_state:
        st.session_state.last_logged_query = ""
    
    current_time = time.time()
    
    # 条件: 「5秒以上経過している」 または 「検索ワードが前回と違う」 場合のみ送る
    is_time_passed = (current_time - st.session_state.last_log_time > 5.0)
    is_new_query = (query != st.session_state.last_logged_query)

    if is_time_passed or is_new_query:
        # 先にステートを更新（ロック）して、二重送信を防ぐ
        st.session_state.last_log_time = current_time
        st.session_state.last_logged_query = query
        
        # その後に送信処理
        log_to_bigquery(query, user_genres, price_range[0], price_range[1])
    else:
        if DEBUG_MODE: st.sidebar.warning("⚠️ Skipping duplicate log")

    st.divider()
    status_text = st.empty()
    progress_bar = st.progress(0)
    
    with st.spinner('AIが脳みそフル回転中...'):
        results, message = search_engine(query, user_genres, price_range[0], price_range[1], mode=mode_key, logic_mode=logic_mode, progress_bar=progress_bar, status_text=status_text)
    
    time.sleep(0.2)
    progress_bar.empty()
    status_text.empty()
    
    # ...（以下同じなので省略、もしコピーミスが不安なら元のままでもUI部分は動きに影響しません）
    if message: st.caption(message)
    
    if results:
        cols = st.columns(3)
        for i, item in enumerate(results):
            with cols[i % 3]:
                with st.container(height=450, border=True): 
                    if item.get('image_url'): st.image(item['image_url'], use_container_width=True)
                    else: st.text("No Image")
                    
                    st.progress(item['match_score'], text=f"Match: {int(item['match_score']*100)}%")
                    st.write(f"**{item['name'][:30]}**")
                    price_str = f"¥{item['price']:,}"
                    st.caption(f"🏷 {item.get('genre')} | 💰 {price_str}")
                    st.link_button("楽天で見る ➤", item['url'], use_container_width=True)
    else:
        if message != "システムエラー":
            st.warning("⚠️ 結果が見つかりませんでした")

# --- サイドバー：利用規約とプライバシーポリシー ---
with st.sidebar.expander("ℹ️ 利用規約・プライバシーポリシー"):
    st.markdown("""
    **1. データの収集について**
    当アプリでは、サービス向上のため以下の情報を取得・保存します。
    - 入力された検索キーワード、選択されたフィルタ情報
    - サイトの利用状況（Google Analyticsを使用）
    - セッション識別子（個人を特定しないランダムなID）
    
    **2. Google Analyticsの使用**
    当アプリはアクセス解析のためにGoogle Analyticsを使用しています。データ収集のためにCookieを使用しますが、個人を特定する情報は含まれません。
    
    **3. 免責事項**
    - 検索ボックスには、個人名や電話番号などの**個人情報は絶対に入力しないでください**。
    - 当アプリの利用により生じた損害について、開発者は一切の責任を負いません。
    - 商品情報は楽天API等を利用していますが、最新の価格や在庫状況はリンク先の店舗でご確認ください。
    
    **4. お問い合わせ**
    不具合や削除依頼は [開発者のX (Twitter)](https://x.com/asahirk44) までご連絡ください。
    """)