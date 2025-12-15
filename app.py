import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import warnings
warnings.filterwarnings('ignore')

# ============================
# CONFIG
# ============================
MODEL_NAME = "bert-base-uncased"
MAX_LEN = 128
EMBED_DIM = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================
# MODEL CLASS
# ============================
class BertEmbedForMultiLabel(nn.Module):
    def __init__(self, model_name, num_labels, embed_dim=128):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden = self.bert.config.hidden_size
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_labels)
        )
        
        self.embed_head = nn.Sequential(
            nn.Linear(hidden, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, embed_dim)
        )
    
    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        pooled = out.pooler_output
        logits = self.classifier(pooled)
        emb = self.embed_head(pooled)
        emb = nn.functional.normalize(emb, p=2, dim=1)
        return logits, emb

# ============================
# LOAD MODEL & DATA (CACHED)
# ============================
@st.cache_resource
def load_model_and_data():
    """Load model, tokenizer, embeddings, and product data"""
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained("tokenizer_saved")
        
        # Load model
        model = BertEmbedForMultiLabel(MODEL_NAME, num_labels=5, embed_dim=EMBED_DIM)
        model.load_state_dict(torch.load("best_model.pt", map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        
        # Load embeddings and product index
        embeddings = np.load("product_embeddings.npy")
        product_df = pd.read_pickle("product_index.pkl")
        
        return tokenizer, model, embeddings, product_df
    except Exception as e:
        st.error(f"Error loading model/data: {str(e)}")
        return None, None, None, None

# ============================
# HELPER FUNCTIONS
# ============================
def get_query_embedding(query_text, tokenizer, model):
    """Generate embedding for query text"""
    with torch.no_grad():
        enc = tokenizer(
            query_text,
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt"
        )
        input_ids = enc["input_ids"].to(DEVICE)
        attention_mask = enc["attention_mask"].to(DEVICE)
        
        _, embedding = model(input_ids=input_ids, attention_mask=attention_mask)
        return embedding.cpu().numpy()

def cosine_similarity(query_emb, product_embs):
    """Calculate cosine similarity between query and all products"""
    # query_emb shape: (1, 128)
    # product_embs shape: (N, 128)
    dot_product = np.dot(product_embs, query_emb.T).squeeze()
    return dot_product

def filter_by_allergens(df, allergens):
    """Filter out products containing allergen ingredients - FIXED VERSION"""
    if not allergens:
        return df
    
    # IMPORTANT: Use df.index to align with the dataframe
    mask = pd.Series([True] * len(df), index=df.index)
    
    for allergen in allergens:
        # More robust allergen checking
        allergen_lower = allergen.lower().strip()
        allergen_mask = ~df['ingredients_list'].fillna('').astype(str).str.lower().str.contains(
            allergen_lower, 
            case=False, 
            na=False,
            regex=False
        )
        mask &= allergen_mask
    
    return df[mask]

def build_query_text(skin_types, product_types, problems):
    """Build query text from user inputs"""
    parts = []
    
    if skin_types:
        parts.append(f"Skin type: {', '.join(skin_types)}")
    
    if product_types:
        parts.append(f"Product: {', '.join(product_types)}")
    
    if problems:
        parts.append(f"Problems: {', '.join(problems)}")
    
    return ". ".join(parts)

# ============================
# STREAMLIT UI
# ============================
def main():
    st.set_page_config(
        page_title="Skincare Recommendation System",
        page_icon="🧴",
        layout="wide"
    )
    
    # Custom CSS theme
    st.markdown("""
        <style>
        .stApp {
            background: linear-gradient(135deg, #ffeef8 0%, #e0f4ff 100%);
        }

        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #ffe5f1 0%, #e5f3ff 100%);
        }

        /* =========================
        TEXT COLOR FIX (IMPORTANT)
        ========================= */
        .stApp,
        .stMarkdown,
        .stText,
        p, li, span, label {
            color: #222222 !important;
        }

        /* Headers */
        h1 {
            color: #d4568c !important;
            text-shadow: 2px 2px 4px rgba(212, 86, 140, 0.2);
        }

        h2, h3, h4 {
            color: #9c5d8a !important;
        }

        /* Text inside cards */
        .element-container {
            color: #222222 !important;
        }

        /* Columns text */
        [data-testid="column"] {
            color: #222222 !important;
        }

        /* =========================
        BUTTON
        ========================= */
        .stButton > button {
            background: linear-gradient(135deg, #ff9ec5 0%, #a8d8ff 100%);
            color: white !important;
            border: none;
            border-radius: 20px;
            padding: 0.5rem 2rem;
            font-weight: bold;
            box-shadow: 0 4px 6px rgba(212, 86, 140, 0.3);
            transition: all 0.3s ease;
        }

        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(212, 86, 140, 0.4);
        }

        /* =========================
        METRICS
        ========================= */
        [data-testid="stMetricLabel"] {
            color: #444444 !important;
        }

        [data-testid="stMetricValue"] {
            color: #222222 !important;
            font-weight: bold;
        }

        /* =========================
        ALERT BOXES
        ========================= */
        .stAlert p {
            color: #222222 !important;
        }

        /* =========================
        INGREDIENTS TEXTAREA
        ========================= */
        .stTextArea textarea {
            background: rgba(255, 255, 255, 0.9);
            border: 2px solid #ffe5f1;
            border-radius: 10px;
            color: #222222 !important;
        }
        /* =========================
        SELECTBOX & MULTISELECT FIX TOTAL
        ========================= */

        /* BOX UTAMA (yang kelihatan sebelum diklik) */
        div[data-baseweb="select"] > div {
            background-color: #ffffff !important;
            color: #222222 !important;
            border-radius: 14px !important;
            border: 2px solid #ffd1e6 !important;
            min-height: 48px;
        }

        /* Text di dalam box */
        div[data-baseweb="select"] span {
            color: #222222 !important;
            font-weight: 500;
        }

        /* =========================
        SELECTBOX - FOCUSED / ACTIVE STATE
        ========================= */
        div[data-baseweb="select"]:focus-within > div {
            background-color: #fff0f7 !important; /* PINK MUDA */
            color: #555555 !important;            /* TULISAN ABU TUA */
            border: 2px solid #ffb6d5 !important;
        }

        /* =========================
        DROPDOWN CONTAINER
        ========================= */
        div[data-baseweb="popover"] {
            background-color: #fff0f7 !important;
        }

        /* DROPDOWN MENU (yang tadinya abu tua) */
        div[data-baseweb="menu"] {
            background-color: #fff0f7 !important;
            border-radius: 14px !important;
            padding: 6px !important;
        }

        /* ITEM DI DALAM DROPDOWN */
        div[data-baseweb="option"] {
            background-color: #fff0f7 !important;
            color: #222222 !important;
            border-radius: 10px !important;
            margin: 4px 0 !important;
        }

        /* HOVER ITEM */
        div[data-baseweb="option"]:hover {
            background-color: #ffd6e8 !important;
        }

        /* =========================
        SELECTED ITEM (CHIP)
        ========================= */
        div[data-baseweb="tag"] {
            background-color: #ffd6e8 !important;   /* PINK MUDA */
            color: #7a2c52 !important;
            border-radius: 18px !important;
            font-weight: 600 !important;
            padding: 4px 10px !important;
        }

        /* ICON X (hapus pilihan) */
        div[data-baseweb="tag"] svg {
            color: #7a2c52 !important;
        }

        /* =========================
        PLACEHOLDER TEXT
        ========================= */
        div[data-baseweb="select"] input::placeholder {
            color: #999999 !important;
        }

        /* =========================
        INGREDIENTS TEXTAREA
        ========================= */
        .stTextArea textarea {
            background-color: #ffffff !important;
            color: #222222 !important;
            border-radius: 10px !important;
        }

        </style>
    """, unsafe_allow_html=True)

    
    st.title("🧴 MilihSkincare")
    st.markdown("### ✨ Find Your Perfect Skincare Match ✨")
    st.markdown("---")
    
    # Load model and data
    with st.spinner("💄 Loading model and data..."):
        tokenizer, model, embeddings, product_df = load_model_and_data()
    
    if tokenizer is None:
        st.error("Failed to load model. Please check if all required files exist.")
        st.stop()
    
    st.success("✅ Model loaded successfully!")
    
    # Sidebar for inputs
    st.sidebar.header("🔍 Product Search Criteria")
    
    # 1. Skin Type Selection (SINGLE SELECT)
    st.sidebar.subheader("1. Select Your Skin Type")
    skin_type_options = ["Combination", "Dry", "Normal", "Oily", "Sensitive"]
    selected_skin_type = st.sidebar.selectbox(
        "Choose your skin type:",
        [""] + skin_type_options,
        index=0
    )
    selected_skin_types = [selected_skin_type] if selected_skin_type else []
    
    # 2. Product Type Selection
    st.sidebar.subheader("2. Select Product Type(s)")
    product_type_options = sorted(product_df['Label'].dropna().unique().tolist())
    selected_product_types = st.sidebar.multiselect(
        "Choose one or more:",
        product_type_options,
        default=[]
    )
    
    # 3. Skin Problems Selection
    st.sidebar.subheader("3. Select Skin Problem(s)")
    problem_options = ['Fine Lines', 'Pigmentation', 'Radiance', 'Wrinkles', 
                       'Acne', 'Dark Spots', 'Dullness', 'Uneven Texture']
    selected_problems = st.sidebar.multiselect(
        "Choose one or more:",
        problem_options,
        default=[]
    )
    
    # 4. Allergen Selection
    st.sidebar.subheader("4. Select Allergen Ingredients (to avoid)")
    common_allergens = ['Alcohol', 'Fragrance', 'Parabens', 'Sulfates', 
                        'Alcohol Denat', 'Parfum', 'Dimethicone', 'Silicones']
    
    selected_allergens = st.sidebar.multiselect(
        "Choose ingredients to avoid:",
        sorted(common_allergens),
        default=[]
    )
    
    st.sidebar.markdown("---")
    search_button = st.sidebar.button("🔍 Search Products", type="primary", use_container_width=True)
    
    # Main content area
    if search_button:
        if not selected_skin_types:
            st.warning("⚠️ Please select your skin type!")
        else:
            with st.spinner("🔍 Searching for the best products for you..."):
                # Build query text
                query_text = build_query_text(selected_skin_types, selected_product_types, selected_problems)
                
                st.info(f"**Search Query:** {query_text}")
                
                # Get query embedding
                query_emb = get_query_embedding(query_text, tokenizer, model)
                
                # Calculate similarities
                similarities = cosine_similarity(query_emb, embeddings)
                
                # Add similarities to dataframe
                results_df = product_df.copy()
                results_df['similarity'] = similarities
                
                # Filter by skin type (at least one match)
                skin_type_mask = pd.Series([False] * len(results_df))
                for skin_type in selected_skin_types:
                    if skin_type in results_df.columns:
                        skin_type_mask |= (results_df[skin_type] == 1)
                
                results_df = results_df[skin_type_mask]
                
                # Filter by product type if selected
                if selected_product_types:
                    results_df = results_df[results_df['Label'].isin(selected_product_types)]
                
                # Filter by allergens
                results_df = filter_by_allergens(results_df, selected_allergens)
                
                # Sort by similarity and get top 10
                results_df = results_df.sort_values('similarity', ascending=False).head(10)
            
            # Display results (outside spinner so it shows after loading)
            st.markdown("---")
            st.header(f"🎯 Top 10 Recommended Products")
            st.markdown(f"*Found {len(results_df)} matching products*")
            
            if len(results_df) == 0:
                st.warning("No products found matching your criteria. Try adjusting your filters.")
            else:
                for idx, (_, row) in enumerate(results_df.iterrows(), 1):
                    # Product card container
                    st.markdown(f"""
                        <div style="background: rgba(255, 255, 255, 0.9); 
                             border-radius: 15px; padding: 1.5rem; margin: 1rem 0;
                             box-shadow: 0 4px 6px rgba(156, 93, 138, 0.15);
                             border-left: 5px solid #ff9ec5;">
                    """, unsafe_allow_html=True)
                    
                    # Product number badge
                    st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #ff9ec5 0%, #a8d8ff 100%); 
                             color: white; padding: 0.5rem 1rem; border-radius: 20px; 
                             display: inline-block; font-weight: bold; margin-bottom: 1rem;">
                            #{idx} Recommendation
                        </div>
                    """, unsafe_allow_html=True)
                    
                    col1, col2 = st.columns([1, 3])
                    
                    with col1:
                        st.markdown(f"### 💄 {row['brand']}")
                        
                        # Display skin type badge
                        suitable_skin_types = []
                        for skin_type in skin_type_options:
                            if skin_type in row and row[skin_type] == 1:
                                suitable_skin_types.append(skin_type)
                        
                        if suitable_skin_types:
                            st.markdown("**🌸 Suitable for:**")
                            for skin_type in suitable_skin_types:
                                st.markdown(f"- {skin_type}")
                    
                    with col2:
                        st.markdown(f"### {row['product_name']}")
                        
                        # Display "Good For" information
                        if pd.notna(row.get('who_is_it_good_for')):
                            st.markdown("**✨ Good For:**")
                            good_for_text = str(row['who_is_it_good_for'])
                            st.markdown(f"*{good_for_text}*")
                        
                        # Display product description if available
                        if pd.notna(row.get('short_description')):
                            st.markdown("**📝 Description:**")
                            st.markdown(f"_{row['short_description']}_")
                        
                        # Display ingredients
                        st.markdown("**🧪 Ingredients:**")
                        ingredients = str(row['ingredients_list']) if pd.notna(row['ingredients_list']) else "N/A"
                        st.text_area(
                            f"ingredients_{idx}", 
                            ingredients, 
                            height=100, 
                            disabled=True,
                            label_visibility="collapsed"
                        )
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                    st.markdown("---")
    
    else:
        # Display welcome message
        st.markdown("""
            <div style="background: rgba(255, 255, 255, 0.8); padding: 2rem; 
                 border-radius: 20px; text-align: center; margin: 2rem 0;
                 box-shadow: 0 4px 12px rgba(212, 86, 140, 0.2);">
                <h2 style="color: #d4568c;">💖 Welcome to Your Skincare Journey! 💖</h2>
                <p style="font-size: 1.1rem; color: #9c5d8a;">
                    Select your criteria from the sidebar and discover products perfect for you!
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        # Display some stats
        st.markdown("### 📊 Our Collection")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("💄 Total Products", len(product_df))
        with col2:
            st.metric("🌸 Product Categories", product_df['Label'].nunique())
        with col3:
            st.metric("✨ Brands", product_df['brand'].nunique())

if __name__ == "__main__":
    main()