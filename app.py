import streamlit as st
import pandas as pd
import numpy as np
from pycaret.classification import load_model, predict_model
from datetime import datetime, date
import plotly.express as px
import plotly.graph_objects as go 
from PIL import Image 

# --- Configuração da Página e Constantes ---
try:
    page_icon_img = Image.open("image/icon_ifood.jpeg")
except FileNotFoundError:
    page_icon_img = "🛒" 
st.set_page_config(
    page_title="iFood - Previsor de Propensão de Compra",
    page_icon=page_icon_img,
    layout="wide"
)

# --- TÍTULO PRINCIPAL DA APLICAÇÃO ---
st.title("Previsor de Propensão de Compra iFood") # O título deve aparecer aqui

MODEL_PATH = './pickle/modelo_LGBM' 
DATA_REFERENCIA_TREINO_STR = "2014-06-29"
MEDIANA_INCOME_TREINO = 51381.5 

# --- Inicialização do Estado da Sessão ---
if 'threshold' not in st.session_state:
    st.session_state.threshold = 0.50
if 'uploaded_df_original' not in st.session_state:
    st.session_state.uploaded_df_original = None
if 'predictions_df_full' not in st.session_state: 
    st.session_state.predictions_df_full = None
if 'show_online_prediction' not in st.session_state:
    st.session_state.show_online_prediction = False
if 'online_prediction_label' not in st.session_state:
    st.session_state.online_prediction_label = None
if 'online_score_classe1' not in st.session_state: 
    st.session_state.online_score_classe1 = None
if 'online_history' not in st.session_state:
    st.session_state.online_history = []

# --- Carregar Modelo ---
@st.cache_resource
def carregar_modelo_pycaret(model_path):
    try:
        modelo = load_model(model_path)
        return modelo
    except Exception as e:
        print(f"Erro crítico ao carregar o modelo de '{model_path}': {e}") 
        return None

modelo_preditivo = carregar_modelo_pycaret(MODEL_PATH)

# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    with st.container(border=True): 
        col_sb_buf1, col_sb_logo, col_sb_buf2 = st.columns([1, 3, 1]) 
        with col_sb_logo:
            try:
                logo_sidebar_img = Image.open("image/ifood_app.png")
                st.image(logo_sidebar_img, width=180) 
            except FileNotFoundError:
                st.warning("Logo 'ifood_app.png' não encontrado.")
    
    st.title("Painel de Controle") 
    st.markdown("Ajuste as configurações de predição.")
    st.divider() 

    st.header("Threshold de Decisão")
    # Função de callback para mudança de threshold
    def on_threshold_change_callback():
        if st.session_state.get('predictions_df_full') is not None and \
           'score_classe1' in st.session_state.predictions_df_full.columns:
            df = st.session_state.predictions_df_full.copy()
            df['prediction_label'] = (df['score_classe1'] >= st.session_state.threshold).astype(int)
            st.session_state.predictions_df_full = df
        if st.session_state.get('show_online_prediction') and \
           st.session_state.get('online_score_classe1') is not None:
            st.session_state.online_prediction_label = \
                (st.session_state.online_score_classe1 >= st.session_state.threshold).astype(int)

    st.number_input(
        "Defina o Limiar (Probabilidade Mínima):",
        min_value=0.01, max_value=0.99,
        step=0.01, format="%.2f", 
        key='threshold', 
        on_change=on_threshold_change_callback, 
        help="Probabilidade mínima para classificar um cliente como 'Propenso a Comprar (1)'."
    )

    with st.expander("ℹ️ Como funciona o Threshold?", expanded=False):
        st.markdown("""O threshold (limiar) define a sensibilidade da classificação:
        - **Threshold mais baixo:** Mais clientes são classificados como "propensos". Aumenta a chance de encontrar todos que realmente comprariam (**maior Revocação**), mas também o risco de classificar incorretamente os que não comprariam (**menor Precisão**).
        - **Threshold mais alto:** Menos clientes são classificados como "propensos". Aumenta a chance de que os "propensos" realmente comprem (**maior Precisão**), mas pode perder alguns que comprariam (**menor Revocação**).
        Ajuste para o balanço ideal!""")
    st.divider()
    if modelo_preditivo:
        st.subheader("Informações do Modelo")
        st.caption(f"Modelo em uso: `modelo_LGBM`") 
        AUC_CONHECIDO = 0.90 
        st.metric(label="AUC do Modelo (Treino)", value=f"{AUC_CONHECIDO:.2f}", help="AUC > 0.8 é geralmente bom/excelente.")
    else:
        st.error("⚠️ Modelo de predição não carregado!")


# --- Função de Pré-processamento ---
def preprocessar_dataframe(df_input_original, is_online_input=False):
    df_proc = df_input_original.copy()
    data_referencia_treino = pd.to_datetime(DATA_REFERENCIA_TREINO_STR)
    current_year_reference = data_referencia_treino.year

    if 'Income' in df_proc.columns:
        df_proc['Income'] = pd.to_numeric(df_proc['Income'], errors='coerce') 
        if df_proc['Income'].isnull().any(): df_proc['Income'] = df_proc['Income'].fillna(MEDIANA_INCOME_TREINO)
    elif not is_online_input: df_proc['Income'] = MEDIANA_INCOME_TREINO
    else: df_proc['Income'] = MEDIANA_INCOME_TREINO

    if 'Dt_Customer' in df_proc.columns:
        if is_online_input and isinstance(df_proc['Dt_Customer'].iloc[0], date) and not isinstance(df_proc['Dt_Customer'].iloc[0], datetime) :
             df_proc['Dt_Customer_datetime'] = pd.to_datetime(df_proc['Dt_Customer'])
        else:
             df_proc['Dt_Customer_datetime'] = pd.to_datetime(df_proc['Dt_Customer'], errors='coerce')
        valid_dates_mask = df_proc['Dt_Customer_datetime'].notna()
        df_proc['Dt_Customer_Days'] = np.nan 
        if valid_dates_mask.any():
            df_proc.loc[valid_dates_mask, 'Dt_Customer_Days'] = (data_referencia_treino - df_proc.loc[valid_dates_mask, 'Dt_Customer_datetime']).dt.days
    else: df_proc['Dt_Customer_Days'] = np.nan

    if 'Year_Birth' in df_proc.columns:
        df_proc['Year_Birth'] = pd.to_numeric(df_proc['Year_Birth'], errors='coerce')
        df_proc['Age'] = current_year_reference - df_proc['Year_Birth']
    else: df_proc['Age'] = np.nan

    mnt_cols = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
    for col in mnt_cols: 
        if col in df_proc.columns: df_proc[col] = pd.to_numeric(df_proc[col], errors='coerce').fillna(0)
        elif is_online_input: df_proc[col] = 0
        else: df_proc[col] = np.nan 
            
    if all(col in df_proc.columns for col in mnt_cols): df_proc['Spent'] = df_proc[mnt_cols].sum(axis=1)
    else: df_proc['Spent'] = np.nan

    if 'Marital_Status' in df_proc.columns:
        df_proc['Living_With'] = df_proc['Marital_Status'].replace({'Married': 'Partner', 'Together': 'Partner', 'Absurd': 'Alone','Widow': 'Alone', 'YOLO': 'Alone', 'Divorced': 'Alone', 'Single': 'Alone'})
    else: df_proc['Living_With'] = "Unknown"

    base_child_cols_present = True
    for col_child in ['Kidhome', 'Teenhome']:
        if col_child not in df_proc.columns:
            if is_online_input: df_proc[col_child] = 0
            else: base_child_cols_present = False; df_proc[col_child] = np.nan
        else:
            df_proc[col_child] = pd.to_numeric(df_proc[col_child], errors='coerce').fillna(0)
    
    if base_child_cols_present or is_online_input:
        df_proc['Children'] = df_proc['Kidhome'] + df_proc['Teenhome']
    else: df_proc['Children'] = np.nan

    if 'Living_With' in df_proc.columns and 'Children' in df_proc.columns and df_proc['Children'].notna().any():
        living_with_map = df_proc['Living_With'].map({'Alone': 1, 'Partner': 2}).fillna(1)
        df_proc['Family_Size'] = living_with_map + df_proc['Children']
    else: df_proc['Family_Size'] = np.nan

    if 'Children' in df_proc.columns and df_proc['Children'].notna().any():
        df_proc['Is_Parent'] = np.where(df_proc['Children'] > 0, 1, 0)
    else: df_proc['Is_Parent'] = np.nan
    
    cols_to_make_category = ['Education', 'Marital_Status', 'Living_With', 'Is_Parent']
    for col in cols_to_make_category:
        if col in df_proc.columns:
            try:
                if df_proc[col].notna().any(): 
                    if pd.api.types.is_categorical_dtype(df_proc[col]):
                        if df_proc[col].isnull().any() and np.nan not in df_proc[col].cat.categories:
                            df_proc[col] = df_proc[col].cat.add_categories(np.nan)
                    df_proc[col] = df_proc[col].astype('category')
            except: df_proc[col] = df_proc[col].astype('object')

    expected_features_for_model = ['Education', 'Marital_Status', 'Income', 'Kidhome', 'Teenhome', 'Recency','MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds','NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth','AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1', 'AcceptedCmp2', 'Complain','Dt_Customer_Days', 'Age', 'Spent', 'Living_With', 'Children', 'Family_Size', 'Is_Parent']
    df_final_para_modelo = pd.DataFrame() 
    for col in expected_features_for_model:
        if col in df_proc.columns:
            df_final_para_modelo[col] = df_proc[col]
        else: 
            df_final_para_modelo[col] = np.nan 
            if col in ['Income', 'Age', 'Spent', 'Dt_Customer_Days', 'Recency']: df_final_para_modelo[col] = df_final_para_modelo[col].astype(float)
            elif col in ['Kidhome', 'Teenhome', 'Children', 'Family_Size', 'Is_Parent', 'Complain'] or 'AcceptedCmp' in col or 'Num' in col : df_final_para_modelo[col] = pd.to_numeric(df_final_para_modelo[col], errors='coerce').astype('Int64')
            else: df_final_para_modelo[col] = df_final_para_modelo[col].astype('object')
    return df_final_para_modelo

# --- Funções Auxiliares ---
@st.cache_data
def convert_df_to_csv(df_to_convert):
    return df_to_convert.to_csv(index=False).encode('utf-8')

# --- Abas ---
st.markdown("Utilize este aplicativo para prever a propensão de compra de clientes para uma nova campanha.")
tab_pred_csv, tab_pred_online, tab_analytics = st.tabs(["📤 Predição via CSV", "👤 Predição Online Individual", "📊 Analytics Simplificado"])

with tab_pred_csv:
    st.header("Predição em Lote via Arquivo CSV")
    uploaded_file = st.file_uploader("Escolha um arquivo CSV (.csv)", type="csv", key="csv_uploader_tab1_final_v7") 

    if uploaded_file is not None and modelo_preditivo is not None:
        progress_bar = st.progress(0, text="Iniciando...")
        try:
            sample_bytes = uploaded_file.read(2048); uploaded_file.seek(0)
            sample_str = sample_bytes.decode(errors='ignore').lower()
            sep = ','
            if sample_str.count(';') > sample_str.count(',') and ';' in sample_str.splitlines()[0]: sep = ';'
            
            progress_bar.progress(10, text=f"Lendo CSV com separador: '{sep}'...")
            df_original_para_prever = pd.read_csv(uploaded_file, sep=sep)
            st.session_state.uploaded_df_original = df_original_para_prever.copy()
            
            st.subheader("Amostra dos Dados Carregados:")
            st.dataframe(st.session_state.uploaded_df_original.head())
            progress_bar.progress(30, text="Pré-processando dados...")
            
            df_para_modelo = preprocessar_dataframe(st.session_state.uploaded_df_original.copy(), is_online_input=False)
            progress_bar.progress(60, text="Fazendo predições...")

            with st.spinner("Modelo está prevendo..."):
                predicoes_pycaret_output = predict_model(modelo_preditivo, data=df_para_modelo, raw_score=True)
            
            progress_bar.progress(90, text="Previsões concluídas. Aplicando threshold...")
            
            df_final_com_predicoes = st.session_state.uploaded_df_original.copy()
            score_col_name = None
            possible_score_cols = ['1', 1, 'prediction_score_1', 'Score_1'] 
            for col_candidate in possible_score_cols:
                if col_candidate in predicoes_pycaret_output.columns:
                    score_col_name = col_candidate; break
            
            if score_col_name is not None:
                if df_final_com_predicoes.shape[0] == predicoes_pycaret_output.shape[0]:
                    df_final_com_predicoes['score_classe1'] = predicoes_pycaret_output[score_col_name].values
                else: 
                    score_df_to_join = predicoes_pycaret_output[[score_col_name]].rename(columns={score_col_name: 'score_classe1'})
                    df_final_com_predicoes = df_final_com_predicoes.join(score_df_to_join)

                if 'score_classe1' in df_final_com_predicoes.columns:
                    df_final_com_predicoes['prediction_label'] = (df_final_com_predicoes['score_classe1'].fillna(0.0) >= st.session_state.threshold).astype(int)
                    st.session_state.predictions_df_full = df_final_com_predicoes.copy()
                else:
                    st.error("Falha ao juntar 'score_classe1' ao DataFrame original."); st.session_state.predictions_df_full = None
            else:
                st.error("Coluna de score P(1) não encontrada na saída do modelo com raw_score=True."); st.write("Colunas disponíveis:", predicoes_pycaret_output.columns.tolist()); st.dataframe(predicoes_pycaret_output.head()); st.session_state.predictions_df_full = None; raise ValueError("Coluna de score P(1) não encontrada.")

            if st.session_state.predictions_df_full is not None:
                st.subheader(f"Resultados da Predição (limiar {st.session_state.threshold:.2f}):")
                cols_para_mostrar = []
                if 'score_classe1' in st.session_state.predictions_df_full.columns: cols_para_mostrar.append('score_classe1')
                if 'prediction_label' in st.session_state.predictions_df_full.columns: cols_para_mostrar.append('prediction_label')
                cols_para_mostrar.extend([col for col in st.session_state.uploaded_df_original.columns if col in st.session_state.predictions_df_full.columns and col not in cols_para_mostrar])
                outras_cols_pycaret = [col for col in st.session_state.predictions_df_full.columns if col not in cols_para_mostrar]
                cols_para_mostrar.extend(outras_cols_pycaret)
                cols_para_mostrar = [col for col in cols_para_mostrar if col in st.session_state.predictions_df_full.columns]
                st.dataframe(st.session_state.predictions_df_full[cols_para_mostrar])

                if 'prediction_label' in st.session_state.predictions_df_full.columns:
                    contagem = st.session_state.predictions_df_full['prediction_label'].value_counts()
                    st.metric("Clientes Previstos como Propensos (1)", contagem.get(1, 0))
                    st.metric("Clientes Previstos como Não Propensos (0)", contagem.get(0, 0))
                
                csv_dl = convert_df_to_csv(st.session_state.predictions_df_full[cols_para_mostrar])
                st.download_button("Baixar Resultados como CSV", csv_dl, f"previsoes_limiar_{str(st.session_state.threshold).replace('.', '')}.csv", "text/csv", key="download_csv_btn_final_v7")
            
            progress_bar.progress(100, text="Concluído!")
            progress_bar.empty()
        except Exception as e:
            st.error(f"Ocorreu um erro no processamento do CSV: {e}"); st.exception(e)
            if 'progress_bar' in locals() and progress_bar is not None : progress_bar.empty()
            st.session_state.uploaded_df_original = None; st.session_state.predictions_df_full = None
        
    elif modelo_preditivo is None: 
             st.error("⚠️ Modelo de predição não carregado. Não é possível fazer predições.")
    elif modelo_preditivo is None: 
        st.warning("⚠️ Modelo de predição não carregado. Funcionalidade de predição indisponível.")


with tab_pred_online:
    st.header("👤 Simulador de Cliente - Predição Online Individual")
    st.markdown("Insira os dados de um novo cliente para obter uma predição.")
    education_opts = ['Graduation', 'PhD', 'Master', '2n Cycle', 'Basic', 'Outro'] 
    marital_opts = ['Single', 'Together', 'Married', 'Divorced', 'Widow', 'Alone', 'Absurd', 'YOLO', 'Outro']

    with st.form("online_input_form_final_v7"): 
        st.markdown("##### Informações Pessoais e de Cadastro")
        col_form_info1, col_form_info2 = st.columns(2, gap="medium")
        with col_form_info1:
            year_birth = st.number_input("Ano de Nascimento", 1900, date.today().year - 18, 1980, key="onl_yb_final_v7")
            education = st.selectbox("Nível de Educação", education_opts, index=0, key="onl_edu_final_v7")
            income = st.number_input("Renda Anual (€)", 0.0, 1000000.0, 50000.0, step=1000.0, format="%.2f", key="onl_inc_final_v7")
            kidhome = st.number_input("Nº Crianças em Casa", 0, 5, 0, 1, key="onl_kh_final_v7") 
        with col_form_info2:
            marital_status = st.selectbox("Estado Civil", marital_opts, index=0, key="onl_ms_final_v7")
            dt_customer_input = st.date_input("Data de Cadastro do Cliente", date(2013, 7, 2), date(2000,1,1), date.today(), key="onl_dtc_final_v7")
            teenhome = st.number_input("Nº Adolescentes em Casa", 0, 5, 0, 1, key="onl_th_final_v7")
            recency = st.number_input("Dias Desde Última Compra", 0, 3650, 30, 1, key="onl_rec_final_v7")
        
        complain = st.radio("Reclamou nos Últimos 2 Anos?", [0, 1], index=0, format_func=lambda x: "Não" if x == 0 else "Sim", key="onl_comp_final_v7", horizontal=True)
        st.markdown("<br>", unsafe_allow_html=True) 

        st.markdown("##### Gastos em Produtos (Últimos 2 Anos)")
        col_f_mnt1,col_f_mnt2,col_f_mnt3 = st.columns(3); col_f_mnt4,col_f_mnt5,col_f_mnt6 = st.columns(3)
        with col_f_mnt1: mntwines = st.number_input("Gasto Vinhos (€)", 0, 5000, 50, key="onl_mw_final_v7")
        with col_f_mnt2: mntfruits = st.number_input("Gasto Frutas (€)", 0, 1000, 10, key="onl_mf_final_v7")
        with col_f_mnt3: mntmeat = st.number_input("Gasto Carnes (€)", 0, 5000, 30, key="onl_mm_final_v7")
        with col_f_mnt4: mntfish = st.number_input("Gasto Peixes (€)", 0, 1000, 10, key="onl_mfi_final_v7")
        with col_f_mnt5: mntsweet = st.number_input("Gasto Doces (€)", 0, 1000, 5, key="onl_msw_final_v7")
        with col_f_mnt6: mntgold = st.number_input("Gasto Ouro (€)", 0, 1000, 10, key="onl_mg_final_v7")
        
        st.markdown("##### Comportamento de Compra")
        col_f_pur1,col_f_pur2,col_f_pur3 = st.columns(3); col_f_pur4,col_f_pur5, _ = st.columns([1,1,1])
        with col_f_pur1: numdealspurchases = st.number_input("Compras com Desconto", 0, 50, 1, key="onl_ndp_final_v7")
        with col_f_pur2: numwebpurchases = st.number_input("Compras pela Web", 0, 50, 2, key="onl_nwp_final_v7")
        with col_f_pur3: numcatalogpurchases = st.number_input("Compras por Catálogo", 0, 50, 1, key="onl_ncp_final_v7")
        with col_f_pur4: numstorepurchases = st.number_input("Compras na Loja", 0, 50, 3, key="onl_nsp_final_v7")
        with col_f_pur5: numwebvisitsmonth = st.number_input("Visitas ao Website/Mês", 0, 100, 5, key="onl_nwv_final_v7")
        
        st.markdown("##### Aceitação de Campanhas Anteriores")
        c_cmp1,c_cmp2,c_cmp3,c_cmp4,c_cmp5 = st.columns(5)
        with c_cmp1: acceptedcmp1 = st.checkbox("Cmp1", key="onl_ac1_final_v7")
        with c_cmp2: acceptedcmp2 = st.checkbox("Cmp2", key="onl_ac2_final_v7")
        with c_cmp3: acceptedcmp3 = st.checkbox("Cmp3", key="onl_ac3_final_v7")
        with c_cmp4: acceptedcmp4 = st.checkbox("Cmp4", key="onl_ac4_final_v7")
        with c_cmp5: acceptedcmp5 = st.checkbox("Cmp5", key="onl_ac5_final_v7")
        submitted_online = st.form_submit_button("🎯 Realizar Predição deste Cliente")

    if submitted_online and modelo_preditivo is not None:
        online_data_dict = {'Year_Birth': year_birth, 'Education': education, 'Marital_Status': marital_status,'Income': income, 'Kidhome': kidhome, 'Teenhome': teenhome, 'Dt_Customer': dt_customer_input,'Recency': recency, 'MntWines': mntwines, 'MntFruits': mntfruits,'MntMeatProducts': mntmeat, 'MntFishProducts': mntfish, 'MntSweetProducts': mntsweet,'MntGoldProds': mntgold, 'NumDealsPurchases': numdealspurchases,'NumWebPurchases': numwebpurchases, 'NumCatalogPurchases': numcatalogpurchases,'NumStorePurchases': numstorepurchases, 'NumWebVisitsMonth': numwebvisitsmonth,'AcceptedCmp1': int(acceptedcmp1), 'AcceptedCmp2': int(acceptedcmp2),'AcceptedCmp3': int(acceptedcmp3), 'AcceptedCmp4': int(acceptedcmp4),'AcceptedCmp5': int(acceptedcmp5), 'Complain': complain}
        df_online_input = pd.DataFrame([online_data_dict])
        with st.spinner("Processando e prevendo..."):
            df_online_processed = preprocessar_dataframe(df_online_input.copy(), is_online_input=True)
            try:
                online_pred_output = predict_model(modelo_preditivo, data=df_online_processed, raw_score=True)
                score_col_name_online = None
                possible_score_cols_onl = ['1', 1, 'prediction_score_1', 'Score_1']
                for col_candidate_onl in possible_score_cols_onl:
                    if col_candidate_onl in online_pred_output.columns:
                        score_col_name_online = col_candidate_onl; break
                if score_col_name_online:
                    st.session_state.online_score_classe1 = online_pred_output[score_col_name_online].iloc[0]
                    st.session_state.online_prediction_label = (st.session_state.online_score_classe1 >= st.session_state.threshold).astype(int)
                    st.session_state.show_online_prediction = True

                    history_entry = online_data_dict.copy()
                    features_derivadas_online = ['Dt_Customer_Days', 'Age', 'Spent', 'Living_With', 'Children', 'Family_Size', 'Is_Parent']
                    for feature_derivada in features_derivadas_online:
                        if feature_derivada in df_online_processed.columns:
                            history_entry[feature_derivada] = df_online_processed[feature_derivada].iloc[0]
                        else: history_entry[feature_derivada] = np.nan 
                    history_entry['score_classe1'] = st.session_state.online_score_classe1
                    history_entry['prediction_label'] = st.session_state.online_prediction_label
                    history_entry['Timestamp_Predicao'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    st.session_state.online_history.append(history_entry)
                else:
                    st.error("Coluna de score P(1) não encontrada na predição online."); st.dataframe(online_pred_output); raise ValueError("P(1) não encontrada online.")
            except Exception as e: st.error(f"Erro na predição online: {e}"); st.exception(e); st.session_state.show_online_prediction = False
    
    if st.session_state.show_online_prediction: 
        score_classe1_disp = st.session_state.online_score_classe1
        label_disp = st.session_state.online_prediction_label
        st.markdown("---"); st.subheader("Resultado da Predição Online:")
        if label_disp == 1:
            st.markdown(f"""<div style="background-color: #D4EDDA; color: #155724; border: 1px solid #C3E6CB; padding: 15px; border-radius: 5px; text-align: center;">
                            <h3 style="color: #155724;">✅ Cliente PROPENSO a Comprar!</h3>
                            <p style="font-size: 17px;">Probabilidade de Compra (Score da Classe 1): <strong>{score_classe1_disp:.4f}</strong><br>
                            <em>(Limiar utilizado: {st.session_state.threshold:.2f})</em></p></div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div style="background-color: #F8D7DA; color: #721C24; border: 1px solid #F5C6CB; padding: 15px; border-radius: 5px; text-align: center;">
                            <h3 style="color: #721C24;">❌ Cliente NÃO Propenso a Comprar.</h3>
                            <p style="font-size: 17px;">Probabilidade de Compra (Score da Classe 1): <strong>{score_classe1_disp:.4f}</strong><br>
                            <em>(Limiar utilizado: {st.session_state.threshold:.2f})</em></p></div>""", unsafe_allow_html=True)
        st.markdown("---")

    if st.session_state.online_history:
        st.markdown("---")
        st.subheader("📜 Histórico de Predições Online Individuais")
        df_history = pd.DataFrame(st.session_state.online_history)
        
        cols_input_base_ord_hist = ['Timestamp_Predicao','Year_Birth', 'Education', 'Marital_Status', 'Income', 'Kidhome', 'Teenhome', 'Dt_Customer','Recency', 'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth', 'AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'Complain']
        cols_derivadas_ord_hist = ['Dt_Customer_Days', 'Age', 'Spent', 'Living_With', 'Children', 'Family_Size', 'Is_Parent']
        cols_predicao_ord_hist = ['score_classe1', 'prediction_label']
        
        colunas_finais_hist_ordenadas = []
        for col_list_hist in [cols_predicao_ord_hist, cols_input_base_ord_hist, cols_derivadas_ord_hist]:
            for col_hist in col_list_hist:
                if col_hist in df_history.columns and col_hist not in colunas_finais_hist_ordenadas: 
                    colunas_finais_hist_ordenadas.append(col_hist)
        colunas_restantes_hist = [col for col in df_history.columns if col not in colunas_finais_hist_ordenadas]
        colunas_finais_hist_ordenadas.extend(colunas_restantes_hist)
        colunas_finais_hist_ordenadas = [col for col in colunas_finais_hist_ordenadas if col in df_history.columns]

        st.dataframe(df_history[colunas_finais_hist_ordenadas if colunas_finais_hist_ordenadas else df_history].reset_index(drop=True))
        
        st.markdown("##### Gerenciar Histórico")
        if len(st.session_state.online_history) > 0:
            col_download_hist_online_v3, col_clear_hist_online_v3 = st.columns([1,1], gap="small") 
            
            with col_download_hist_online_v3:
                 csv_history_dl = convert_df_to_csv(df_history[colunas_finais_hist_ordenadas if colunas_finais_hist_ordenadas else df_history])
                 st.download_button(label="📥 Baixar Histórico",data=csv_history_dl,
                                   file_name="historico_predicoes_online.csv",mime="text/csv",
                                   key="download_online_hist_btn_final_v3", use_container_width=True)
            with col_clear_hist_online_v3:
                if st.button("🧹 Limpar Histórico", key="btn_clear_all_online_hist_final_v6", type="secondary", use_container_width=True): # NOME DO BOTÃO ALTERADO
                    if st.session_state.online_history:
                        st.session_state.online_history = []
                        st.session_state.show_online_prediction = False 
                        st.session_state.online_score_classe1 = None
                        st.session_state.online_prediction_label = None
                        st.success("Histórico de predições online foi limpo.")
                        st.rerun() 
                    else:
                        st.info("Histórico já está vazio.")
        st.markdown("---")


with tab_analytics: # Aba Analytics (mantida como na última versão funcional)
    st.header("📊 Analytics Simplificado das Predições do CSV")
    if st.session_state.predictions_df_full is None:
        st.info("👈 Por favor, carregue um arquivo CSV e realize as predições na aba 'Predição via CSV' para popular esta análise.")
    else:
        df_analise = st.session_state.predictions_df_full.copy()
        st.markdown(f"Análise baseada no **threshold atual de {st.session_state.threshold:.2f}**.")
        
        label_map_display = {0: 'Não Propenso (0)', 1: 'Propenso (1)'}
        if 'prediction_label' in df_analise.columns:
            df_analise.dropna(subset=['prediction_label'], inplace=True) 
            df_analise['Predição (Status)'] = df_analise['prediction_label'].astype(int).map(label_map_display).astype('category')
        else: st.warning("Coluna 'prediction_label' não encontrada para análise."); st.stop()

        color_map_display = {label_map_display[0]: 'royalblue', label_map_display[1]: 'firebrick'}
        category_order_display = [label_map_display[0], label_map_display[1]]

        st.subheader("Análise de Features Numéricas") 
        features_numericas_default = ['Income', 'Age', 'Spent', 'Recency', 'Dt_Customer_Days']
        features_numericas_disponiveis = [col for col in features_numericas_default if col in df_analise.columns and pd.api.types.is_numeric_dtype(df_analise[col])]
        
        if not features_numericas_disponiveis: st.warning("Nenhuma feature numérica padrão para análise encontrada.")
        else:
            feature_num_selecionada = st.selectbox("Escolha uma Feature Numérica:", options=features_numericas_disponiveis, key="sel_num_an_final_v6")
            if feature_num_selecionada and feature_num_selecionada in df_analise.columns:
                df_para_plotar_num = df_analise.copy()
                nota_filtro_num_especifica = ""
                eixo_y_boxplot_label_num = feature_num_selecionada 
                eixo_x_hist_label_num = feature_num_selecionada 

                if feature_num_selecionada == 'Income': 
                    if df_analise[feature_num_selecionada].notna().sum() > 20: 
                        limite_inferior_viz = df_analise[feature_num_selecionada].quantile(0.01)
                        limite_superior_viz = df_analise[feature_num_selecionada].quantile(0.99)
                        if pd.notna(limite_inferior_viz) and pd.notna(limite_superior_viz) and limite_inferior_viz < limite_superior_viz:
                            df_para_plotar_num = df_analise[(df_analise[feature_num_selecionada] >= limite_inferior_viz) & (df_analise[feature_num_selecionada] <= limite_superior_viz)].copy() 
                            if df_para_plotar_num.shape[0] < df_analise[feature_num_selecionada].notna().sum():
                                nota_filtro_num_especifica = f"Nota: Para '{feature_num_selecionada}', os gráficos (Boxplot e Histograma) focam na faixa entre o 1º e 99º percentil (aprox. {limite_inferior_viz:.0f} - {limite_superior_viz:.0f}) para melhor clareza da distribuição principal."
                                eixo_y_boxplot_label_num = f"{feature_num_selecionada} (1º-99ºp)"
                                eixo_x_hist_label_num = f"{feature_num_selecionada} (1º-99ºp)"
                
                col_box_an, col_hist_an = st.columns(2)
                with col_box_an:
                    try:
                        fig_box = px.box(df_para_plotar_num, x='Predição (Status)', y=feature_num_selecionada, color='Predição (Status)',
                                         title=f'Boxplot de "{feature_num_selecionada}"', color_discrete_map=color_map_display,
                                         category_orders={'Predição (Status)': category_order_display})
                        fig_box.update_layout(xaxis_title="Status da Predição", yaxis_title=eixo_y_boxplot_label_num, legend_title_text="Status")
                        st.plotly_chart(fig_box, use_container_width=True)
                    except Exception as e: st.warning(f"Erro Boxplot '{feature_num_selecionada}': {e}")
                with col_hist_an:
                    try: 
                        fig_hist = px.histogram(df_para_plotar_num, x=feature_num_selecionada, color='Predição (Status)', 
                                                  barmode='overlay', histnorm='probability density', 
                                                  title=f'Distribuição de "{feature_num_selecionada}"', color_discrete_map=color_map_display, opacity=0.65,
                                                  category_orders={'Predição (Status)': category_order_display})
                        fig_hist.update_layout(xaxis_title=eixo_x_hist_label_num, yaxis_title="Densidade de Probabilidade", legend_title_text="Status")
                        st.plotly_chart(fig_hist, use_container_width=True)
                    except Exception as e: st.warning(f"Erro Histograma '{feature_num_selecionada}': {e}")
                if nota_filtro_num_especifica: st.caption(nota_filtro_num_especifica)
        
        st.markdown("---") 
        st.subheader("Análise de Features Categóricas")
        features_categoricas_default = ['Education', 'Marital_Status', 'Living_With', 'Is_Parent']
        features_categoricas_disponiveis = [col for col in features_categoricas_default if col in df_analise.columns]
        if not features_categoricas_disponiveis: st.warning("Nenhuma feature categórica padrão para análise encontrada.")
        else:
            feature_cat_selecionada = st.selectbox("Escolha uma Feature Categórica:", options=features_categoricas_disponiveis, key="sel_cat_an_final_v5")
            if feature_cat_selecionada and feature_cat_selecionada in df_analise.columns:
                if df_analise[feature_cat_selecionada].notna().any():
                    col_cat_bar1_an, col_cat_bar2_an = st.columns(2)
                    with col_cat_bar1_an:
                        try: 
                            counts_df_group = df_analise.groupby('Predição (Status)', observed=True)[feature_cat_selecionada].value_counts(normalize=True).mul(100).rename('percentage').reset_index()
                            fig_cat_bar = px.bar(counts_df_group, x=feature_cat_selecionada, y='percentage', color='Predição (Status)', 
                                                 barmode='group', title=f'Distribuição de "{feature_cat_selecionada}" por Predição',
                                                 color_discrete_map=color_map_display, text_auto='.1f', 
                                                 category_orders={'Predição (Status)': category_order_display})
                            fig_cat_bar.update_layout(xaxis_title=feature_cat_selecionada, yaxis_title="% no Grupo de Predição", legend_title_text="Status")
                            st.plotly_chart(fig_cat_bar, use_container_width=True)
                        except Exception as e: st.warning(f"Erro Barras Agrupadas '{feature_cat_selecionada}': {e}")
                    with col_cat_bar2_an: 
                        if df_analise[feature_cat_selecionada].nunique() < 15 : 
                            try:
                                cross_tab_norm = pd.crosstab(df_analise[feature_cat_selecionada], df_analise['Predição (Status)'], normalize='index').mul(100)
                                cross_tab_melted = cross_tab_norm.reset_index().melt(id_vars=feature_cat_selecionada, var_name='Predição (Status)', value_name='percentage')
                                fig_cat_stack = px.bar(cross_tab_melted, x=feature_cat_selecionada, y='percentage', color='Predição (Status)',
                                                        title=f'Composição de Predições por "{feature_cat_selecionada}"',
                                                        color_discrete_map=color_map_display, text_auto='.1f',
                                                        category_orders={'Predição (Status)': category_order_display})
                                fig_cat_stack.update_layout(xaxis_title=feature_cat_selecionada, yaxis_title="% na Categoria da Feature", legend_title_text="Status")
                                st.plotly_chart(fig_cat_stack, use_container_width=True)
                            except Exception as e: st.warning(f"Erro Barras Empilhadas '{feature_cat_selecionada}': {e}")

if modelo_preditivo is None:
    st.sidebar.error("Modelo de predição não carregado.")
    st.error("ERRO FATAL: Modelo de ML não carregado.")