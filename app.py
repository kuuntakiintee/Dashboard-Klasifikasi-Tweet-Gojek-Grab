import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pickle
import numpy as np
import plotly.graph_objects as go
import streamlit_antd_components as sac 

st.set_page_config(
    page_title="Dashboard Analisis Data",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
    <style>
    .metric-card {
        background-color: #0e1117;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        color: white;
    }
    .metric-card h3 {
        font-size: 1.2rem;
        color: #a0aec0;
        margin-bottom: 8px;
    }
    .metric-card p {
        font-size: 2rem;
        font-weight: bold;
        margin: 0;
        color: white;
    }
    .prediction-box {
        background-color: #0e1117;
        padding: 20px;
        border-radius: 12px;
        color: white;
        margin-top: 1rem;
    }
    
    .main .block-container {
        padding-bottom: 5rem; 
    }

    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #121212; 
        color: #64748b;
        text-align: center;
        padding: 10px;
        z-index: 100; 
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data(uploaded_file):
    df = None 
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        if df is not None and 'intention' in df.columns and 'type' not in df.columns:
            df.rename(columns={'intention': 'type'}, inplace=True)

        return df
    except Exception as e:
        st.error(f"Error: Gagal membaca file: {e}")
        return None

@st.cache_resource 
def load_models():
    try:
        models = {
            "tfidf_topic": pickle.load(open("model/topic/tfidf_vectorizer7030_hyper_topic.pkl","rb")),
            "model_topic": pickle.load(open("model/topic/stacking_model_7030_hyper_topic.pkl", "rb")),
            "tfidf_intention": pickle.load(open("model/intention/tfidf_vectorizer8020_hyper_intention.pkl", "rb")),
            "model_intention": pickle.load(open("model/intention/stacking_model_8020_hyper_intention.pkl", "rb"))
        }
        return models
    except FileNotFoundError:
        st.error("Error Kritis: File model (.pkl) tidak ditemukan.")
        st.error("Pastikan file model berada di direktori yang sama dengan aplikasi.")
        return None
    except Exception as e:
        st.error(f"Error Kritis: Gagal memuat model: {e}")
        return None
    
models = load_models()

def reset_secondary_filter():
    if 'secondary_dimension_filter' in st.session_state:
        st.session_state.secondary_dimension_filter = "Semua"

@st.cache_data
def generate_wordcloud_from_text(text_data):
    if not text_data or len(text_data.strip()) == 0:
        return None
    try:
        wordcloud = WordCloud(
            width=1600, height=600, background_color="white",
            colormap="viridis", random_state=42
        ).generate(text_data)
        return wordcloud
    except Exception as e:
        print(f"Error saat membuat Word Cloud: {e}")
        return None

# SIDEBAR
with st.sidebar:
    st.markdown("## DASHBOARD KLASIFIKASI TWEET")
    st.divider()
    st.markdown("## Unggah Dataset") 
    uploaded_file = st.file_uploader(
        "Pilih File Data (.csv/.xlsx)", 
        type=["csv", "xlsx", "xls"],
        help="Unggah file CSV atau Excel yang berisi data ulasan." 
    )
    st.divider()

    menu = sac.menu([
        sac.MenuItem('Dataset', icon='file-earmark-text'),
        sac.MenuItem('Dashboard', icon='bar-chart-line'),
        sac.MenuItem('Prediction', icon='robot'),
        sac.MenuItem('Evaluation', icon='graph-up'),
        sac.MenuItem('About', icon='info-circle')
    ], format_func='title', open_all=False, index=0)

    st.divider()
    
    st.markdown("### Status Sistem")
    if models is not None:
        st.success("Model Siap.", icon="🤖")
    else:
        st.warning("Model Gagal Dimuat.", icon="🚨")

df = None
if uploaded_file is not None:
    df = load_data(uploaded_file) 

if menu == "Dataset":
    st.title("📄Dataset Viewer")

    if df is not None:
        st.success(f"Informasi: File berhasil dimuat: **{uploaded_file.name}**")
        
        df_filtered = df.copy()

        with st.expander("Tampilkan Filter Data", expanded=True):
            filt_col1, filt_col2, filt_col3 = st.columns(3)
            
            with filt_col1:
                if 'topic' in df.columns:
                    conditions = [
                        df_filtered['topic'].isin(['gojek', 'gosend', 'gopay', 'gofood', 'gocar']),
                        df_filtered['topic'].isin(['grab', 'grabcar', 'grabexpress', 'ovo', 'grabfood'])
                    ]
                    choices = ['Gojek', 'Grab']
                    df_filtered['brand_filter'] = np.select(conditions, choices, default='Lainnya')
                    
                    available_brands = ["Semua"] + sorted([b for b in df_filtered['brand_filter'].unique() if b != 'Lainnya'])
                    
                    selected_brand = st.selectbox(
                        "Filter Merek:", 
                        available_brands, 
                        key='dataset_brand_filter'
                    )
                    
                    if selected_brand != "Semua":
                        df_filtered = df_filtered[df_filtered['brand_filter'] == selected_brand]
                    
                    df_filtered = df_filtered.drop(columns=['brand_filter'])
                else:
                    st.text_input("Filter Merek:", disabled=True, value="Kolom 'topic' tidak ada")

            with filt_col2:
                if 'topic' in df.columns:
                    available_topics = ["Semua"] + sorted(df_filtered['topic'].dropna().unique())
                    selected_topic = st.selectbox(
                        "Filter Topic:", 
                        available_topics, 
                        key='dataset_topic_filter'
                    )
                    
                    if selected_topic != "Semua":
                        df_filtered = df_filtered[df_filtered['topic'] == selected_topic]
                else:
                    st.text_input("Filter Topic:", disabled=True, value="Kolom 'topic' tidak ada")

            with filt_col3:
                if 'type' in df.columns:
                    available_types = ["Semua"] + sorted(df_filtered['type'].dropna().unique())
                    selected_type = st.selectbox(
                        "Filter Intention (Type):", 
                        available_types, 
                        key='dataset_type_filter'
                    )
                    
                    if selected_type != "Semua":
                        df_filtered = df_filtered[df_filtered['type'] == selected_type]
                else:
                    st.text_input("Filter Intention (Type):", disabled=True, value="Kolom 'type' tidak ada")
        
        st.caption(f"Menampilkan **{len(df_filtered):,}** baris dari total **{len(df):,}** baris data (setelah filter).")
        st.divider()

        st.markdown("### Statistik Ringkas Dataset (Hasil Filter)")
        col1, col2, col3, col4 = st.columns(4) 

        total_baris = len(df_filtered)
        total_topic = df_filtered['topic'].nunique() if 'topic' in df_filtered.columns else 0
        total_type = df_filtered['type'].nunique() if 'type' in df_filtered.columns else 0
        total_null = df_filtered.isnull().sum().sum() 

        with col1:
            st.markdown(f"<div class='metric-card'><h3>Total Baris</h3><p>{total_baris:,}</p></div>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<div class='metric-card'><h3>Jumlah Topic Unik</h3><p>{total_topic}</p></div>", unsafe_allow_html=True)
        with col3:
            st.markdown(f"<div class='metric-card'><h3>Jumlah Intention Unik</h3><p>{total_type}</p></div>", unsafe_allow_html=True)
        with col4:
            st.markdown(f"<div class='metric-card'><h3>Total Nilai Kosong</h3><p>{total_null:,}</p></div>", unsafe_allow_html=True)

        st.divider()
        
        col_title, col_toggle = st.columns([0.8, 0.2]) 
        with col_title:
            st.markdown("### Tinjauan Data (Hasil Filter)")
        with col_toggle:
            show_all_data = st.toggle(
                "Tampilkan Semua", 
                help="Centang untuk menampilkan seluruh baris data hasil filter. (Hati-hati jika data berukuran besar)"
            )
        
        if show_all_data:
            st.dataframe(df_filtered, use_container_width=True)
        else:
            st.markdown(f"*(Menampilkan 20 baris pertama dari {len(df_filtered):,} baris hasil filter)*")
            st.dataframe(df_filtered.head(20), use_container_width=True)
        
        with st.expander("Tampilkan Info Dataset Asli (Sebelum Filter)"):
            st.markdown("##### Informasi Kolom")
            
            info_df = pd.DataFrame({
                "Kolom": df.columns,
                "Tipe Data": df.dtypes.astype(str), 
                "Jumlah Non-Null": df.count().values
            })
            info_df["Jumlah Null"] = len(df) - info_df["Jumlah Non-Null"]
            
            info_df = info_df[["Kolom", "Tipe Data", "Jumlah Non-Null", "Jumlah Null"]]
            
            st.dataframe(info_df, use_container_width=True, hide_index=True) 
            
            mem_usage_kb = df.memory_usage(deep=True).sum() / 1024
            mem_usage_mb = mem_usage_kb / 1024
            
            if mem_usage_mb >= 1:
                mem_display = f"{mem_usage_mb:.2f} MB"
            else:
                mem_display = f"{mem_usage_kb:.1f} KB"
                
            st.caption(f"Total Baris Asli: {len(df):,} | Penggunaan Memori: {mem_display}")
    
    else:
        st.info(
            """
            ### Selamat Datang di Prototipe Dashboard Analisis Ulasan

            Aplikasi ini memiliki dua alur fungsi utama:

            1.  **Alur Analisis (Menu Dataset & Dashboard):**
                Digunakan jika Anda telah memiliki data terstruktur (file .csv/.xlsx) yang sudah berisi kolom label `topic` dan `type` (Intention).
                **Unggah file tersebut pada sidebar (di sebelah kiri)** untuk melihat visualisasinya.

            2.  **Alur Prediksi (Menu Prediction):**
                Digunakan jika Anda memiliki data mentah (file .csv/.xlsx) yang hanya berisi kolom `full_text`.
                Buka menu **"Prediction"** di sidebar untuk melakukan klasifikasi batch dan mengunduh hasilnya.
            """
        )
    
elif menu == "Dashboard":
    st.title("📊Dashboard Analisis Interaktif")
    st.markdown("")

    selected_brand = "Semua"
    primary_display_choice = None
    primary_col_select = None
    selected_secondary = "Semua"
    secondary_col_select = None
    secondary_label = ""
    available_primary_options = {}
    df_display = pd.DataFrame() 

    if df is not None:
        df_filtered = df.copy() 
        brand_filter_active = False

        with st.expander("Tampilkan Filter Data", expanded=True):

            if 'topic' in df.columns:
                st.markdown("#### Filter Merek")
                conditions = [
                    df_filtered['topic'].isin(['gojek', 'gosend', 'gopay', 'gofood', 'gocar']),
                    df_filtered['topic'].isin(['grab', 'grabcar', 'grabexpress', 'ovo', 'grabfood'])
                ]
                choices = ['Gojek', 'Grab']
                df_filtered['brand_temp'] = np.select(conditions, choices, default='Lainnya')

                available_brands = ["Semua"] + sorted([b for b in df_filtered['brand_temp'].unique() if b != 'Lainnya'])

                if len(available_brands) > 1:
                    selected_brand = sac.chip(
                        items=[sac.ChipItem(label=b) for b in available_brands],
                        index=0,
                        format_func='title',
                        align='start',
                        radius='md',
                        key='brand_filter_chip',
                        on_change=reset_secondary_filter
                    )
                    if selected_brand != "Semua":
                        df_filtered = df_filtered[df_filtered['brand_temp'] == selected_brand]
                        brand_filter_active = True

                if 'brand_temp' in df_filtered.columns:
                    df_filtered = df_filtered.drop(columns=['brand_temp'])

            else: 
                st.warning("Peringatan: Kolom 'topic' tidak ditemukan, filter Merek tidak dapat diterapkan.")
                selected_brand = "Semua" 

            column_mapping = {"Topic": "topic", "Intention": "type"}
            available_primary_options = {
                display_name: actual_col
                for display_name, actual_col in column_mapping.items()
                if actual_col in df_filtered.columns
            }

            if not available_primary_options:
                st.warning("Peringatan: Kolom `topic` dan/atau `type` tidak ditemukan dalam data yang difilter.")
            else:
                st.markdown("#### Filter Dimensi Analisis")
                filter_col1, filter_col2 = st.columns(2)

                with filter_col1:
                    primary_display_choice = st.selectbox(
                        "Pilih Dimensi Utama:",
                        options=list(available_primary_options.keys()), # Pastikan list
                        key='primary_dimension_filter',
                        help="Dimensi ini akan menentukan apa yang ditampilkan di grafik distribusi/proporsi.",
                        on_change=reset_secondary_filter
                    )
                    primary_col_select = available_primary_options[primary_display_choice]

                secondary_options = ["Semua"] 
                secondary_col_select = None
                secondary_label = ""
                selected_secondary = "Semua" 

                if primary_display_choice == "Intention" and "topic" in df_filtered.columns:
                    secondary_label = "Filter tambahan berdasarkan Topik:"
                    secondary_col_select = "topic"
                    secondary_options.extend(sorted(df_filtered[secondary_col_select].dropna().unique()))

                elif primary_display_choice == "Topic" and "type" in df_filtered.columns:
                    secondary_label = "Filter tambahan berdasarkan Intensi:"
                    secondary_col_select = "type"
                    secondary_options.extend(sorted(df_filtered[secondary_col_select].dropna().unique()))

                with filter_col2:
                    if secondary_col_select and len(secondary_options) > 1:
                        selected_secondary = st.selectbox(
                            secondary_label,
                            options=secondary_options,
                            key='secondary_dimension_filter',
                            help=f"Filter data lebih lanjut berdasarkan {secondary_col_select} spesifik."
                        )

        if primary_col_select and primary_col_select in df_filtered.columns:
             df_display = df_filtered.copy()
             df_display = df_display[df_display[primary_col_select].notna()]

             if secondary_col_select and selected_secondary != "Semua":
                  if secondary_col_select in df_display.columns:
                       df_display = df_display[df_display[secondary_col_select] == selected_secondary]
                  else:
                       st.warning(f"Kolom sekunder '{secondary_col_select}' tidak ditemukan saat filtering.")
                       df_display = pd.DataFrame() 
        else:
             
             df_display = pd.DataFrame() 

        if not df_display.empty:
            st.divider()

            viz_title = f"Hasil Analisis: Distribusi {primary_display_choice}"
            if selected_brand != "Semua": 
                viz_title += f" untuk Merek '{selected_brand}'"
            if selected_secondary != "Semua": 
                secondary_dim_name = "Topik" if secondary_col_select == "topic" else "Intensi" if secondary_col_select == "type" else ""
                if secondary_dim_name:
                    viz_title += f" dengan {secondary_dim_name} '{selected_secondary}'"
            st.markdown(f"#### {viz_title}")

            col_a, col_b = st.columns(2)

            value_counts = df_display[primary_col_select].value_counts().reset_index()
            value_counts.columns = [primary_col_select, "Jumlah"] 
            value_counts = value_counts.sort_values("Jumlah", ascending=False)

            # Grafik Batang
            with col_a:
                fig_bar = px.bar(
                    value_counts,
                    x=primary_col_select, y="Jumlah", color="Jumlah",
                    color_continuous_scale="Blues", title=f"Distribusi {primary_display_choice}"
                )
                fig_bar.update_layout(showlegend=False, height=400, xaxis_tickangle=-45)
                st.plotly_chart(fig_bar, use_container_width=True)

            # Grafik Pie
            with col_b:
                fig_pie = px.pie(
                    value_counts, values="Jumlah", names=primary_col_select,
                    title=f"Proporsi {primary_display_choice}",
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                fig_pie.update_traces(textposition="inside", textinfo="percent+label", sort=False)
                st.plotly_chart(fig_pie, use_container_width=True)

            st.divider()
            st.subheader("Word Cloud (dari kolom 'full_text')")

            # Word Cloud
            if "full_text" in df_display.columns:
                text_data = " ".join(df_display["full_text"].dropna().astype(str))
                if len(text_data.strip()) > 0:
                    try:
                        with st.spinner("Membuat Word Cloud..."):
                            wordcloud_image = generate_wordcloud_from_text(text_data)
                            if wordcloud_image:
                                fig_wc, ax = plt.subplots(figsize=(16, 6))
                                ax.imshow(wordcloud_image, interpolation="bilinear")
                                ax.axis("off")
                                st.pyplot(fig_wc)
                            else:
                                st.error("Error: Gagal membuat Word Cloud.")
                    except Exception as e:
                        st.error(f"Error saat menampilkan Word Cloud: {e}")
                else:
                    st.warning("Peringatan: Tidak ada data teks ('full_text') yang valid setelah menerapkan filter untuk Word Cloud.")
            else:
                st.warning("Peringatan: Kolom 'full_text' tidak ditemukan dalam data setelah filter untuk Word Cloud.")

        elif df is not None and available_primary_options: 
            st.warning("Informasi: Tidak ada data yang cocok dengan kombinasi filter yang dipilih.")

    else: 
        st.info("Informasi: Silakan unggah file data pada sidebar untuk memulai analisis.")

            
elif menu == "Prediction":
    st.title("🔍 Prediksi Topik dan Intensi")

    if models is None:
        st.error("Error: Model tidak dapat dimuat. Fitur prediksi tidak tersedia.")
        st.stop()

    try:
        tfidf_topic = models["tfidf_topic"]
        model_topic = models["model_topic"]
        tfidf_intention = models["tfidf_intention"]
        model_intention = models["model_intention"]
    except KeyError as e:
        st.error(f"Error: Komponen model hilang: {e}. Harap periksa file model Anda.")
        st.stop()
    except Exception as e:
        st.error(f"Error tidak terduga saat membongkar model: {e}")
        st.stop()


    st.subheader("Prediksi Teks Tunggal")
    user_input = st.text_area("Masukkan satu ulasan untuk diuji:", height=150, placeholder="Contoh: Driver gojek saya sangat baik, terimakasih.")

    if st.button("Prediksi Tunggal"):
        if user_input.strip() == "":
            st.warning("Peringatan: Harap masukkan teks ulasan terlebih dahulu.")
        else:
            with st.spinner("Sedang memproses..."):
                try:
                    X_topic = tfidf_topic.transform([user_input])
                    proba_topic = model_topic.predict_proba(X_topic)[0]
                    classes_topic = model_topic.classes_
                    df_topic_proba = pd.DataFrame({
                        "Topic": classes_topic,
                        "Probabilitas": proba_topic
                    }).sort_values(by="Probabilitas", ascending=False)
                    
                    y_topic_pred = df_topic_proba.iloc[0]["Topic"]
                    y_topic_prob = df_topic_proba.iloc[0]["Probabilitas"]

                    X_intention = tfidf_intention.transform([user_input])
                    proba_intention = model_intention.predict_proba(X_intention)[0]
                    classes_intention = model_intention.classes_
                    df_intention_proba = pd.DataFrame({
                        "Intention": classes_intention,
                        "Probabilitas": proba_intention
                    }).sort_values(by="Probabilitas", ascending=False)

                    y_intention_pred = df_intention_proba.iloc[0]["Intention"]
                    y_intention_prob = df_intention_proba.iloc[0]["Probabilitas"]

                    st.divider()
                    st.markdown("### Hasil Prediksi Tunggal")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Prediksi Topic", y_topic_pred, f"{y_topic_prob*100:.1f}% Keyakinan")
                    with col2:
                        st.metric("Prediksi Intention", y_intention_pred, f"{y_intention_prob*100:.1f}% Keyakinan")

                    with st.expander("Lihat Detail Probabilitas Prediksi"):
                        col_det1, col_det2 = st.columns(2)
                        with col_det1:
                            fig_topic = px.bar(
                                df_topic_proba.head(5), 
                                x="Probabilitas", 
                                y="Topic", 
                                orientation='h',
                                title="Keyakinan Prediksi Topic (Top 5)"
                            )
                            fig_topic.update_layout(yaxis={'categoryorder':'total ascending'}, height=350)
                            st.plotly_chart(fig_topic, use_container_width=True)

                        with col_det2:
                            fig_intention = px.bar(
                                df_intention_proba.head(5), 
                                x="Probabilitas", 
                                y="Intention", 
                                orientation='h',
                                title="Keyakinan Prediksi Intention (Top 5)"
                            )
                            fig_intention.update_layout(yaxis={'categoryorder':'total ascending'}, height=350)
                            st.plotly_chart(fig_intention, use_container_width=True)
                
                except AttributeError as e:
                    if "predict_proba" in str(e):
                        st.error("Error: Model yang digunakan tidak mendukung 'predict_proba'. Pastikan model stacking Anda memiliki opsi ini.")
                    else:
                        st.error(f"Error saat melakukan prediksi tunggal: {e}")
                except Exception as e:
                    st.error(f"Error saat melakukan prediksi tunggal: {e}")

    st.divider()
    st.subheader("Prediksi Batch (Unggah File)")
    st.markdown("Unggah file data mentah (.csv/.xlsx). File *harus* memiliki kolom bernama `full_text`.")

    batch_file = st.file_uploader(
        "Pilih File Data Mentah",
        type=["csv", "xlsx", "xls"],
        key="batch_uploader" 
    )

    if batch_file is not None:
        df_batch = None
        try:
            df_batch = load_data(batch_file)
        except Exception as e:
            st.stop() 

        if df_batch is not None:
            st.markdown("##### Tinjauan Data Mentah")
            st.dataframe(df_batch.head(), use_container_width=True)

            if "full_text" not in df_batch.columns:
                st.error("Error: Kolom `full_text` tidak ditemukan.")
                st.warning("Peringatan: Harap periksa file Anda dan pastikan nama kolom sudah benar.")
                st.stop() 
            else:
                st.success(f"Informasi: Kolom `full_text` terdeteksi. ({len(df_batch):,} baris siap diproses)")

            if st.button("Mulai Prediksi Batch"):
                with st.spinner(f"Memproses {len(df_batch):,} baris data... Ini mungkin perlu waktu."):
                    try:
                        texts = df_batch["full_text"].fillna("").astype(str)

                        X_topic = tfidf_topic.transform(texts)
                        proba_topic = model_topic.predict_proba(X_topic)
                        pred_topics = model_topic.classes_[np.argmax(proba_topic, axis=1)]
                        pred_topic_scores = np.max(proba_topic, axis=1) # Skor keyakinan

                        X_intention = tfidf_intention.transform(texts)
                        proba_intention = model_intention.predict_proba(X_intention)
                        pred_intentions = model_intention.classes_[np.argmax(proba_intention, axis=1)]
                        pred_intention_scores = np.max(proba_intention, axis=1) # Skor keyakinan

                        df_result = df_batch.copy()
                        df_result["topic"] = pred_topics
                        df_result["topic_probability"] = pred_topic_scores.round(4)
                        df_result["type"] = pred_intentions
                        df_result["type_probability"] = pred_intention_scores.round(4)

                        st.markdown("### Tinjauan Hasil Prediksi Batch")
                        st.dataframe(df_result.head(), use_container_width=True)

                        st.markdown("### Ringkasan Hasil Prediksi")
                        col_sum1, col_sum2 = st.columns(2)

                        with col_sum1:
                            st.markdown("#### Distribusi Topic yang Diprediksi")
                            topic_counts = df_result["topic"].value_counts().reset_index()
                            topic_counts.columns = ["Topic", "Jumlah"]
                            fig_topic_pie = px.pie(topic_counts, names="Topic", values="Jumlah", 
                                                   title="Proporsi Prediksi Topic", hole=0.3)
                            fig_topic_pie.update_traces(textposition='inside', textinfo='percent+label')
                            st.plotly_chart(fig_topic_pie, use_container_width=True)

                        with col_sum2:
                            st.markdown("#### Distribusi Intention yang Diprediksi")
                            type_counts = df_result["type"].value_counts().reset_index()
                            type_counts.columns = ["Intention", "Jumlah"]
                            fig_type_pie = px.pie(type_counts, names="Intention", values="Jumlah", 
                                                   title="Proporsi Prediksi Intention", hole=0.3)
                            fig_type_pie.update_traces(textposition='inside', textinfo='percent+label')
                            st.plotly_chart(fig_type_pie, use_container_width=True)
                        
                        @st.cache_data 
                        def convert_df_to_csv(df_to_convert):
                            return df_to_convert.to_csv(index=False).encode('utf-8')

                        csv_data = convert_df_to_csv(df_result)

                        st.download_button(
                            label="⬇️ Unduh Hasil Prediksi Lengkap (.csv)",
                            data=csv_data,
                            file_name=f"hasil_prediksi_{batch_file.name}.csv",
                            mime="text/csv",
                        )

                        st.info("💡 **Langkah Selanjutnya:** Anda sekarang dapat mengunggah file yang baru saja diunduh pada sidebar (Menu 'Dataset') untuk divisualisasikan.")
                    
                    except Exception as e:
                        st.error(f"Error: Terjadi kesalahan saat prediksi batch: {e}")

elif menu == "Evaluation":
    st.title("📈 Model Evaluation")

    topic_default_7030 = {
        "train_acc": 0.9871,"test_acc": 0.8550,
        "classes": ["gocar", "gofood", "gojek", "gopay", "gosend", "grab", "grabcar", "grabexpress", "grabfood", "ovo"],
        "report": [("gocar", 0.91, 0.89, 0.90, 180), ("gofood", 0.92, 0.82, 0.86, 180), ("gojek", 0.79, 0.87, 0.83, 180),("gopay", 0.98, 0.92, 0.95, 180), ("gosend", 0.96, 0.89, 0.93, 180), ("grab", 0.62, 0.78, 0.69, 180),("grabcar", 0.91, 0.82, 0.87, 180), ("grabexpress", 0.94, 0.91, 0.93, 180), ("grabfood", 0.78, 0.72, 0.75, 180),("ovo", 0.82, 0.93, 0.87, 180),],
        "macro_avg": (0.86, 0.86, 0.86, 1800),"weighted_avg": (0.86, 0.85, 0.86, 1800),
        "cm": np.array([[160, 0, 7, 1, 0, 3, 8, 0, 0, 1], [0, 147, 12, 1, 1, 1, 0, 0, 15, 3], [3, 1, 157, 1, 0, 14, 0, 0, 2, 2],[1, 1, 6, 165, 1, 0, 0, 0, 0, 6], [1, 0, 8, 0, 161, 3, 0, 3, 1, 3], [1, 0, 4, 0, 0, 141, 4, 3, 13, 14],[9, 0, 2, 0, 0, 20, 148, 0, 1, 0], [0, 0, 1, 0, 4, 8, 0, 164, 2, 1], [0, 11, 1, 0, 0, 27, 2, 3, 129, 7],[0, 0, 0, 1, 0, 9, 0, 1, 2, 167]])
    }
    topic_default_8020 = {
        "train_acc": 0.9871,"test_acc": 0.8550,
        "classes": ["gocar", "gofood", "gojek", "gopay", "gosend", "grab", "grabcar", "grabexpress", "grabfood", "ovo"],
        "report": [("gocar", 0.90, 0.87, 0.88, 120), ("gofood", 0.92, 0.81, 0.86, 120), ("gojek", 0.80, 0.86, 0.83, 120),("gopay", 0.98, 0.93, 0.95, 120), ("gosend", 0.97, 0.89, 0.93, 120), ("grab", 0.64, 0.80, 0.71, 120),("grabcar", 0.89, 0.82, 0.86, 120), ("grabexpress", 0.96, 0.92, 0.94, 120), ("grabfood", 0.74, 0.72, 0.73, 120),("ovo", 0.84, 0.93, 0.88, 120),],
        "macro_avg": (0.86, 0.85, 0.86, 1200),"weighted_avg": (0.86, 0.85, 0.86, 1200),
        "cm": np.array([[104, 0, 6, 0, 0, 2, 8, 0, 0, 0], [0, 97, 5, 1, 1, 2, 0, 0, 13, 1], [2, 1, 103, 1, 0, 9, 0, 0, 2, 2],[0, 1, 4, 111, 0, 1, 0, 0, 0, 3], [1, 0, 5, 0, 107, 0, 0, 3, 1, 3], [2, 0, 4, 0, 0, 96, 3, 0, 8, 7],[7, 0, 1, 0, 0, 10, 99, 0, 2, 1], [0, 0, 1, 0, 2, 5, 0, 110, 1, 1], [0, 7, 0, 0, 0, 20, 1, 1, 87, 4],[0, 0, 0, 0, 0, 4, 0, 0, 4, 112]])
    }
    topic_hyper_7030 = {
        "train_acc": 0.9571,"test_acc": 0.8594,
        "classes": ["gocar", "gofood", "gojek", "gopay", "gosend", "grab", "grabcar", "grabexpress", "grabfood", "ovo"],
        "report": [("gocar", 0.93, 0.89, 0.91, 180), ("gofood", 0.95, 0.80, 0.87, 180), ("gojek", 0.81, 0.87, 0.84, 180),("gopay", 0.96, 0.92, 0.94, 180), ("gosend", 0.95, 0.92, 0.93, 180), ("grab", 0.63, 0.81, 0.71, 180),("grabcar", 0.92, 0.82, 0.86, 180), ("grabexpress", 0.96, 0.90, 0.93, 180), ("grabfood", 0.77, 0.76, 0.76, 180),("ovo", 0.82, 0.92, 0.87, 180),],
        "macro_avg": (0.87, 0.86, 0.86, 1800),"weighted_avg": (0.87, 0.86, 0.86, 1800),
        "cm": np.array([[160, 0, 7, 1, 1, 4, 7, 0, 0, 0], [0, 144, 13, 1, 1, 2, 0, 0, 18, 1], [2, 0, 157, 1, 0, 14, 0, 1, 1, 4],[1, 1, 5, 165, 1, 1, 0, 0, 0, 6], [1, 0, 6, 0, 165, 2, 0, 1, 1, 4], [0, 0, 4, 0, 0, 146, 4, 2, 10, 14],[8, 0, 0, 0, 0, 21, 147, 0, 3, 1], [0, 0, 1, 0, 5, 7, 1, 162, 3, 1], [0, 7, 0, 1, 0, 28, 1, 2, 136, 5],[0, 0, 0, 2, 0, 8, 0, 0, 5, 165]])
    }
    topic_hyper_8020 = {
        "train_acc": 0.9406,"test_acc": 0.8583,
        "classes": ["gocar", "gofood", "gojek", "gopay", "gosend", "grab", "grabcar", "grabexpress", "grabfood", "ovo"],
        "report": [("gocar", 0.94, 0.87, 0.90, 120), ("gofood", 0.93, 0.82, 0.87, 120), ("gojek", 0.80, 0.86, 0.83, 120),("gopay", 0.98, 0.93, 0.95, 120), ("gosend", 0.96, 0.90, 0.93, 120), ("grab", 0.64, 0.81, 0.71, 120),("grabcar", 0.93, 0.82, 0.87, 120), ("grabexpress", 0.96, 0.90, 0.93, 120), ("grabfood", 0.75, 0.74, 0.74, 120),("ovo", 0.81, 0.94, 0.87, 120),],
        "macro_avg": (0.87, 0.86, 0.86, 1200),"weighted_avg": (0.87, 0.86, 0.86, 1200),
        "cm": np.array([[104, 0, 9, 0, 0, 1, 5, 0, 0, 1], [0, 98, 4, 1, 1, 1, 0, 0, 14, 1], [1, 1, 103, 1, 0, 10, 0, 0, 1, 3],[1, 1, 4, 111, 0, 0, 0, 0, 0, 3], [1, 0, 5, 0, 108, 1, 0, 1, 1, 3], [0, 0, 2, 0, 1, 97, 2, 1, 7, 10],[4, 0, 0, 0, 0, 13, 99, 1, 1, 2], [0, 0, 1, 0, 3, 4, 1, 108, 2, 1], [0, 5, 0, 0, 0, 22, 0, 2, 89, 2],[0, 0, 0, 0, 0, 3, 0, 0, 4, 113]])
    }

    intention_default_7030 = {
        "train_acc": 0.9250, "test_acc": 0.7028, 
        "classes": ["komplain", "pernyataan", "pertanyaan", "pujian", "saran-kritik"],
        "report": [("komplain", 0.73, 0.76, 0.74, 450), ("pernyataan", 0.58, 0.71, 0.64, 450), ("pertanyaan", 0.76, 0.68, 0.71, 300), ("pujian", 0.76, 0.69, 0.72, 300), ("saran-kritik", 0.80, 0.64, 0.71, 300),], 
        "macro_avg": (0.73, 0.70, 0.71, 1800), "weighted_avg": (0.71, 0.70, 0.70, 1800), 
        "cm": np.array([[342, 62, 10, 16, 20], [46, 321, 32, 39, 12], [24, 65, 203, 2, 6], [23, 53, 8, 207, 9], [35, 49, 15, 9, 192]]) 
    }
    intention_default_8020 = {
        "train_acc": 0.9185, "test_acc": 0.7108, 
        "classes": ["komplain", "pernyataan", "pertanyaan", "pujian", "saran-kritik"],
        "report": [("komplain", 0.73, 0.77, 0.75, 300), ("pernyataan", 0.61, 0.74, 0.67, 300), ("pertanyaan", 0.76, 0.70, 0.73, 200), ("pujian", 0.78, 0.68, 0.73, 200), ("saran-kritik", 0.77, 0.62, 0.69, 200),], 
        "macro_avg": (0.73, 0.70, 0.71, 1200), "weighted_avg": (0.72, 0.71, 0.71, 1200), 
        "cm": np.array([[230, 40, 8, 6, 16], [31, 222, 17, 21, 9], [11, 41, 141, 2, 5], [18, 32, 7, 135, 8], [24, 30, 13, 8, 125]]) 
    }
    intention_hyper_7030 = {
        "train_acc": 0.8886, "test_acc": 0.6972,
        "classes": ["komplain", "pernyataan", "pertanyaan", "pujian", "saran-kritik"],
        "report": [("komplain", 0.73, 0.75, 0.74, 450), ("pernyataan", 0.58, 0.72, 0.64, 450), ("pertanyaan", 0.75, 0.64, 0.69, 300), ("pujian", 0.75, 0.71, 0.73, 300), ("saran-kritik", 0.77, 0.63, 0.69, 300),], 
        "macro_avg": (0.72, 0.69, 0.70, 1800), "weighted_avg": (0.71, 0.70, 0.70, 1800), 
        "cm": np.array([[339, 59, 12, 20, 20], [44, 322, 34, 36, 14], [25, 68, 193, 4, 10], [21, 48, 8, 212, 11], [37, 54, 10, 10, 189]]) 
    }
    intention_hyper_8020 = {
        "train_acc": 0.8829, "test_acc": 0.7075, 
        "classes": ["komplain", "pernyataan", "pertanyaan", "pujian", "saran-kritik"],
        "report": [("komplain", 0.74, 0.76, 0.75, 300), ("pernyataan", 0.59, 0.75, 0.66, 300), ("pertanyaan", 0.76, 0.69, 0.72, 200), ("pujian", 0.80, 0.70, 0.74, 200), ("saran-kritik", 0.78, 0.60, 0.68, 200),], 
        "macro_avg": (0.73, 0.70, 0.71, 1200), "weighted_avg": (0.72, 0.71, 0.71, 1200), 
        "cm": np.array([[228, 43, 8, 8, 13], [30, 224, 19, 19, 8], [12, 46, 137, 1, 4], [15, 32, 4, 140, 9], [25, 35, 12, 8, 120]]) 
    }

    topic_eval = {
        "Parameter Default": {"70/30": topic_default_7030,"80/20": topic_default_8020},
        "Hyperparameter Tuned": {"70/30": topic_hyper_7030,"80/20": topic_hyper_8020}
    }
    intention_eval = {
        "Parameter Default": {"70/30": intention_default_7030,"80/20": intention_default_8020},
        "Hyperparameter Tuned": {"70/30": intention_hyper_7030,"80/20": intention_hyper_8020}
    }

    st.markdown("#### Filter Tampilan Evaluasi")
    filter_c1, filter_c2, filter_c3 = st.columns(3)

    with filter_c1: model_choice = st.selectbox("Pilih Model:", ["Topic", "Intention"])
    with filter_c2: experiment_type = st.selectbox("Pilih Tipe Eksperimen:",["Parameter Default", "Hyperparameter Tuned"])
    with filter_c3: split = st.selectbox("Pilih Split Data:", ["70/30", "80/20"])

    st.divider()

    try:
        selected_eval = topic_eval if model_choice == "Topic" else intention_eval
        data_eval = selected_eval[experiment_type][split]
        st.subheader(f"Hasil Evaluasi: Model {model_choice} ({experiment_type} - {split})")
    except KeyError:
        st.error("Kombinasi filter tidak valid atau data tidak ditemukan."); st.stop()

    classes = data_eval["classes"]
    c1, c2 = st.columns(2)
    c1.metric("Train Accuracy", f"{data_eval['train_acc']:.4f}")
    c2.metric("Test Accuracy", f"{data_eval['test_acc']:.4f}")

    st.divider()
    st.subheader("Classification Report (Test Set)")

    rep_df = pd.DataFrame(data_eval["report"], columns=["Label", "Precision", "Recall", "F1-Score", "Support"])
    macro_p, macro_r, macro_f1, macro_sup = data_eval["macro_avg"]
    w_p, w_r, w_f1, w_sup = data_eval["weighted_avg"]

    if "Macro Avg" not in rep_df["Label"].values:
        rep_df = pd.concat([rep_df,pd.DataFrame([["Macro Avg", macro_p, macro_r, macro_f1, macro_sup],["Weighted Avg", w_p, w_r, w_f1, w_sup],], columns=rep_df.columns)], ignore_index=True)

    st.dataframe(rep_df, use_container_width=True, hide_index=True) 

    csv = rep_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Report",csv,file_name=f"report_{model_choice}_{experiment_type.replace(' ','_')}_{split.replace('/','-')}.csv",mime="text/csv")

    st.divider()
    st.subheader("Confusion Matrix (Test Set)")

    cm = data_eval["cm"]
    fig = go.Figure(data=go.Heatmap(z=cm,x=classes,y=classes,colorscale="Blues",text=cm,texttemplate="%{text}",hovertemplate="Actual: %{y}<br>Pred: %{x}<br>Count: %{z}<extra></extra>"))
    fig.update_layout(xaxis_title="Predicted Label",yaxis_title="Actual Label",height=600,margin=dict(l=40, r=40, t=40, b=40))
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Tampilkan Matrix Ternormalisasi per Kelas"):
        checkbox_key = f"norm_cm_check_{model_choice}_{split}_{experiment_type}"
        if st.checkbox(f"Aktifkan Normalisasi Confusion Matrix", key=checkbox_key):
            row_sums = cm.sum(axis=1, keepdims=True)
            with np.errstate(divide='ignore', invalid='ignore'):
                cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)
                cm_norm = np.nan_to_num(cm_norm)

            fig_norm = go.Figure(data=go.Heatmap(z=cm_norm,x=classes,y=classes,colorscale="Blues",text=np.round(cm_norm, 2),texttemplate="%{text}",hovertemplate="Actual: %{y}<br>Pred: %{x}<br>Proporsi: %{z:.2f}<extra></extra>",zmin=0, zmax=1,showscale=True))
            fig_norm.update_layout(xaxis_title="Predicted Label",yaxis_title="Actual Label",title=f"Normalized Confusion Matrix ({model_choice} {split})",height=600,margin=dict(l=40, r=40, t=50, b=40))
            st.plotly_chart(fig_norm, use_container_width=True)

    st.divider()
    st.subheader("Analisis Kesalahan Terbesar (Test Set)")

    cm = data_eval["cm"]
    classes = data_eval["classes"]
    cm_errors = cm.copy()
    np.fill_diagonal(cm_errors, 0)
    n_errors = 3
    flat_indices = np.argsort(cm_errors.ravel())
    top_indices = np.unravel_index(flat_indices, cm_errors.shape)
    indices_pairs = list(zip(top_indices[0], top_indices[1]))

    st.warning("Berikut adalah 3 kesalahan klasifikasi yang paling sering dilakukan oleh model:")
    error_count = 0
    for i in range(len(indices_pairs) - 1, -1, -1):
        if error_count >= n_errors: break
        actual_idx, pred_idx = indices_pairs[i]
        count = cm_errors[actual_idx, pred_idx]
        if count > 0:
             error_count += 1
             actual_label = classes[actual_idx]
             pred_label = classes[pred_idx]
             st.markdown(f"**{error_count}.** Salah prediksi **'{actual_label}'** sebagai **'{pred_label}'** ({count} kali)")
    if error_count == 0: st.info("Tidak ada kesalahan klasifikasi yang signifikan (di luar diagonal utama).")
    
elif menu == "About":
    st.title("Tentang Aplikasi") 
    st.caption("Versi 1.0 · Oktober 2025")

    st.markdown("### Deskripsi Aplikasi")
    st.markdown(
        """
        Aplikasi ini merupakan **prototipe dashboard interaktif** yang dirancang untuk menganalisis **topik** dan **intensi** dari ulasan konsumen Gojek dan Grab di Twitter.  
        Tujuannya adalah untuk **mengubah ribuan data teks yang tidak terstruktur** menjadi **wawasan visual yang mudah dipahami**, guna mendukung proses **pengambilan keputusan berbasis data**.
        """
    )

    st.divider()

    st.markdown("### Informasi Pengerjaan")
    
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            <div class="prediction-box" style="margin-top: 0;">
                <h4 style="margin-top: 0px; margin-bottom: 12px;">Pengembang</h4>
                <ul style="margin-bottom: 0px; padding-left: 20px;">
                    <li><b>Nama</b> : Hans Santoso</li>
                    <li><b>NIM</b> : 535220129</li>
                    <li><b>Program Studi</b> : S1 Teknik Informatika</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            """
            <div class="prediction-box" style="margin-top: 0;">
                <h4 style="margin-top: 0px; margin-bottom: 12px;">Dosen Pembimbing</h4>
                <p style="margin-bottom: 0px;">Tri Sutrisno, S.Si., M.Sc.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("") 
    st.markdown("### Konteks Akademis")
    
    st.info(
        """
        Aplikasi ini dikembangkan sebagai bagian dari pengerjaan **Skripsi** untuk memenuhi salah satu syarat kelulusan **Program Sarjana (S1)** di **Fakultas Teknologi Informasi, Universitas Tarumanagara**.
        """,
        icon="🎓"
    )

st.markdown(
    """
    <div class="footer">
        <p style="margin: 0;">© 2025 · Prototipe Dashboard Analisis Ulasan</p>
    </div>
    """,
    unsafe_allow_html=True
)