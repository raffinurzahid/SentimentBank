import streamlit as st
import pickle
import os

# Set page title
st.set_page_config(page_title="Analisis Sentimen Perbankan")

# Load model and preprocessing tools
@st.cache_resource
def load_model():
    model_path = os.path.join("model", "model_clf.pkl")
    tfidf_path = os.path.join("model", "tfidf.pkl")
    selector_path = os.path.join("model", "selector.pkl")
    
    model = pickle.load(open(model_path, "rb"))
    tfidf = pickle.load(open(tfidf_path, "rb"))
    selector = pickle.load(open(selector_path, "rb"))
    
    return model, tfidf, selector

# Load models
try:
    model, tfidf, selector = load_model()
    models_loaded = True
except Exception as e:
    st.error(f"Error loading models: {str(e)}")
    models_loaded = False

# App header
st.write("""# Sentiment Perbankan
Sentimen analisis dengan menggunakan Prerpocessing NLP, system ini dibangun dengan  menggunakan dataset yang diambil dari ulasan tiga aplikasi m-banking 
yang tujuanya untuk mengetahui bedasarkan respon dari para user mana yang merupakan aplikasi m-banking yang memiliki respon positif terbanyak. 
Model yang digunakan untuk membangun system ini adalah Naive Bayes.
""")

st.write("")

# User input
input_sentiment = st.text_area("**Masukan teks untuk dianalisis:**", height=100)

# Analysis button
if st.button('Analisis Sentiment', type="primary"):
    st.write("\n")
    st.write("\n")
    st.write("\n")
    if input_sentiment:
        if models_loaded:
            try:
                # Process text and predict
                text_tfidf = tfidf.transform([input_sentiment])
                text_selected = selector.transform(text_tfidf)
                sentiment = model.predict(text_selected)[0]
                
                st.write("**Hasil Analisis:**")
                
                # Display result
                if sentiment == 'positive':
                    st.success(f"Sentimen: Positif")
                    st.balloons()
                else:
                    st.error(f"Sentimen: Negatif")
            except Exception as e:
                st.error(f"Error dalam analisis: {str(e)}")
        else:
            st.warning("Model tidak dapat dimuat. Harap periksa file model.")
    else:
        st.warning("Silakan masukkan teks untuk dianalisis.")