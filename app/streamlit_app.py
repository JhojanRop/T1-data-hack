import streamlit as st
import pandas as pd
import plotly.express as px
import shap
from src.models.predictor import Predictor

def main():
    st.title("Clasificador de Literatura Biomédica")
    
    # Input
    title = st.text_input("Título del artículo:")
    abstract = st.text_area("Abstract:")
    
    if st.button("Clasificar"):
        # Predicción
        predictor = Predictor()
        predictions, probas = predictor.predict(title, abstract)
        
        # Visualización
        st.subheader("Resultados")
        fig = px.bar(
            x=predictor.classes,
            y=probas[0],
            title="Probabilidades por categoría"
        )
        st.plotly_chart(fig)
        
        # SHAP values
        st.subheader("Explicación")
        explainer = shap.Explainer(predictor.model)
        shap_values = explainer([predictor.transform(title, abstract)])
        st.pyplot(shap.plots.bar(shap_values[0]))

if __name__ == "__main__":
    main()
