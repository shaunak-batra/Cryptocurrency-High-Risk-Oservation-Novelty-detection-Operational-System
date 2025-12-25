"""
CHRONOS Dashboard - Error Analysis
Analyze model errors and misclassifications using real predictions.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(page_title="Error Analysis", page_icon="🔴", layout="wide")

st.title("🔴 Error Analysis")
st.markdown("Analyze model errors and misclassifications")
st.markdown("---")

@st.cache_data
def load_predictions():
    """Load real predictions from Colab inference."""
    path = 'results/real_data/predictions.csv'
    if os.path.exists(path):
        df = pd.read_csv(path)
        return df, True
    return None, False

@st.cache_data
def load_confusion_matrix():
    """Load confusion matrix from Colab inference."""
    path = 'results/real_data/confusion_matrix.csv'
    if os.path.exists(path):
        df = pd.read_csv(path, index_col=0)
        return df, True
    return None, False

@st.cache_data
def load_metrics():
    """Load test metrics from Colab inference."""
    path = 'results/real_data/test_metrics.csv'
    if os.path.exists(path):
        df = pd.read_csv(path)
        return dict(zip(df['metric'], df['value'])), True
    return None, False

# Load data
predictions_df, has_predictions = load_predictions()
cm_df, has_cm = load_confusion_matrix()
metrics, has_metrics = load_metrics()

if not has_predictions:
    st.warning("⚠️ No real predictions found. Run Colab inference to generate them.")
    st.stop()

st.success("✅ Using real model predictions from Colab inference")

# Overview metrics
st.subheader("📊 Classification Metrics")

if has_metrics:
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("F1 Score", f"{metrics.get('f1_score', 0):.4f}")
    with col2:
        st.metric("Precision", f"{metrics.get('precision', 0):.4f}")
    with col3:
        st.metric("Recall", f"{metrics.get('recall', 0):.4f}")
    with col4:
        total = len(predictions_df)
        correct = ((predictions_df['prediction'] == predictions_df['label'])).sum()
        accuracy = correct / total if total > 0 else 0
        st.metric("Accuracy", f"{accuracy:.4f}")

st.markdown("---")

# Confusion Matrix Visualization
st.subheader("📈 Confusion Matrix")

if has_cm:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = go.Figure(data=go.Heatmap(
            z=cm_df.values,
            x=['Pred Licit', 'Pred Illicit'],
            y=['Actual Licit', 'Actual Illicit'],
            colorscale='RdYlGn_r',
            text=cm_df.values,
            texttemplate='%{text}',
            textfont={"size": 20},
            hoverongaps=False
        ))
        fig.update_layout(
            template='plotly_dark',
            height=400,
            title="Confusion Matrix"
        )
        st.plotly_chart(fig, use_container_width=True, key="confusion_matrix")
    
    with col2:
        st.markdown("### Interpretation")
        tn, fp = cm_df.iloc[0]
        fn, tp = cm_df.iloc[1]
        
        st.write(f"**True Positives (TP):** {int(tp)}")
        st.write(f"**True Negatives (TN):** {int(tn)}")
        st.write(f"**False Positives (FP):** {int(fp)}")
        st.write(f"**False Negatives (FN):** {int(fn)}")
        
        st.write("")
        st.write(f"FP Rate: {fp/(fp+tn)*100:.2f}%" if (fp+tn) > 0 else "N/A")
        st.write(f"FN Rate: {fn/(fn+tp)*100:.2f}%" if (fn+tp) > 0 else "N/A")

st.markdown("---")

# Error Distribution
st.subheader("📊 Error Distribution by Probability")

# Identify errors
predictions_df['correct'] = predictions_df['prediction'] == predictions_df['label']
predictions_df['error_type'] = 'Correct'
predictions_df.loc[(predictions_df['label'] == 1) & (predictions_df['prediction'] == 0), 'error_type'] = 'False Negative'
predictions_df.loc[(predictions_df['label'] == 0) & (predictions_df['prediction'] == 1), 'error_type'] = 'False Positive'

errors_df = predictions_df[predictions_df['error_type'] != 'Correct']

if len(errors_df) > 0:
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(
            errors_df, x='probability', color='error_type',
            nbins=50, marginal='box',
            color_discrete_map={'False Positive': '#FF6B6B', 'False Negative': '#FFA500'},
            title="Error Distribution by Prediction Probability"
        )
        fig.update_layout(template='plotly_dark', height=400)
        st.plotly_chart(fig, use_container_width=True, key="error_dist")
    
    with col2:
        error_counts = errors_df['error_type'].value_counts()
        fig = px.pie(
            values=error_counts.values,
            names=error_counts.index,
            color=error_counts.index,
            color_discrete_map={'False Positive': '#FF6B6B', 'False Negative': '#FFA500'},
            title="Error Type Breakdown"
        )
        fig.update_layout(template='plotly_dark', height=400)
        st.plotly_chart(fig, use_container_width=True, key="error_pie")
else:
    st.info("No errors found - perfect classification!")

st.markdown("---")

# Threshold Analysis
st.subheader("🎯 Threshold Impact Analysis")

threshold = st.slider("Classification Threshold", 0.0, 1.0, 0.5, 0.01)

# Recalculate predictions with new threshold
new_preds = (predictions_df['probability'] >= threshold).astype(int)
new_correct = (new_preds == predictions_df['label']).sum()
new_fp = ((new_preds == 1) & (predictions_df['label'] == 0)).sum()
new_fn = ((new_preds == 0) & (predictions_df['label'] == 1)).sum()
new_tp = ((new_preds == 1) & (predictions_df['label'] == 1)).sum()
new_tn = ((new_preds == 0) & (predictions_df['label'] == 0)).sum()

new_precision = new_tp / (new_tp + new_fp) if (new_tp + new_fp) > 0 else 0
new_recall = new_tp / (new_tp + new_fn) if (new_tp + new_fn) > 0 else 0
new_f1 = 2 * new_precision * new_recall / (new_precision + new_recall) if (new_precision + new_recall) > 0 else 0

col1, col2, col3, col4 = st.columns(4)
col1.metric("New F1", f"{new_f1:.4f}", delta=f"{new_f1 - metrics.get('f1_score', 0):.4f}" if has_metrics else None)
col2.metric("New Precision", f"{new_precision:.4f}")
col3.metric("New Recall", f"{new_recall:.4f}")
col4.metric("Accuracy", f"{new_correct/len(predictions_df):.4f}")

st.markdown("---")

# High Confidence Errors
st.subheader("⚠️ High Confidence Errors")

high_conf_errors = errors_df[
    ((errors_df['error_type'] == 'False Positive') & (errors_df['probability'] > 0.8)) |
    ((errors_df['error_type'] == 'False Negative') & (errors_df['probability'] < 0.2))
]

if len(high_conf_errors) > 0:
    st.write(f"Found **{len(high_conf_errors)}** high confidence errors (model was very wrong)")
    st.dataframe(high_conf_errors.head(20), use_container_width=True)
else:
    st.success("✅ No high confidence errors found!")

st.info(f"Total test samples: {len(predictions_df)} | Total errors: {len(errors_df)} | Error rate: {len(errors_df)/len(predictions_df)*100:.2f}%")
