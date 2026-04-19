"""Unified Streamlit App — Solar Forecasting + Grid Optimization Assistant."""

import os
import io
import json
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from dotenv import load_dotenv
from agent import agent, chat_with_context

# ── Initialization ────────────────────────────────────────────────────────────

load_dotenv(override=True)

# Pull secrets from st.secrets when env vars are absent (Streamlit Cloud)
for _key in ("GROQ_API_KEY", "HF_REPO_ID", "HF_TOKEN"):
    if not os.environ.get(_key):
        try:
            os.environ[_key] = st.secrets[_key]
        except Exception:
            pass

st.set_page_config(page_title="Solar Energy AI System", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .stMetric { background: linear-gradient(135deg, #667eea22, #764ba222);
                border-radius: 10px; padding: 15px; border: 1px solid #e0e0e0; }
    div[data-testid="stExpander"] { border: 1px solid #e0e0e0; border-radius: 8px;
                                     margin-bottom: 8px; }
    .rec-card { background: rgba(102, 126, 234, 0.1); border-left: 4px solid #667eea;
                padding: 16px 20px; border-radius: 0 8px 8px 0; margin-bottom: 12px; }
    .rec-card-warn { background: rgba(249, 115, 22, 0.1); border-left: 4px solid #f97316;
                     padding: 16px 20px; border-radius: 0 8px 8px 0; margin-bottom: 12px; }
    .rec-card h4 { margin: 0 0 8px 0; color: #93a3f8; }
    .rec-card p, .rec-card-warn p { margin: 0; color: inherit; line-height: 1.6; }
    .rec-card-warn h4 { margin: 0 0 8px 0; color: #fb923c; }
    .action-item { background: rgba(255, 255, 255, 0.05); border: 1px solid rgba(255,255,255,0.12);
                   border-radius: 8px; padding: 12px 16px; margin-bottom: 8px;
                   display: flex; align-items: flex-start; color: inherit; }
    .action-num { background: #667eea; color: white; border-radius: 50%; width: 24px;
                  height: 24px; display: inline-flex; align-items: center; justify-content: center;
                  font-size: 12px; font-weight: bold; margin-right: 12px; flex-shrink: 0; }
    .strategy-item { background: rgba(34, 197, 94, 0.08); border: 1px solid rgba(34,197,94,0.25);
                     border-radius: 8px; padding: 12px 16px; margin-bottom: 8px; color: inherit; }
    .ref-tag { display: inline-block; background: rgba(255,255,255,0.1); color: inherit;
               padding: 4px 12px; border-radius: 16px; font-size: 13px; margin: 4px;
               border: 1px solid rgba(255,255,255,0.15); }
    .risk-high { background: rgba(239, 68, 68, 0.15); border: 1px solid rgba(239,68,68,0.4);
                 border-radius: 10px; padding: 15px; color: #fca5a5; }
    .risk-medium { background: rgba(234, 179, 8, 0.15); border: 1px solid rgba(234,179,8,0.4);
                   border-radius: 10px; padding: 15px; color: #fde68a; }
    .risk-low { background: rgba(34, 197, 94, 0.15); border: 1px solid rgba(34,197,94,0.4);
                border-radius: 10px; padding: 15px; color: #86efac; }
</style>
""", unsafe_allow_html=True)

st.title("AI-Based Solar Energy System")

# ── Model loading (once, cached) ──────────────────────────────────────────────

HF_REPO_ID = os.environ.get("HF_REPO_ID", "")
LOCAL_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "solar_forecast_model.pkl")


@st.cache_resource(show_spinner="Loading forecast model...")
def load_model():
    if HF_REPO_ID:
        from huggingface_hub import hf_hub_download
        token = os.environ.get("HF_TOKEN") or None
        path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename="solar_forecast_model.pkl",
            token=token,
        )
        return joblib.load(path)
    # Local fallback for development
    if not os.path.exists(LOCAL_MODEL_PATH):
        raise FileNotFoundError(LOCAL_MODEL_PATH)
    return joblib.load(LOCAL_MODEL_PATH)


try:
    model = load_model()
except FileNotFoundError:
    st.error(
        "Model not found. Either set `HF_REPO_ID` in `.env` to pull from HuggingFace, "
        "or place `solar_forecast_model.pkl` in the project directory."
    )
    st.stop()
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# ── Required columns ─────────────────────────────────────────────────────────

GEN_REQUIRED = {"DATE_TIME", "DC_POWER"}
WEATHER_REQUIRED = {"DATE_TIME", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"}


# ── Helper functions (must be defined before tabs) ────────────────────────────

def _generate_pdf(analysis: dict, recommendations: dict) -> bytes:
    """Generate a PDF report using fpdf2 (pure Python, no system dependencies)."""
    from fpdf import FPDF
    from datetime import date
    import io

    metrics = st.session_state.get("forecast_metrics", {})

    class ReportPDF(FPDF):
        def header(self):
            if self.page_no() == 1:
                return
            self.set_font("Helvetica", "B", 8)
            self.set_text_color(100, 100, 100)
            self.cell(0, 6, "Solar Energy Forecasting — Grid Optimization Report", align="L")
            self.ln(2)
            self.set_draw_color(46, 134, 193)
            self.set_line_width(0.3)
            self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
            self.ln(4)

        def footer(self):
            if self.page_no() == 1:
                return
            self.set_y(-15)
            self.set_font("Helvetica", "", 8)
            self.set_text_color(150, 150, 150)
            self.cell(0, 10, f"Page {self.page_no() - 1}", align="C")

    pdf = ReportPDF()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.set_margins(20, 20, 20)

    # ── Title page ──────────────────────────────────────────────────────
    pdf.add_page()
    pdf.ln(35)
    pdf.set_draw_color(46, 134, 193)
    pdf.set_line_width(1)
    pdf.line(20, pdf.get_y(), pdf.w - 20, pdf.get_y())
    pdf.ln(10)

    pdf.set_font("Helvetica", "B", 24)
    pdf.set_text_color(0, 82, 155)
    pdf.multi_cell(0, 11, "Solar Energy Forecasting\nGrid Optimization Report", align="C")
    pdf.ln(8)

    pdf.set_draw_color(46, 134, 193)
    pdf.line(20, pdf.get_y(), pdf.w - 20, pdf.get_y())
    pdf.ln(20)

    report_date = date.today().strftime("%B %d, %Y")
    for label, val in [
        ("Project:", "Solar Power Generation Forecasting"),
        ("Dataset:", "Kaggle Solar Plant Operational Data"),
        ("Model:", "Random Forest Regressor (200 trees, depth 12)"),
        ("Date:", report_date),
    ]:
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(50, 50, 50)
        pdf.cell(45, 9, label)
        pdf.set_font("Helvetica", "", 11)
        pdf.multi_cell(0, 9, val)

    # ── Content pages ────────────────────────────────────────────────────
    pdf.add_page()

    def section_header(title):
        pdf.set_font("Helvetica", "B", 13)
        pdf.set_text_color(0, 82, 155)
        pdf.cell(0, 9, title, ln=True)
        pdf.set_draw_color(46, 134, 193)
        pdf.set_line_width(0.5)
        pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
        pdf.ln(5)
        pdf.set_text_color(50, 50, 50)

    def subsection_header(title):
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(0, 82, 155)
        pdf.cell(0, 8, title, ln=True)
        pdf.ln(1)
        pdf.set_text_color(50, 50, 50)

    def table_row(label, value, shade=False):
        pdf.set_fill_color(240, 245, 255) if shade else pdf.set_fill_color(255, 255, 255)
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(50, 50, 50)
        pdf.cell(110, 7, label, border=0, fill=True)
        pdf.cell(0, 7, str(value), border=0, fill=True, ln=True)

    def table_header():
        pdf.set_fill_color(0, 82, 155)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(110, 7, "Metric", border=0, fill=True)
        pdf.cell(0, 7, "Value", border=0, fill=True, ln=True)

    # Forecast Analysis
    if analysis:
        section_header("Forecast Analysis")
        rows = [
            ("Mean Power (W)", f"{analysis.get('mean_power', 0):,.2f}"),
            ("Max Power (W)", f"{analysis.get('max_power', 0):,.2f}"),
            ("Min Power (W)", f"{analysis.get('min_power', 0):,.2f}"),
            ("Std. Deviation (W)", f"{analysis.get('std_power', 0):,.2f}"),
            ("Variability (%)", f"{analysis.get('variability_pct', 0):.2f}"),
            ("Risk Level", str(analysis.get('risk_level', 'N/A'))),
            ("Mean Irradiation (kW/m2)", f"{analysis.get('mean_irradiation', 0):.4f}"),
        ]
        table_header()
        for i, (label, val) in enumerate(rows):
            table_row(label, val, shade=(i % 2 == 0))
        pdf.ln(8)

    # Model Performance
    if metrics:
        section_header("Model Performance")
        mrows = [
            ("Mean Absolute Error", str(metrics.get("mae", "N/A"))),
            ("Root Mean Squared Error", str(metrics.get("rmse", "N/A"))),
            ("R2 Score", str(metrics.get("r2", "N/A"))),
            ("Anomalies Detected", str(metrics.get("anomaly_count", "N/A"))),
            ("Total Data Points", str(metrics.get("total_points", "N/A"))),
        ]
        table_header()
        for i, (label, val) in enumerate(mrows):
            table_row(label, val, shade=(i % 2 == 0))
        pdf.ln(8)

    # Recommendations
    if recommendations:
        section_header("Recommendations")

        summary = recommendations.get("forecast_summary", "")
        if summary:
            subsection_header("Forecast Summary")
            pdf.set_font("Helvetica", "", 10)
            pdf.multi_cell(0, 6, str(summary))
            pdf.ln(4)

        risk_text = recommendations.get("risk_analysis", "")
        if risk_text:
            subsection_header("Risk Assessment")
            pdf.set_font("Helvetica", "", 10)
            pdf.multi_cell(0, 6, str(risk_text))
            pdf.ln(4)

        actions = recommendations.get("actions", [])
        if actions:
            subsection_header("Immediate Actions")
            for i, action in enumerate(actions, 1):
                if isinstance(action, dict):
                    action = action.get("action", action.get("strategy", str(action)))
                pdf.set_font("Helvetica", "", 10)
                pdf.multi_cell(0, 6, f"{i}. {action}")
            pdf.ln(4)

        strategies = recommendations.get("optimization_strategies", [])
        if strategies:
            subsection_header("Optimization Strategies")
            for i, s in enumerate(strategies, 1):
                if isinstance(s, dict):
                    s = s.get("strategy", s.get("action", str(s)))
                pdf.set_font("Helvetica", "", 10)
                pdf.multi_cell(0, 6, f"{i}. {s}")
            pdf.ln(4)

        refs = recommendations.get("references", [])
        if refs:
            subsection_header("Referenced Guidelines")
            for ref in refs:
                pdf.set_font("Helvetica", "", 10)
                pdf.multi_cell(0, 6, f"- {ref}")

    buf = io.BytesIO()
    pdf.output(buf)
    return buf.getvalue()


tab1, tab2 = st.tabs(["Solar Forecasting", "Grid Optimization Assistant"])

# ── Tab 1: Solar Forecasting ─────────────────────────────────────────────────

with tab1:
    st.header("Solar Power Forecasting")

    col_up1, col_up2 = st.columns(2)
    with col_up1:
        gen_file = st.file_uploader("Upload Generation Data", type=["csv"], key="gen")
    with col_up2:
        weather_file = st.file_uploader("Upload Weather Data", type=["csv"], key="weather")

    if gen_file is not None and weather_file is not None:
        gen_df = pd.read_csv(gen_file)
        weather_df = pd.read_csv(weather_file)

        # Column validation
        gen_missing = GEN_REQUIRED - set(gen_df.columns)
        weather_missing = WEATHER_REQUIRED - set(weather_df.columns)
        if gen_missing or weather_missing:
            if gen_missing:
                st.error(f"Generation CSV missing columns: {', '.join(gen_missing)}")
            if weather_missing:
                st.error(f"Weather CSV missing columns: {', '.join(weather_missing)}")
            st.stop()

        gen_df["DATE_TIME"] = pd.to_datetime(gen_df["DATE_TIME"])
        weather_df["DATE_TIME"] = pd.to_datetime(weather_df["DATE_TIME"])

        gen_agg = (
            gen_df.groupby("DATE_TIME").agg({"DC_POWER": "sum"}).reset_index()
        )

        weather_cols = [
            "DATE_TIME",
            "AMBIENT_TEMPERATURE",
            "MODULE_TEMPERATURE",
            "IRRADIATION",
        ]
        df = pd.merge(
            gen_agg, weather_df[weather_cols], on="DATE_TIME", how="inner"
        )

        df = df.sort_values("DATE_TIME").reset_index(drop=True)

        df["hour"] = df["DATE_TIME"].dt.hour
        df["dayofyear"] = df["DATE_TIME"].dt.dayofyear
        df["month"] = df["DATE_TIME"].dt.month
        df["lag_1"] = df["DC_POWER"].shift(1)
        df["lag_4"] = df["DC_POWER"].shift(4)
        df["rolling_mean_4"] = df["DC_POWER"].rolling(4).mean()

        df["target"] = df["DC_POWER"].shift(-1)

        df = df.dropna().reset_index(drop=True)

        X = df.drop(columns=["DATE_TIME", "DC_POWER", "target"])
        y = df["target"]

        preds = model.predict(X)
        errors = np.abs(y.values - preds)

        mae = mean_absolute_error(y, preds)
        rmse = np.sqrt(mean_squared_error(y, preds))
        r2 = r2_score(y, preds)

        # ── Metrics ──────────────────────────────────────────────────────
        st.divider()
        st.subheader("Model Performance")
        col1, col2, col3 = st.columns(3)
        col1.metric("MAE", f"{mae:.2f}")
        col2.metric("RMSE", f"{rmse:.2f}")
        col3.metric("R² Score", f"{r2:.4f}")

        # ── Feature 2: Forecast range slider ─────────────────────────────
        st.divider()
        st.subheader("15-Minute Ahead Forecast")
        max_points = len(preds)
        display_range = st.slider(
            "Data points to display",
            min_value=50,
            max_value=min(max_points, 1000),
            value=min(200, max_points),
            step=50,
            key="forecast_range",
        )
        n = display_range

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(y.values[:n], label="Actual", linewidth=1.5, color="#667eea")
        ax.plot(preds[:n], label="Predicted", linewidth=1.5, color="#f97316", linestyle="--")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("DC Power")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

        # ── Feature 6: Anomaly detection ─────────────────────────────────
        st.divider()
        st.subheader("Anomaly Detection")
        anomaly_threshold = st.slider(
            "Error threshold multiplier (× MAE)",
            min_value=1.5,
            max_value=5.0,
            value=2.5,
            step=0.5,
            key="anomaly_thresh",
        )
        threshold_val = mae * anomaly_threshold
        anomaly_mask = errors > threshold_val
        anomaly_count = int(anomaly_mask.sum())

        col_a1, col_a2 = st.columns([1, 3])
        with col_a1:
            st.metric("Anomalies Detected", anomaly_count)
            st.metric("Threshold", f"{threshold_val:,.0f}")
            pct = (anomaly_count / len(preds)) * 100
            st.metric("Anomaly Rate", f"{pct:.1f}%")

        with col_a2:
            fig_a, ax_a = plt.subplots(figsize=(12, 4))
            ax_a.plot(y.values[:n], label="Actual", linewidth=1, color="#667eea", alpha=0.7)
            ax_a.plot(preds[:n], label="Predicted", linewidth=1, color="#f97316", alpha=0.7)
            # Highlight anomalies
            anomaly_indices = np.where(anomaly_mask[:n])[0]
            if len(anomaly_indices) > 0:
                ax_a.scatter(
                    anomaly_indices,
                    y.values[:n][anomaly_indices],
                    color="#ff4b4b",
                    s=20,
                    zorder=5,
                    label=f"Anomalies ({len(anomaly_indices)})",
                )
            ax_a.set_xlabel("Time Step")
            ax_a.set_ylabel("DC Power")
            ax_a.legend()
            ax_a.grid(True, alpha=0.3)
            st.pyplot(fig_a)
            plt.close(fig_a)

        # ── Trend Analysis ───────────────────────────────────────────────
        st.divider()
        st.subheader("Trend Analysis")
        col_t1, col_t2 = st.columns(2)

        with col_t1:
            hourly_trend = df.groupby("hour")["DC_POWER"].mean()
            fig2, ax2 = plt.subplots()
            ax2.bar(hourly_trend.index, hourly_trend.values, color="#667eea", alpha=0.8)
            ax2.set_title("Average Power by Hour")
            ax2.set_xlabel("Hour of Day")
            ax2.set_ylabel("Avg DC Power")
            ax2.grid(True, alpha=0.3, axis="y")
            st.pyplot(fig2)
            plt.close(fig2)

        with col_t2:
            fig3, ax3 = plt.subplots()
            ax3.scatter(df["IRRADIATION"][:n], preds[:n], alpha=0.5, s=10, color="#764ba2")
            ax3.set_title("Irradiation vs Predicted Power")
            ax3.set_xlabel("Irradiation")
            ax3.set_ylabel("Predicted DC Power")
            ax3.grid(True, alpha=0.3)
            st.pyplot(fig3)
            plt.close(fig3)

        # ── Feature 1: Download predictions CSV ─────────────────────────
        st.divider()
        export_df = pd.DataFrame({
            "Timestamp": df["DATE_TIME"].values,
            "Actual_DC_Power": y.values,
            "Predicted_DC_Power": preds,
            "Absolute_Error": errors,
            "Is_Anomaly": anomaly_mask,
        })

        csv_buffer = export_df.to_csv(index=False)
        st.download_button(
            label="Download Predictions CSV",
            data=csv_buffer,
            file_name="solar_forecast_predictions.csv",
            mime="text/csv",
        )

        # Store data for Tab 2
        st.session_state["forecast_data"] = {
            "predictions": preds.tolist(),
            "irradiation": df["IRRADIATION"].tolist(),
            "timestamps": df["DATE_TIME"].dt.strftime("%Y-%m-%d %H:%M").tolist(),
        }
        st.session_state["forecast_metrics"] = {
            "mae": round(mae, 2),
            "rmse": round(rmse, 2),
            "r2": round(r2, 4),
            "anomaly_count": anomaly_count,
            "total_points": len(preds),
        }
        st.success("Forecast complete! Switch to Grid Optimization tab for analysis.")

# ── Tab 2: Grid Optimization Assistant ───────────────────────────────────────

with tab2:
    st.header("Agentic Grid Optimization Assistant")

    if "forecast_data" not in st.session_state:
        st.info("Upload CSVs and run forecast in Tab 1 first.")
    else:
        # ── Run Analysis button ──────────────────────────────────────────
        col_btn, col_status = st.columns([1, 3])
        with col_btn:
            run_clicked = st.button("Run Analysis", type="primary")
        with col_status:
            if "agent_result" in st.session_state:
                st.caption("Analysis available. Re-run to refresh.")

        if run_clicked:
            with st.spinner("Running agentic analysis pipeline..."):
                try:
                    result = agent.invoke({
                        "predictions": st.session_state["forecast_data"]["predictions"],
                        "irradiation": st.session_state["forecast_data"]["irradiation"],
                        "timestamps": st.session_state["forecast_data"]["timestamps"],
                        "analysis": {},
                        "retrieved_docs": [],
                        "recommendations": {},
                        "error": None,
                    })
                except Exception as e:
                    st.error(f"Agent pipeline failed: {e}")
                    result = {"analysis": {}, "recommendations": {}, "error": str(e)}

                st.session_state["agent_result"] = result
                st.session_state["chat_history"] = []  # reset chat on new analysis

        # ── Display results from session state ───────────────────────────
        result = st.session_state.get("agent_result")
        if result:
            analysis = result.get("analysis", {})
            recs = result.get("recommendations", {})

            if analysis:
                st.divider()
                st.subheader("Forecast Analysis")

                risk = analysis.get("risk_level", "Low")
                risk_css = {"High": "risk-high", "Medium": "risk-medium"}.get(risk, "risk-low")

                col1, col2, col3 = st.columns(3)
                col1.metric("Mean Power", f"{analysis['mean_power']:,.0f}")
                col2.metric("Variability", f"{analysis['variability_pct']:.1f}%")
                col3.metric("Mean Irradiation", f"{analysis['mean_irradiation']:.4f}")

                st.markdown(
                    f'<div class="{risk_css}" style="text-align:center; margin: 12px 0;">'
                    f'<span style="font-size:14px; opacity:0.8;">Risk Level</span><br>'
                    f'<span style="font-size:28px; font-weight:bold;">{risk}</span></div>',
                    unsafe_allow_html=True,
                )

                col5, col6, col7 = st.columns(3)
                col5.metric("Max Power", f"{analysis.get('max_power', 0):,.0f}")
                col6.metric("Min Power", f"{analysis.get('min_power', 0):,.0f}")
                col7.metric("Std Dev", f"{analysis.get('std_power', 0):,.0f}")

            if result.get("error"):
                st.error(result["error"])
                st.info("Analysis stats above are still valid. LLM recommendations unavailable.")
            elif recs:
                st.divider()
                st.subheader("Recommendations")

                # Forecast Summary
                summary = recs.get("forecast_summary", "N/A")
                if isinstance(summary, dict):
                    summary = json.dumps(summary)
                st.markdown(
                    f'<div class="rec-card"><h4>Forecast Summary</h4>'
                    f'<p>{summary}</p></div>',
                    unsafe_allow_html=True,
                )

                # Risk Analysis
                risk_text = recs.get("risk_analysis", "N/A")
                if isinstance(risk_text, dict):
                    risk_text = json.dumps(risk_text)
                st.markdown(
                    f'<div class="rec-card-warn"><h4>Risk Analysis</h4>'
                    f'<p>{risk_text}</p></div>',
                    unsafe_allow_html=True,
                )

                # Immediate Actions
                actions = recs.get("actions", [])
                if actions:
                    st.markdown("**Immediate Actions**")
                    for i, action in enumerate(actions, 1):
                        if isinstance(action, dict):
                            action = action.get("action", action.get("description", str(action)))
                        st.markdown(
                            f'<div class="action-item">'
                            f'<span class="action-num">{i}</span>'
                            f'<span>{action}</span></div>',
                            unsafe_allow_html=True,
                        )

                # Optimization Strategies
                strategies = recs.get("optimization_strategies", [])
                if strategies:
                    st.markdown("**Optimization Strategies**")
                    for strategy in strategies:
                        if isinstance(strategy, dict):
                            strategy = strategy.get("strategy", strategy.get("description", str(strategy)))
                        st.markdown(
                            f'<div class="strategy-item">{strategy}</div>',
                            unsafe_allow_html=True,
                        )

                # References
                refs = recs.get("references", [])
                if refs:
                    st.markdown("**References**")
                    tags_html = "".join(
                        f'<span class="ref-tag">{r if isinstance(r, str) else str(r)}</span>'
                        for r in refs
                    )
                    st.markdown(f'<div style="margin-top:4px">{tags_html}</div>', unsafe_allow_html=True)

            # ── PDF report export ────────────────────────────────────────
            st.divider()
            if st.button("Generate Report (PDF)", key="pdf_btn"):
                with st.spinner("Generating report..."):
                    pdf_bytes = _generate_pdf(analysis, recs)
                    st.download_button(
                        label="Download Report",
                        data=pdf_bytes,
                        file_name="solar_grid_report.pdf",
                        mime="application/pdf",
                        key="pdf_download",
                    )

            # ── Chat interface ───────────────────────────────────────────
            st.divider()
            st.subheader("Ask Follow-up Questions")

            # Init chat history
            if "chat_history" not in st.session_state:
                st.session_state["chat_history"] = []

            # Chat container with all history
            chat_container = st.container(height=400)
            with chat_container:
                if not st.session_state["chat_history"]:
                    st.markdown(
                        '<div style="text-align:center; padding:40px 20px; opacity:0.5;">'
                        '<p style="font-size:16px;">No messages yet</p>'
                        '<p style="font-size:13px;">Ask about risk periods, optimization strategies, '
                        'or anything related to the analysis above.</p></div>',
                        unsafe_allow_html=True,
                    )
                else:
                    for msg in st.session_state["chat_history"]:
                        with st.chat_message(msg["role"]):
                            st.markdown(msg["content"])

            # Chat input below container
            user_question = st.chat_input(
                "e.g. Why is the risk level high?",
                key="chat_input",
            )
            if user_question:
                # Append user message
                st.session_state["chat_history"].append({"role": "user", "content": user_question})

                # Generate response
                try:
                    answer = chat_with_context(
                        question=user_question,
                        analysis=analysis,
                        recommendations=recs,
                        retrieved_docs=result.get("retrieved_docs", []),
                    )
                except Exception as e:
                    answer = f"Could not generate response: {e}"

                st.session_state["chat_history"].append({"role": "assistant", "content": answer})
                st.rerun()
