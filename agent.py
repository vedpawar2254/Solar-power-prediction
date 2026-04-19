"""LangGraph agentic workflow: 3-node linear pipeline for grid optimization."""

import os
import re
import json
from typing import Optional, TypedDict

import numpy as np
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq

from rag import retrieve


class AgentState(TypedDict):
    predictions: list[float]
    irradiation: list[float]
    timestamps: list[str]
    analysis: dict
    retrieved_docs: list[str]
    recommendations: dict
    error: Optional[str]


# ── Node 1: Pure-Python forecast analysis ────────────────────────────────────

def analyze_forecast(state: AgentState) -> dict:
    preds = np.array(state["predictions"])
    irrad = np.array(state["irradiation"])
    timestamps = state["timestamps"]

    mean_pred = float(np.mean(preds))
    std_pred = float(np.std(preds))
    variability_pct = (std_pred / mean_pred * 100) if mean_pred > 0 else 0.0

    # Risk level
    if variability_pct > 40:
        risk = "High"
    elif variability_pct > 20:
        risk = "Medium"
    else:
        risk = "Low"

    # Identify risk periods: timestamps where prediction drops > 30% from mean
    threshold = mean_pred * 0.7
    risk_periods = [
        timestamps[i] for i, p in enumerate(preds) if p < threshold
    ]

    # Peak hours (top 10% of predictions)
    top_k = max(1, len(preds) // 10)
    peak_indices = np.argsort(preds)[-top_k:]
    peak_hours = [timestamps[i] for i in peak_indices]

    analysis = {
        "mean_power": round(mean_pred, 2),
        "max_power": round(float(np.max(preds)), 2),
        "min_power": round(float(np.min(preds)), 2),
        "std_power": round(std_pred, 2),
        "variability_pct": round(variability_pct, 2),
        "risk_level": risk,
        "risk_periods": risk_periods[:10],  # cap at 10
        "peak_hours": peak_hours[:10],
        "mean_irradiation": round(float(np.mean(irrad)), 4),
    }
    return {"analysis": analysis}


# ── Node 2: RAG retrieval ────────────────────────────────────────────────────

def retrieve_guidelines(state: AgentState) -> dict:
    analysis = state["analysis"]
    query = (
        f"Solar grid management for {analysis['risk_level']} risk, "
        f"{analysis['variability_pct']}% variability, "
        f"mean irradiation {analysis['mean_irradiation']}"
    )
    docs = retrieve(query, k=4)
    return {"retrieved_docs": docs}


# ── Node 3: LLM recommendation generation ───────────────────────────────────

def generate_recommendations(state: AgentState) -> dict:
    analysis = state["analysis"]
    docs = state["retrieved_docs"]

    prompt = f"""You are a grid optimization expert. Based on the solar forecast analysis
and grid management guidelines below, provide structured recommendations.

## Forecast Analysis
{json.dumps(analysis, indent=2)}

## Relevant Grid Management Guidelines
{chr(10).join(f"- {d}" for d in docs)}

Respond with ONLY valid JSON (no markdown, no code fences) with these 5 keys:
- "forecast_summary": 2-3 sentence summary of forecast conditions
- "risk_analysis": 2-3 sentence risk assessment
- "actions": list of 3-5 immediate action strings
- "optimization_strategies": list of 3-5 strategy strings
- "references": list of guideline topics used
"""

    try:
        api_key = os.environ.get("GROQ_API_KEY", "")
        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            api_key=api_key,
            temperature=0.3,
        )
        response = llm.invoke(prompt)
        content = response.content.strip()

        # Strip markdown code fences if present
        content = re.sub(r'^```\w*\n?', '', content, count=1)
        content = re.sub(r'\n?```\s*$', '', content)
        content = content.strip()

        recommendations = json.loads(content)
        return {"recommendations": recommendations}

    except Exception as e:
        return {
            "error": f"LLM generation failed: {str(e)}",
            "recommendations": {},
        }


# ── Build graph ──────────────────────────────────────────────────────────────

def build_agent():
    graph = StateGraph(AgentState)

    graph.add_node("analyze_forecast", analyze_forecast)
    graph.add_node("retrieve_guidelines", retrieve_guidelines)
    graph.add_node("generate_recommendations", generate_recommendations)

    graph.set_entry_point("analyze_forecast")
    graph.add_edge("analyze_forecast", "retrieve_guidelines")
    graph.add_edge("retrieve_guidelines", "generate_recommendations")
    graph.add_edge("generate_recommendations", END)

    return graph.compile()


agent = build_agent()


# ── Chat follow-up function ──────────────────────────────────────────────────

def chat_with_context(
    question: str,
    analysis: dict,
    recommendations: dict,
    retrieved_docs: list[str],
) -> str:
    """Answer follow-up questions using analysis context + RAG docs."""
    context_parts = []
    if analysis:
        context_parts.append(f"## Forecast Analysis\n{json.dumps(analysis, indent=2)}")
    if recommendations:
        context_parts.append(f"## Current Recommendations\n{json.dumps(recommendations, indent=2)}")
    if retrieved_docs:
        context_parts.append("## Grid Management Guidelines\n" + "\n".join(f"- {d}" for d in retrieved_docs))

    # Retrieve additional docs relevant to the question
    extra_docs = retrieve(question, k=2)
    if extra_docs:
        context_parts.append("## Additional Guidelines\n" + "\n".join(f"- {d}" for d in extra_docs))

    context = "\n\n".join(context_parts)

    prompt = f"""You are a solar grid optimization expert. Use the context below to answer the user's question.
Be specific and reference data from the analysis when relevant. Keep answers concise.

{context}

User question: {question}"""

    api_key = os.environ.get("GROQ_API_KEY", "")
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=api_key,
        temperature=0.3,
    )
    response = llm.invoke(prompt)
    return response.content.strip()
