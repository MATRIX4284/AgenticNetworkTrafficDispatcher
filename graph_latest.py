from load_complaints import load_complaints
#from model import GraphState
from typing import TypedDict, List, Dict
import pandas as pd
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from topic_classification import classify_complaint
from complaint_summarizer import SUMMARY_PROMPT
from typing import TypedDict, List, Dict, Optional

class State(TypedDict):
    csv_path: str
    rows: List[Dict]              
    tool_called: bool
    priority: Optional[str]


llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0
)


def summarize_node(state: State):
    summarized_rows = []

    for row in state["rows"]:
        response = llm.invoke(
            SUMMARY_PROMPT.format(complaint=row["complaint"])
        )

        summarized_rows.append({
            "summary": response.content.strip(),
            "ticket": row["ticket"],
            "state": row["state"],
            "initial_department": row["initial_department"],
            "priority": row["priority"]
        })

    return {"rows": summarized_rows}


def process(state: State):
    df = pd.read_csv(state["csv_path"])

    required_columns = ["Ticket #","Customer Complaint", "State"]
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    rows = (
        df[required_columns]
        .apply(
            lambda row: {
                "ticket": str(row["Ticket #"]),
                "complaint": str(row["Customer Complaint"]),
                "state": str(row["State"])
            },
            axis=1
        )
        .tolist()
    )

    return {"rows": rows}

from vectorstore import get_vectorstore
vectorstore = get_vectorstore()

# ---- Node 2: Classify ----
def classify_node(state: State):
    classified = []

    for row in state["rows"]:
        result = classify_complaint(
            complaint=row["complaint"],
            state=row["state"],
            vectorstore=vectorstore,
            llm=llm
        )

        classified.append({
            "ticket": row["ticket"],
            "complaint": row["complaint"],  # needed for summary
            "state": row["state"],
            "initial_department": result["initial_department"],
            "all_departments": result["all_departments"],
            "priority": result["priority"]
        })

    return {"rows": classified}


import uuid

def finalize_ticket_node(state: State):
    final_rows = []

    for row in state["rows"]:
        final_rows.append({
            "ticket_id": row["ticket"], 
            "department": row["initial_department"],
            "priority": row["priority"],
            "summary": row["summary"]
        })

    return {"rows": final_rows}

# ---- Graph Definition ----
graph = StateGraph(State)

graph.add_node("process", process)
graph.add_node("classify", classify_node)
graph.add_node("summarize", summarize_node)
graph.add_node("finalize", finalize_ticket_node)

graph.set_entry_point("process")
graph.add_edge("process", "classify")
graph.add_edge("classify", "summarize")
graph.add_edge("summarize", "finalize")
graph.add_edge("finalize", END)

# ---- Compile ----
app = graph.compile()

# ---- Invoke ----
result = app.invoke({
    "csv_path": "Comcast.csv",
    "rows": []
})

print(result)