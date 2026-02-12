KEYWORD_PROMPT = """
Extract the key technical issue keywords from the complaint below.
Return ONLY a comma-separated list of keywords.
Do not explain.

Complaint:
{complaint}
"""


def detect_departments_from_sop(keywords, vectorstore):
    departments_found = set()
    sop_evidence = []

    for keyword in keywords:
        docs = vectorstore.similarity_search(keyword, k=2)

        for doc in docs:
            text = doc.page_content.lower()
            sop_evidence.append(doc.page_content)

            if any(k in text for k in ["outages", "data caps", "slow speeds", "signal drops"]):
                departments_found.add("NOC")

            if any(k in text for k in ["billing", "overcharges", "contract termination", "refunds"]):
                departments_found.add("Billing")

            if any(k in text for k in ["modems", "routers", "wiring", "installation"]):
                departments_found.add("Hardware")

    return sorted(departments_found), sop_evidence


def resolve_initial_department(departments: list[str]) -> str:
    """
    SOP precedence: NOC > Billing > Hardware
    """
    if "NOC" in departments:
        return "NOC"
    if "Billing" in departments:
        return "Billing"
    if "Hardware" in departments:
        return "Hardware"
    return "Unknown"

def extract_keywords_llm(complaint: str, llm) -> list[str]:
    response = llm.invoke(
        KEYWORD_PROMPT.format(complaint=complaint)
    )

    # Normalize keywords
    keywords = [
        k.strip().lower()
        for k in response.content.split(",")
        if k.strip()
    ]

    return keywords

from tools import call_validate_region_tool

def classify_complaint(
    complaint: str,
    state: str,
    vectorstore,
    llm
) -> dict:
    # 1. Extract keywords via LLM
    keywords = extract_keywords_llm(complaint, llm)

    # 2. Detect ALL departments from SOP
    all_departments, sop_evidence = detect_departments_from_sop(
        keywords, vectorstore
    )

    # 3. Resolve initial routing using precedence
    initial_department = resolve_initial_department(all_departments)

    # 4. Tool-based priority (MANDATORY)
    is_urgent = call_validate_region_tool(state)
    priority = "URGENT" if is_urgent else "NORMAL"

    return {
        "initial_department": initial_department,
        "all_departments": all_departments,
        "priority": priority,
        "keywords": keywords,
        "tool_call": f"validate_region('{state}') -> {is_urgent}",
        "sop_evidence": sop_evidence[:2]
    }