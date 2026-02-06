import pandas as pd
import re
from nodes import analyst_node, research_node


def check_answer(expected_fact: str, answer: str):
    """
    Check if the expected fact is present in the answer.
    Uses flexible matching: case-insensitive and extracts key numeric values.

    """
    expected_lower = expected_fact.lower()
    answer_lower = answer.lower()
    
    # Direct substring match (case-insensitive)
    if expected_lower in answer_lower:
        return True
    
    # Extract numeric values from expected fact (e.g., "13.8", "921.9", "24.8")
    expected_numbers = re.findall(r'\d+\.?\d*', expected_fact)
    
    # Check if all expected numbers appear in the answer
    if expected_numbers:
        all_numbers_found = all(num in answer for num in expected_numbers)
        if all_numbers_found:
            return True
    
    # Handle currency amounts (e.g., "Rp921.9 trillion" -> check for "921.9" and "trillion")
    if 'rp' in expected_lower or 'trillion' in expected_lower or 'billion' in expected_lower:
        key_terms = re.findall(r'[\d.]+|trillion|billion', expected_lower)
        if key_terms and all(term in answer_lower for term in key_terms):
            return True
    
    # Handle percentages (e.g., "13.8% YoY" -> check for "13.8" and "%")
    if '%' in expected_fact:
        percentage_match = re.search(r'([\d.]+)\s*%', expected_fact)
        if percentage_match:
            pct_value = percentage_match.group(1)
            if pct_value in answer and '%' in answer:
                return True
    
    return False


# Define 'Ground Truth' test cases based on the BCA 2024 Report
test_dataset = [
    {
        "question": "What was BCA's total loan portfolio in 2024?",
        "expected_fact": "Rp921.9 trillion"
    },
    {
        "question": "What was BCA's year-over-year loan growth rate in 2024?",
        "expected_fact": "13.8% YoY"
    },
    {
        "question": "What was BCA's sustainable financing portfolio in 2024?",
        "expected_fact": "Rp229 trillion"
    },
    {
        "question": "What percentage of total loans was sustainable financing?",
        "expected_fact": "24.8%"
    },
    {
        "question": "What was the year-over-year growth rate of BCA's sustainable financing in 2024?",
        "expected_fact": "12.5% YoY"
    },
]

def run_evaluation():
    results = []
    
    for test in test_dataset:
        state = {"question": test["question"]}
        state.update(research_node(state))
        state.update(analyst_node(state))
        
        is_correct = check_answer(test["expected_fact"], state["answer"])
        
        results.append({
            "Question": test["question"],
            "Correct": is_correct,
            "Agent_Response": state["answer"][:100] + "..."
        })
        
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.expand_frame_repr', False)
    df = pd.DataFrame(results)
    print("\n--- LEDGERLENS PERFORMANCE REPORT ---")
    print(df.to_string())
    return df

if __name__ == "__main__":
    run_evaluation()