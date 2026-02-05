import pandas as pd
from nodes import analyst_node, research_node

# Define 'Ground Truth' test cases based on the BCA 2024 Report
test_dataset = [
    {
        "question": "What was BCA's total loan portfolio in 2024?",
        "expected_fact": "Rp921.9 trillion  
    },
    {
        "question": "What was BCA's year-over-year loan growth rate in 2024?",
        "expected_fact": "13.8%"
    },
    {
        "question": "What was BCA's green financing amount in 2024?",
        "expected_fact": "Rp229 trillion"
    },
    {
        "question": "What percentage of BCA's total loan portfolio consisted of green financing in 2024?",
        "expected_fact": "24.8%"
    },
    {
        "question": "What was the year-over-year growth rate of BCA's green financing in 2024?",
        "expected_fact": "12.5%"
    },
]

def run_evaluation():
    results = []
    
    for test in test_dataset:
        state = {"question": test["question"]}
        state.update(research_node(state))
        state.update(analyst_node(state))
        
        is_correct = test["expected_fact"] in state["answer"]
        
        results.append({
            "Question": test["question"],
            "Correct": is_correct,
            "Agent_Response": state["answer"][:100] + "..."
        })
    
    df = pd.DataFrame(results)
    print("\n--- LEDGERLENS PERFORMANCE REPORT ---")
    print(df)
    return df

if __name__ == "__main__":
    run_evaluation()