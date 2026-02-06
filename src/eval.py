import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    AnswerRelevancy,
    Faithfulness,
    ContextRecall,
    ContextPrecision,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from nodes import analyst_node, research_node

# Configure RAGAS to use OpenAI models explicitly
ragas_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini", temperature=0))
ragas_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())

# Instantiate metrics with LLM and embeddings
answer_relevancy = AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings)
faithfulness = Faithfulness(llm=ragas_llm)
context_recall = ContextRecall(llm=ragas_llm)
context_precision = ContextPrecision(llm=ragas_llm)

# Define 'Ground Truth' test cases based on the BCA 2024 Report
test_dataset = [
    {
        "question": "What was BCA's total loan portfolio in 2024?",
        "ground_truth": "BCA's total loan portfolio in 2024 was Rp921.9 trillion, representing a 13.8% year-over-year growth."
    },
    {
        "question": "What was BCA's year-over-year loan growth rate in 2024?",
        "ground_truth": "BCA's year-over-year loan growth rate in 2024 was 13.8%."
    },
    {
        "question": "What was BCA's sustainable financing portfolio in 2024?",
        "ground_truth": "BCA's sustainable financing portfolio in 2024 was Rp229 trillion, which represented 24.8% of total loans."
    },
    {
        "question": "What percentage of total loans was sustainable financing?",
        "ground_truth": "Sustainable financing represented 24.8% of BCA's total loans in 2024."
    },
    {
        "question": "What was the year-over-year growth rate of BCA's sustainable financing in 2024?",
        "ground_truth": "BCA's sustainable financing grew by 12.5% year-over-year in 2024."
    },
]


def run_evaluation():
    """
    Run RAGAS evaluation on the LedgerLens agent.
    
    RAGAS Metrics:
    - Answer Relevancy: How relevant is the answer to the question?
    - Faithfulness: Is the answer grounded in the retrieved context?
    - Context Recall: How much of the ground truth is captured in the context?
    - Context Precision: Are the relevant contexts ranked higher?
    """
    results = []
    
    print("\n--- Running LedgerLens Agent on Test Cases ---")
    for test in test_dataset:
        state = {"question": test["question"]}
        state.update(research_node(state))
        state.update(analyst_node(state))
        
        results.append({
            "question": test["question"],
            "answer": state["answer"],
            "contexts": state["contexts"],  # RAGAS expects a list of contexts
            "ground_truth": test["ground_truth"]
        })
        
        print(f"✓ Processed: {test['question'][:60]}...")
    
    # Convert to RAGAS-compatible dataset
    dataset = Dataset.from_list(results)
    
    print("\n--- Evaluating with RAGAS Metrics ---")
    evaluation_result = evaluate(
        dataset,
        metrics=[
            answer_relevancy,
            faithfulness,
            context_recall,
            context_precision,
        ],
    )
    
    # Display results
    df = evaluation_result.to_pandas()
    
    pd.set_option('display.max_colwidth', 50)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.expand_frame_repr', False)
    pd.set_option('display.precision', 3)
    
    print("\n" + "="*80)
    print("LEDGERLENS RAGAS EVALUATION REPORT")
    print("="*80)
    print("\nOverall Scores:")
    print("-" * 80)
    
    # Show aggregate metrics
    metrics_summary = {
        "Answer Relevancy": df["answer_relevancy"].mean(),
        "Faithfulness": df["faithfulness"].mean(),
        "Context Recall": df["context_recall"].mean(),
        "Context Precision": df["context_precision"].mean(),
    }
    
    for metric, score in metrics_summary.items():
        print(f"{metric:.<30} {score:.3f}")
    
    print("\n" + "="*80)
    print("Detailed Results:")
    print("="*80)

    question_col = "user_input" if "user_input" in df.columns else "question"
    metric_cols = ["answer_relevancy", "faithfulness", "context_recall", "context_precision"]
    display_cols = [question_col] + [c for c in metric_cols if c in df.columns]
    print(df[display_cols].to_string(index=False))
    
    print("\n")
    
    return evaluation_result, df


if __name__ == "__main__":
    run_evaluation()