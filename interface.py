
import gradio as gr

# ============================
# Import Project Modules
# ============================
from module1_claim_extraction import extract_claims
from module2_claim_simplification import simplify_claims
from module3_debatability_detection import classify_debatability

# ============================
# Pipeline Processing Function
# ============================
def process_text(paragraph: str):
    """
    Full pipeline execution:
    1. Claim Extraction
    2. Claim Simplification
    3. Debatability Classification

    Returns formatted outputs for Gradio display.
    """

    if not paragraph or not paragraph.strip():
        return "No input provided.", "No input provided.", "No input provided."

    # ============================
    # Step 1: Extract Claims
    # ============================
    claims_list = extract_claims(paragraph)

    if not claims_list:
        return "No claims extracted.", "No claims extracted.", "No claims extracted."

    # ============================
    # Step 2: Simplify Claims
    # ============================
    simplified_list = simplify_claims(claims_list)

    # ============================
    # Step 3: Debatability Classification
    # NOTE: As per requirement, classification is done on claims_list
    # ============================
    debatability_results = classify_debatability(claims_list)

    # ============================
    # Formatting Outputs
    # ============================

    # ---- Extracted Claims Table ----
    extracted_text = ""
    for item in claims_list:
        extracted_text += f"[{item['claim_id']}] {item['claim']}\n\n"

    # ---- Simplified Claims Table ----
    simplified_text = ""
    for item in simplified_list:
        simplified_text += (
            f"[{item['claim_id']}]\n"
            f"Original: {item['original_claim']}\n"
            f"Simplified: {item['simplified_claim']}\n\n"
        )

    # ---- Debatability Results Table ----
    debatability_text = ""
    for item in debatability_results:
        debatability_text += (
            f"[{item['claim_id']}]\n"
            f"Claim: {item['claim']}\n"
            f"Label: {item['label']}\n\n"
        )

    return extracted_text.strip(), simplified_text.strip(), debatability_text.strip()

# ============================
# Gradio Interface Definition
# ============================
with gr.Blocks(title="Debate-Based Claim Analysis System") as demo:
    gr.Markdown(
        """
        # 🧠 Debate-Based Claim Analysis System

        This interface performs:
        1. **Claim Extraction**
        2. **Claim Simplification with Entity Definitions**
        3. **Debatability Classification**

        Enter a paragraph to analyze its factual claims.
        """
    )

    input_text = gr.Textbox(
        label="Input Paragraph",
        placeholder="Enter a paragraph containing factual statements...",
        lines=8
    )

    run_button = gr.Button("Analyze Text")

    extracted_output = gr.Textbox(
        label="Extracted Claims",
        lines=10
    )

    simplified_output = gr.Textbox(
        label="Simplified Claims (with Wikipedia Definitions)",
        lines=12
    )

    debatability_output = gr.Textbox(
        label="Debatability Classification",
        lines=10
    )

    run_button.click(
        fn=process_text,
        inputs=input_text,
        outputs=[
            extracted_output,
            simplified_output,
            debatability_output
        ]
    )

# ============================
# Launch Interface
# ============================
if __name__ == "__main__":
    demo.launch(share=True)
