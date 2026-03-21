import gradio as gr
import time

# ============================
# Import Project Modules
# ============================
from module1_claim_extraction import extract_claims
from module2_claim_simplification import simplify_claims
from module3_debatability_detection import classify_debatability
from module4_webscraping import retrieve_evidence_chunks

# ✅ NEW MODULE 5
from module5_evidence_classification import filter_and_rank_evidence


# ============================
# Progressive Pipeline Function
# ============================
def process_text(paragraph: str):

    if not paragraph or not paragraph.strip():
        yield "No input provided.", "", "", "", ""
        return

    # =====================================================
    # STEP 1: Claim Extraction
    # =====================================================
    claims_list = extract_claims(paragraph)

    if not claims_list:
        yield "No claims extracted.", "", "", "", ""
        return

    extracted_text = ""
    for item in claims_list:
        extracted_text += f"[{item['claim_id']}] {item['claim']}\n\n"

    yield extracted_text.strip(), "", "", "", ""
    time.sleep(0.5)

    # =====================================================
    # STEP 2: Claim Simplification
    # =====================================================
    simplified_list = simplify_claims(claims_list)

    for c, s in zip(claims_list, simplified_list):
        c["simplified_claim"] = s["simplified_claim"]

    simplified_text = ""
    for item in simplified_list:
        simplified_text += (
            f"[{item['claim_id']}]\n"
            f"Original: {item['original_claim']}\n"
            f"Simplified: {item['simplified_claim']}\n\n"
        )

    yield extracted_text.strip(), simplified_text.strip(), "", "", ""
    time.sleep(0.5)

    # =====================================================
    # STEP 3: Debatability Classification
    # =====================================================
    debatability_results = classify_debatability(claims_list)

    debatability_text = ""
    for item in debatability_results:
        debatability_text += (
            f"[{item['claim_id']}]\n"
            f"Claim: {item['claim']}\n"
            f"Label: {item['label']}\n\n"
        )

    yield (
        extracted_text.strip(),
        simplified_text.strip(),
        debatability_text.strip(),
        "",
        ""
    )
    time.sleep(0.5)

    # =====================================================
    # STEP 4: Web Retrieval
    # =====================================================
    retrieved_results = retrieve_evidence_chunks(debatability_results)

    scraped_text = ""

    for item in retrieved_results:

        if item["label"] == "debatable":

            scraped_text += f"\n========== Claim {item['claim_id']} ==========\n"
            scraped_text += f"Claim: {item['claim']}\n\n"

            chunks = item.get("evidence_chunks", [])

            if not chunks:
                scraped_text += "No evidence retrieved.\n\n"
                continue

            for chunk in chunks:
                scraped_text += f"Source: {chunk.get('source')}\n"
                scraped_text += f"URL: {chunk.get('url')}\n"
                scraped_text += f"Content:\n{chunk.get('content')}\n"
                scraped_text += "-" * 80 + "\n\n"

    if not scraped_text.strip():
        scraped_text = "No web content retrieved."

    yield (
        extracted_text.strip(),
        simplified_text.strip(),
        debatability_text.strip(),
        scraped_text.strip(),
        ""
    )
    time.sleep(0.5)

    # =====================================================
    # STEP 5: FILTER + RANK (NEW 🔥)
    # =====================================================
    filtered_results = filter_and_rank_evidence(retrieved_results)

    filtered_text = ""

    for item in filtered_results:

        filtered_text += f"\n========== Claim {item['claim_id']} ==========\n"
        filtered_text += f"Claim: {item['claim']}\n\n"

        evidence = item.get("filtered_evidence", [])

        if not evidence:
            filtered_text += "No strong evidence found.\n\n"
            continue

        for e in evidence:
            filtered_text += f"- {e['content'][:250]}...\n"
            filtered_text += f"  Source: {e['source']}\n"
            filtered_text += f"  URL: {e['url']}\n\n"

        filtered_text += "=" * 80 + "\n"

    if not filtered_text.strip():
        filtered_text = "No filtered evidence available."

    yield (
        extracted_text.strip(),
        simplified_text.strip(),
        debatability_text.strip(),
        scraped_text.strip(),
        filtered_text.strip()
    )


# ============================
# Gradio Interface
# ============================
with gr.Blocks(title="Debate-Based Claim Analysis System") as demo:

    gr.Markdown(
        """
        # 🧠 Debate-Based Claim Analysis System

        This system performs:

        1️⃣ Claim Extraction  
        2️⃣ Claim Simplification  
        3️⃣ Debatability Classification  
        4️⃣ Web Retrieval  
        5️⃣ Evidence Filtering & Ranking ✅

        ⚠️ Final reasoning (pro/against) should be done using LLM (Module 6)
        """
    )

    input_text = gr.Textbox(
        label="Input Paragraph",
        lines=8,
        placeholder="Enter a paragraph..."
    )

    run_button = gr.Button("Analyze Text")

    extracted_output = gr.Textbox(label="Extracted Claims", lines=8)
    simplified_output = gr.Textbox(label="Simplified Claims", lines=10)
    debatability_output = gr.Textbox(label="Debatability Classification", lines=8)
    scraped_output = gr.Textbox(label="Retrieved Evidence", lines=20)
    filtered_output = gr.Textbox(label="Filtered Evidence (Top Ranked)", lines=20)

    run_button.click(
        fn=process_text,
        inputs=input_text,
        outputs=[
            extracted_output,
            simplified_output,
            debatability_output,
            scraped_output,
            filtered_output
        ]
    )


# ============================
# Launch
# ============================
if __name__ == "__main__":
    demo.launch(share=True)