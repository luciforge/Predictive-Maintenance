import gradio as gr
import requests, re
import base64
from PIL import Image
from io import BytesIO

# Backend endpoint
BACKEND_URL = "http://127.0.0.1:5000/predict"

def query_chatbot(user_input):
    try:
        unit_match = re.search(r'unit\s+(\d+)', user_input.lower())
        if not unit_match:
            return "❌ Could not detect a unit number in your input.", None
        unit_id = int(unit_match.group(1))

        cycle_match = re.search(r'cycle\s+(\d+)', user_input.lower())
        rul_mode = bool(re.search(r'rul|life|remaining', user_input.lower()))

        payload = {"unit_id": unit_id}
        if cycle_match:
            cycle = int(cycle_match.group(1))
            payload["cycle"] = cycle

        if rul_mode:
            endpoint = "http://127.0.0.1:5000/predict_rul"
        else:
            endpoint = "http://127.0.0.1:5000/predict"

        res = requests.post(endpoint, json=payload)
        res.raise_for_status()
        data = res.json()

    except Exception as e:
        return f"❌ Error: {e}", None

    # RUL mode
    if "predicted_rul" in data:
        cycle_info = data.get("cycle", "latest")
        return f"🧮 Unit {unit_id} at cycle {cycle_info} has approximately {data['predicted_rul']} cycles remaining.", None

    # Classification mode
    pred_class = data.get("predicted_class", "?")
    confidence = data.get("confidence_score", 0.0)
    cycle_info = data.get("cycle", "latest")
    response_text = f"✅ Unit {unit_id} at cycle {cycle_info} is classified as: Class {pred_class} (Confidence: {confidence:.2%})"

    shap_base64 = data.get("shap_plot", None)
    shap_img = None
    if shap_base64:
        shap_img = Image.open(BytesIO(base64.b64decode(shap_base64)))

    return response_text, shap_img

# Gradio Interface
demo = gr.Interface(
    fn=query_chatbot,
    inputs=gr.Textbox(lines=2, placeholder="Ask something like 'What is the status of unit 10?' or 'How is unit 4 at cycle 100?'"),
    outputs=[
        gr.Textbox(label="Chatbot Response"),
        gr.Image(type="pil", label="SHAP Explanation (if available)")
    ],
    title="Predictive Maintenance Chatbot",
    description="Ask about unit status, confidence of failure, or SHAP insight. Examples: 'Status of unit 7?', 'Status of unit 5 at cycle 90'"
)

if __name__ == "__main__":
    demo.launch()
