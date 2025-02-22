import base64
import streamlit as st

def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

def generate_ai_explanation(client, risk_level, delay_days, input_data):
    """Generates an AI-based explanation using Together AI (LLaMA 2-70B)"""
    prompt = f"""
    A metro project has the following characteristics:
    - City: {input_data['City'].values[0]}
    - Tunnel Length: {input_data['Tunnel Length (km)'].values[0]} km
    - Number of Stations: {input_data['Num of Stations'].values[0]}
    - Number of Workers: {input_data['Num of Workers'].values[0]}
    - Equipment Factor: {input_data['Equipment Factor (%)'].values[0]}%
    - Signal System Complexity: {input_data['Signal System Complexity (%)'].values[0]}%
    - Material Factor: {input_data['Material Factor (%)'].values[0]}%
    - Government Regulation Factor: {input_data['Gov Regulation Factor (%)'].values[0]}%
    - Urban Congestion Impact: {input_data['Urban Congestion Impact (%)'].values[0]}%
    - Expected Completion Time: {input_data['Expected Completion Time (months)'].values[0]} months
    - Budget: ₹{input_data['Budget (in Crores)'].values[0]} crores

    The project has been classified as having a **{risk_level}** risk level and is expected to face a delay of **{delay_days:.2f}** days.

    Explain the results in simple terms, highlighting the key factors contributing to the risk level and delay. Also, suggest mitigation strategies.
    """
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    st.write(response.choices[0].message.content)

def generate_detection_explanation(client, detected_objects, detection_type):
    """Generates an AI-based explanation for detected objects using Together AI (LLaMA 2-70B)"""
    prompt = f"Explain the following detected objects in simple terms for {detection_type}:\n\n"
    for obj in detected_objects:
        prompt += f"- Class: {obj['class_name']}, Confidence: {obj['confidence']:.2f}, Bounding Box: {obj['bbox']}\n"
    if detection_type == "Blueprint Detection":
        prompt += "\n Basically purpose of Blueprint detection is to help builders save time, cost and delays which happen due to problems in the blueprint itself. Explain what each term means, why it is important, and how it relates to blueprint analysis. Find faults in the blueprint based on the detection  or appreciate if something is right. Also, suggest any necessary actions or precautions."
    elif detection_type == "Risk Detection":
        prompt += "\nExplain what each term means, why it is important, and what exactly are the risks in the image. Also, suggest any necessary actions or precautions."
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content
