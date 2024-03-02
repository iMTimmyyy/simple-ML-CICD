import gradio as gr
import skops.io as skio

clf = skio.load("./Model/drug_classifier.skops", trusted=True)

def predict(age, sex, bp, cholesterol, na_to_k):
    """
    Predict drug type based on patient features

    Args:
        age (int): Age of patient
        sex (str): Sex of patient
        bp (str): Blood pressure level
        cholesterol (str): Cholesterol level
        na_to_k (float): Na-to-K ratio of patient

    Returns:
        str: Predicted drug type
    """
    features = [age, sex, bp, cholesterol, na_to_k]
    prediction = clf.predict([features])[0]

    label = f"Predicted drug type: {prediction}"
    return label

inputs = [
    gr.Slider(15, 75, step=1, label="Age"),
    gr.Radio(["F", "M"], label="Sex"),
    gr.Radio(["LOW", "NORMAL", "HIGH"], label="BP"),
    gr.Radio(["NORMAL", "HIGH"], label="Cholesterol"),
    gr.Slider(6.0, 40, step=0.1, label="Na_to_K"),
]
outputs = [gr.Label(num_top_classes=5)]

examples = [
    [45, "F", "LOW", "NORMAL", 9.7],
    [25, "M", "HIGH", "HIGH", 8.7],
    [35, "F", "LOW", "HIGH", 7.2],
]

title = "Drug Classification"
description = "Enter the patient's features to predict their drug type."
article = "This app is part of a simple CI/CD pipeline using GitHub Actions only."

demo = gr.Interface(
    fn=predict,
    inputs=inputs,
    outputs=outputs,
    examples=examples,
    title=title,
    description=description,
    article=article,
    theme=gr.themes.Soft(),
)
demo.launch()