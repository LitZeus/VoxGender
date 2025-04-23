import torch
import gradio as gr
import os
from model import ECAPA_gender

# Load the model
model = ECAPA_gender.from_pretrained("JaesungHuh/voice-gender-classifier")
model.eval()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def predict_gender(audio_file):
    """
    Predict gender from an audio file

    Args:
        audio_file: Path to audio file or audio data from microphone

    Returns:
        Dictionary with gender prediction and confidence
    """
    if audio_file is None:
        return {"error": "Please upload or record audio"}

    # If audio is from microphone recording, it will be a tuple of (sample_rate, audio_data)
    if isinstance(audio_file, tuple):
        import tempfile
        import soundfile as sf

        sr, audio_data = audio_file
        # Create a temporary file
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, "recording.wav")

        # Save the audio data to the temporary file
        sf.write(temp_path, audio_data, sr)
        audio_file = temp_path

    try:
        # Predict gender
        with torch.no_grad():
            gender = model.predict(audio_file, device=device)

        # Get confidence scores
        audio = model.load_audio(audio_file)
        audio = audio.to(device)

        with torch.no_grad():
            output = model.forward(audio)
            probabilities = torch.nn.functional.softmax(output, dim=1)[0]

        confidence = {
            "male": float(probabilities[0]),
            "female": float(probabilities[1])
        }

        # Format the result
        result = {
            "gender": gender.upper(),
            "confidence": f"{confidence[gender]:.2%}",
            "male_confidence": f"{confidence['male']:.2%}",
            "female_confidence": f"{confidence['female']:.2%}",
            "male_conf_value": confidence['male'],
            "female_conf_value": confidence['female']
        }

        return result
    except Exception as e:
        return {"error": f"Error processing audio: {str(e)}"}

# Custom CSS for styling
css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

body {
    font-family: 'Inter', sans-serif;
}

.container {
    max-width: 900px;
    margin: 0 auto;
}

.gradio-container {
    font-family: 'Inter', sans-serif !important;
}

.gr-prose h1 {
    font-weight: 700;
    font-size: 2.5rem;
    margin-bottom: 0.5rem;
    background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    display: inline-block;
}

.gr-prose h3 {
    font-weight: 600;
    margin-top: 1rem;
    margin-bottom: 0.5rem;
}

.gr-prose p {
    margin-bottom: 1.5rem;
}

.gr-button {
    background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%) !important;
}

.gr-form {
    background-color: #f8f9fa;
    border-radius: 10px;
    padding: 1.5rem;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    margin-bottom: 1.5rem;
}

.gr-input, .gr-select {
    border-radius: 8px !important;
}

.footer {
    text-align: center;
    margin-top: 2rem;
    font-size: 0.9rem;
    color: #666;
}

.result-container {
    background-color: white;
    border-radius: 10px;
    padding: 1.5rem;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    margin-top: 1rem;
    color: #333;
    line-height: 1.6;
}

.gender-male {
    color: #3498db;
    font-weight: bold;
}

.gender-female {
    color: #e74c3c;
    font-weight: bold;
}

.confidence-bar-container {
    width: 100%;
    height: 24px;
    background-color: #f0f0f0;
    border-radius: 12px;
    margin-bottom: 1rem;
    position: relative;
    overflow: hidden;
}

.confidence-bar {
    height: 100%;
    border-radius: 12px;
    transition: width 0.5s ease;
    display: flex;
    align-items: center;
    justify-content: flex-end;
    padding-right: 10px;
    color: white;
    font-weight: 600;
    font-size: 0.9rem;
    text-shadow: 0 1px 2px rgba(0, 0, 0, 0.5);
}

.male-bar {
    background: linear-gradient(90deg, #3498db 0%, #2980b9 100%);
}

.female-bar {
    background: linear-gradient(90deg, #e74c3c 0%, #c0392b 100%);
}
"""

def format_result(result):
    """Format the prediction result as HTML"""
    if "error" in result:
        return f"""<div class="result-container" style="background-color: #f8f9fa;">
            <h3 style="color: #e74c3c; margin-bottom: 15px; background-color: white; padding: 10px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">Error</h3>
            <div style="background-color: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                <p style="color: #333;">{result["error"]}</p>
            </div>
        </div>"""

    gender_class = "gender-male" if result["gender"] == "MALE" else "gender-female"

    # Use the raw confidence values for the bars
    male_conf = result["male_conf_value"]
    female_conf = result["female_conf_value"]

    male_width = male_conf * 100
    female_width = female_conf * 100

    return f"""<div class="result-container" style="background-color: #f8f9fa;">
        <h3 style="color: #182848; margin-bottom: 15px; background-color: white; padding: 10px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">Prediction Result</h3>

        <div style="background-color: white; padding: 15px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
            <p style="font-size: 1.1rem; color: #333;"><span style="color: #666; font-weight: 500;">Gender:</span> <span class="{gender_class}" style="font-size: 1.3rem;">{result["gender"]}</span></p>
            <p style="color: #333;"><span style="color: #666; font-weight: 500;">Confidence:</span> <strong>{result["confidence"]}</strong></p>
        </div>

        <h3 style="color: #182848; margin-bottom: 15px; margin-top: 25px; background-color: white; padding: 10px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">Confidence Scores</h3>

        <div style="background-color: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
            <p style="margin-bottom: 8px; font-weight: 500; color: #2980b9;">Male:</p>
            <div class="confidence-bar-container">
                <div class="confidence-bar male-bar" style="width: {male_width}%">
                    {result["male_confidence"]}
                </div>
            </div>

            <p style="margin-bottom: 8px; margin-top: 15px; font-weight: 500; color: #c0392b;">Female:</p>
            <div class="confidence-bar-container">
                <div class="confidence-bar female-bar" style="width: {female_width}%">
                    {result["female_confidence"]}
                </div>
            </div>
        </div>
    </div>"""

# Create Gradio interface
with gr.Blocks(css=css, title="Voice Gender Classifier") as demo:
    gr.Markdown(
        """
        # Voice Gender Classifier

        Upload an audio file or record your voice to predict gender.
        """
    )

    with gr.Row():
        with gr.Column():
            # Audio input
            audio_input = gr.Audio(
                label="Upload or Record Audio",
                type="filepath",
                sources=["upload", "microphone"]
            )

            # Submit button
            submit_btn = gr.Button("Predict Gender")

        with gr.Column():
            # Output display
            output = gr.HTML(
                label="Result",
                value="""<div class="result-container" style="background-color: #f8f9fa;">
                    <div style="background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); text-align: center;">
                        <p style="color: #666;">Upload or record audio to see prediction results</p>
                    </div>
                </div>"""
            )

    # Set up the prediction function
    submit_btn.click(
        fn=lambda x: format_result(predict_gender(x)),
        inputs=audio_input,
        outputs=output
    )

    # Examples
    if os.path.exists("data/00001.wav"):
        gr.Examples(
            examples=["data/00001.wav"],
            inputs=audio_input,
            outputs=output,
            fn=lambda x: format_result(predict_gender(x)),
            cache_examples=True,
        )

    # Add sample from our static folder if it exists
    if os.path.exists("static/samples/sample.wav"):
        gr.Examples(
            examples=["static/samples/sample.wav"],
            inputs=audio_input,
            outputs=output,
            fn=lambda x: format_result(predict_gender(x)),
            cache_examples=True,
        )

    gr.Markdown(
        """
        <div class="footer">
            Powered by ECAPA-TDNN model
        </div>
        """
    )

# Launch the app
if __name__ == "__main__":
    demo.launch()
