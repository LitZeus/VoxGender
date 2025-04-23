import torch
import os

from model import ECAPA_gender

def test_model():
    # You could directly download the model from the huggingface model hub
    model = ECAPA_gender.from_pretrained("JaesungHuh/voice-gender-classifier")
    model.eval()

    # If you are using gpu or not.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the example file and use predict function to directly get the output
    example_file = "data/00001.wav"
    if os.path.exists(example_file):
        with torch.no_grad():
            output = model.predict(example_file, device=device)
            print("Gender : ", output)
    else:
        print(f"Example file {example_file} not found. Please provide a valid audio file.")

if __name__ == "__main__":
    print("Running model test...")
    test_model()
    print("\nTo use the Gradio interface, run: python app.py")