# VoxGender: Voice Gender Classifier

<p align="center">
  <img src="https://github.com/LitZeus/VoxGender/raw/main/static/samples/logo.png" alt="VoxGender Logo" width="200">
</p>

VoxGender is a modern voice gender classification tool with a clean, intuitive interface. Features include:

- Real-time voice gender prediction with confidence scores
- Browser-based voice recording capability
- Clean, modern Gradio interface
- High accuracy (98.7% on VoxCeleb1)

## Installation

### Option 1: Clone the repository
```bash
git clone https://github.com/LitZeus/VoxGender.git
cd VoxGender
pip install -r requirements.txt
```

### Option 2: Download and install
```bash
# Download the latest release
pip install -r requirements.txt
```

## Usage

### Web Interface
To use the Gradio web interface, run:
```
python app.py
```
This will start a local web server where you can:
- Upload audio files
- Record your voice directly in the browser
- Get gender predictions with confidence scores

### Python API
```python
import torch

from model import ECAPA_gender

# You could directly download the model from the huggingface model hub
model = ECAPA_gender.from_pretrained("JaesungHuh/voice-gender-classifier")
model.eval()

# If you are using gpu ....
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load the audio file and use predict function to directly get the output
example_file = "data/00001.wav"
with torch.no_grad():
    output = model.predict(example_file, device=device)
    print("Gender : ", output)
```

### Quick Test
To run a quick test of the model:
```
python test.py
```

## Pretrained weights
For those who need pretrained weights, please download them [here](https://drive.google.com/file/d/1ojtaa6VyUhEM49F7uEyvsLSVN3T8bbPI/view?usp=sharing).

## Training details
State-of-the-art speaker verification model already produces good representation of the speaker's gender.

I used the pretrained ECAPA-TDNN from [TaoRuijie's](https://github.com/TaoRuijie/ECAPA-TDNN) repository, added one linear layer to make a two-class classifier, and finetuned the model with the VoxCeleb2 dev set.

The model achieved **98.7%** accuracy on the VoxCeleb1 identification test split.

## Caveat
I would like to note that the training dataset I've used for this model (VoxCeleb) may not represent the global human population. Please be careful of unintended biases when using this model.

## References

- Model architecture based on [TaoRuijie's](https://github.com/TaoRuijie/ECAPA-TDNN) ECAPA-TDNN implementation
- For more details about ECAPA-TDNN, check the [paper](https://arxiv.org/abs/2005.07143)

## License
This project is licensed under the MIT License - see the LICENSE file for details.
