# %matplotlib inline
import torch
import torchaudio
import re
import matplotlib
import matplotlib.pyplot as plt
import IPython
import requests

print(torch.__version__)
print(torchaudio.__version__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H

#print("Sample Rate:", bundle.sample_rate)

#print("Labels:", bundle.get_labels())

model = bundle.get_model().to(device)

#print(model.__class__)


class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, labels, ignore):
        super().__init__()
        self.labels = labels
        self.ignore = ignore

    def forward(self, emission: torch.Tensor) -> str:
        """Given a sequence emission over labels, get the best path string
         Args:
             emission (Tensor): Logit tensors. Shape `[num_seq, num_label]`.
             Returns:
                 str: The resulting transcript
                 """
        indices = torch.argmax(emission, dim=-1)  # [num_seq,]
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i not in self.ignore]
        return ''.join([self.labels[i] for i in indices])


def audiototext(SPEECH_FILE):
    waveform, sample_rate = torchaudio.load(SPEECH_FILE)
    waveform = waveform.to(device)
    if sample_rate != bundle.sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)
    with torch.inference_mode():
        features, _ = model.extract_features(waveform)
    with torch.inference_mode():
        emission, _ = model(waveform)
    decoder = GreedyCTCDecoder(labels=bundle.get_labels(),  ignore=(0, 1, 2, 3),)
    transcript = decoder(emission[0])
    transcript = re.sub('[^a-zA-Z0-9 \n\.]', ' ', transcript).lower()
    return transcript

