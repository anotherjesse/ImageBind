# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import re
import sys
import requests
import tempfile
from io import BytesIO
from typing import List, Optional
from PIL import Image
from cog import BasePredictor, Input, Path, BaseModel
import data
import torch
from models import imagebind_model
from models.imagebind_model import ModalityType

MODALITY_TO_PREPROCESSING = {
    ModalityType.TEXT: data.load_and_transform_text,
    ModalityType.VISION: data.load_and_transform_vision_data,
    ModalityType.AUDIO: data.load_and_transform_audio_data,
}

class NamedEmbedding(BaseModel):
    input: str
    embedding: List[float]

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        model = imagebind_model.imagebind_huge(pretrained=True)
        model.eval()
        self.model = model.to("cuda")

    def predict(
        self,
        inputs: str = Input(
            description="Newline-separated inputs. Can either be strings of text or image URIs starting with http[s]://",
            default='',
        ),
    ) -> List[NamedEmbedding]:
        device = "cuda"
        outputs = []

        for line in inputs.strip().splitlines():
            line = line.strip()

            if re.match("^https?://", line):
                try:
                    url = line
                    print(f"Downloading {url}", file=sys.stderr)
                    response = requests.get(url)
                    with tempfile.NamedTemporaryFile(delete=True) as tmp_file:
                        tmp_file.write(response.content)

                        modality_function = MODALITY_TO_PREPROCESSING['vision']
                        model_input = {'vision': modality_function([tmp_file.name], device)}

                        with torch.no_grad():
                            embeddings = self.model(model_input)

                        emb = embeddings['vision']
                        outputs.append(
                            NamedEmbedding(input=line, embedding=emb.cpu().squeeze().tolist())
                        )

                except Exception as e:
                    print(f"Failed to load {line}: {e}", file=sys.stderr)
            else:
                modality_function = MODALITY_TO_PREPROCESSING['text']
                model_input = {'text': modality_function([line], device)}

                with torch.no_grad():
                    embeddings = self.model(model_input)

                emb = embeddings['text']
                outputs.append(
                    NamedEmbedding(input=line, embedding=emb.cpu().squeeze().tolist())
                )

        return outputs