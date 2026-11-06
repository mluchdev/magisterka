# Requires transformers 4.x (the model was saved with 4.55.0), not the project's default 5.x.

# this is also a baseline but still having problems to start it

from functools import cache

import torch
from transformers import AutoModel

MODEL_ID = "iitolstykh/GigaCheck-Classifier-Multi"


@cache
def _load():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(
        MODEL_ID, trust_remote_code=True, device_map=device, torch_dtype=torch.bfloat16
    )
    ai_idx = next(int(i) for i, label in model.config.id2label.items() if label == "ai")
    return model, ai_idx


def gigacheck_detect(text: str) -> float:
    """Return the probability (0-1) that `text` is AI-generated."""
    model, ai_idx = _load()
    with torch.no_grad():
        output = model([text.replace("\n", " ")])
    return output.classification_head_probs[0, ai_idx].item()


if __name__ == "__main__":
    print(gigacheck_detect("""To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them.
"""))
