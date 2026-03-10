import torch
import torchvision.transforms as transforms
import timm
import numpy as np

from PIL import Image
import torch.nn.functional as F

class_names = [
    "bulging",
    "cataract",
    "crossed",
    "hordeolum",
    "normal",
    "uveitis"
]

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])


def load_model():

    model = timm.create_model(
        "convnextv2_base",
        pretrained=False,
        num_classes=6
    )

    model.load_state_dict(torch.load("Model.pth", map_location="cpu"))

    model.eval()

    return model


def predict(model, image):

    img = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(img)
        probs = F.softmax(output, dim=1)

    probs = probs.numpy()[0]

    results = {
        class_names[i]: float(probs[i])
        for i in range(len(class_names))
    }

    return results