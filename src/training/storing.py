import torch
from src.utils.path import remove_after_last_slash
from pathlib import Path


def save_model(
    path,
    model,
):
    """
    Save the model to a file.

    :param model: PyTorch model to be saved
    :param path: File path where the model will be saved
    """
    Path(remove_after_last_slash(path)).mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Model saved at {path}")


def load_model(path, model_class, property_keywords=None, *model_args, **model_kwargs,):
    """
    Load the model from a file. Can be partial.

    :param path: File path where the model is saved
    :param model_class: The class of the model to be loaded
    :param model_args: Positional arguments for the model class
    :param model_kwargs: Keyword arguments for the model class
    :return: Loaded model
    """
    model = model_class(*model_args, **model_kwargs)
    saved_state_dict = torch.load(path)
    if property_keywords is not None:
        for keyword_list in property_keywords:
            saved_state_dict = {
                k: v for k, v in saved_state_dict.items() if all(keyword in k for keyword in keyword_list)
            }
    model.load_state_dict(saved_state_dict, strict=False)
    model.eval()
    print(f"Model loaded from {path}")
    return model
