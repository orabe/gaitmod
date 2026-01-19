from typing import Any

from gaitmod.models import Seq2SeqLSTM, Seq2VecCNN, Seq2VecLSTM, Seq2VecMLP, Seq2VecMLPLSTM
from gaitmod.pipelines import build_pipeline as _build_pipeline


def build_pipeline(*args, **kwargs):
    """Wrapper around gaitmod.pipelines.build_pipeline for the training API."""
    return _build_pipeline(*args, **kwargs)


def build_model(model_type: str, **kwargs: Any):
    """Construct a model instance by type for direct use."""
    if model_type == 'Seq2SeqLSTM':
        return Seq2SeqLSTM(**kwargs)
    if model_type == 'Seq2VecLSTM':
        return Seq2VecLSTM(**kwargs)
    if model_type == 'Seq2VecMLP':
        return Seq2VecMLP(**kwargs)
    if model_type == 'Seq2VecCNN':
        return Seq2VecCNN(**kwargs)
    if model_type == 'Seq2VecMLPLSTM':
        return Seq2VecMLPLSTM(**kwargs)
    raise ValueError(f"Unsupported model_type for build_model: {model_type}")


__all__ = ["build_model", "build_pipeline"]
