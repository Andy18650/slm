from slm.models.gru import GRULanguageModel
from slm.models.lstm import LSTMLanguageModel
from slm.models.rnn import RNNLanguageModel
from slm.models.transformer import TransformerLanguageModel


def build_model(model_config: dict, vocab_size: int):
    model_type = model_config["type"].lower()
    kwargs = {key: value for key, value in model_config.items() if key != "type"}

    if model_type == "rnn":
        return RNNLanguageModel(vocab_size=vocab_size, **kwargs)
    if model_type == "lstm":
        return LSTMLanguageModel(vocab_size=vocab_size, **kwargs)
    if model_type == "gru":
        return GRULanguageModel(vocab_size=vocab_size, **kwargs)
    if model_type == "transformer":
        return TransformerLanguageModel(vocab_size=vocab_size, **kwargs)

    raise ValueError(f"Unsupported model type: {model_type}")


__all__ = [
    "GRULanguageModel",
    "LSTMLanguageModel",
    "RNNLanguageModel",
    "TransformerLanguageModel",
    "build_model",
]
