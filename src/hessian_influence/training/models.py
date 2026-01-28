from dataclasses import dataclass
from enum import Enum
from typing import Optional

import torch.nn as nn


class ActivationType(str, Enum):
    RELU = "relu"
    GELU = "gelu"
    TANH = "tanh"


@dataclass
class ModelConfig:
    input_dim: int = 64
    output_dim: int = 10
    hidden_sizes: tuple[int, ...] = (64, 32)
    activation: ActivationType = ActivationType.GELU
    use_bias: bool = True


class ModelFactory:
    @staticmethod
    def get_activation(activation_type: ActivationType) -> nn.Module:
        activation_map = {
            ActivationType.RELU: nn.ReLU,
            ActivationType.GELU: nn.GELU,
            ActivationType.TANH: nn.Tanh,
        }
        return activation_map[activation_type]()

    @classmethod
    def create_mlp(cls, config: ModelConfig) -> nn.Module:
        layers: list[nn.Module] = []
        in_features = config.input_dim

        for hidden_size in config.hidden_sizes:
            layers.append(nn.Linear(in_features, hidden_size, bias=config.use_bias))
            layers.append(cls.get_activation(config.activation))
            in_features = hidden_size

        layers.append(nn.Linear(in_features, config.output_dim, bias=config.use_bias))

        return nn.Sequential(*layers)

    @classmethod
    def create_custom_mlp(
        cls,
        input_dim: int,
        output_dim: int,
        hidden_sizes: list[int],
        activation: Optional[str] = None,
    ) -> nn.Module:
        layers: list[nn.Module] = []
        in_features = input_dim

        if activation is not None:
            act_type = ActivationType(activation.lower())
        else:
            act_type = None

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_features, hidden_size, bias=True))
            if act_type is not None:
                layers.append(cls.get_activation(act_type))
            in_features = hidden_size

        layers.append(nn.Linear(in_features, output_dim, bias=True))

        return nn.Sequential(*layers)

    @staticmethod
    def initialize_weights(
        model: nn.Module,
        init_type: str = "he",
    ) -> None:
        init_type = init_type.lower()

        for module in model.modules():
            if isinstance(module, nn.Linear):
                if init_type in {"he", "kaiming"}:
                    nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                elif init_type == "xavier":
                    nn.init.xavier_uniform_(module.weight)
                else:
                    raise ValueError(f"Unknown initialization type: {init_type}")

                if module.bias is not None:
                    nn.init.zeros_(module.bias)
