import warnings
from typing import Any, Dict, Optional, Union

from segmentation_models_pytorch.base import (
    ClassificationHead,
    SegmentationHead,
    SegmentationModel,
)
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base.hub_mixin import supports_config_loading

from .decoder import LinknetDecoder


class Linknet(SegmentationModel):
    """Linknet_ is a fully convolution neural network for image semantic segmentation. Consist of *encoder*
    and *decoder* parts connected with *skip connections*. Encoder extract features of different spatial
    resolution (skip connections) which are used by decoder to define accurate segmentation mask. Use *sum*
    for fusing decoder blocks with skip connections.

    Note:
        This implementation by default has 4 skip connections (original - 3).

    Args:
        encoder_name: Name of the classification model that will be used as an encoder (a.k.a backbone)
            to extract features of different spatial resolution
        encoder_depth: A number of stages used in encoder in range [3, 5]. Each stage generate features
            two times smaller in spatial dimensions than previous one (e.g. for depth 0 we will have features
            with shapes [(N, C, H, W),], for depth 1 - [(N, C, H, W), (N, C, H // 2, W // 2)] and so on).
            Default is 5
        encoder_weights: One of **None** (random initialization), **"imagenet"** (pre-training on ImageNet) and
            other pretrained weights (see table with available weights for each encoder_name)
        decoder_use_norm:     Specifies normalization between Conv2D and activation.
            Accepts the following types:
            - **True**: Defaults to `"batchnorm"`.
            - **False**: No normalization (`nn.Identity`).
            - **str**: Specifies normalization type using default parameters. Available values:
              `"batchnorm"`, `"identity"`, `"layernorm"`, `"instancenorm"`, `"inplace"`.
            - **dict**: Fully customizable normalization settings. Structure:
              ```python
              {"type": <norm_type>, **kwargs}
              ```
              where `norm_name` corresponds to normalization type (see above), and `kwargs` are passed directly to the normalization layer as defined in PyTorch documentation.

            **Example**:
            ```python
            decoder_use_norm={"type": "layernorm", "eps": 1e-2}
            ```
        in_channels: A number of input channels for the model, default is 3 (RGB images)
        classes: A number of classes for output mask (or you can think as a number of channels of output mask)
        activation: An activation function to apply after the final convolution layer.
            Available options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**, **"tanh"**, **"identity"**,
                **callable** and **None**.
            Default is **None**
        aux_params: Dictionary with parameters of the auxiliary output (classification head). Auxiliary output is build
            on top of encoder if **aux_params** is not **None** (default). Supported params:
                - classes (int): A number of classes
                - pooling (str): One of "max", "avg". Default is "avg"
                - dropout (float): Dropout factor in [0, 1)
                - activation (str): An activation function to apply "sigmoid"/"softmax"
                    (could be **None** to return logits)
        kwargs: Arguments passed to the encoder class ``__init__()`` function. Applies only to ``timm`` models. Keys with ``None`` values are pruned before passing.

    Returns:
        ``torch.nn.Module``: **Linknet**

    .. _Linknet:
        https://arxiv.org/abs/1707.03718
    """

    @supports_config_loading
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_use_norm: Union[bool, str, Dict[str, Any]] = "batchnorm",
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[Union[str, callable]] = None,
        aux_params: Optional[dict] = None,
        **kwargs: dict[str, Any],
    ):
        super().__init__()

        if encoder_name.startswith("mit_b"):
            raise ValueError(
                "Encoder `{}` is not supported for Linknet".format(encoder_name)
            )

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
            **kwargs,
        )

        decoder_use_batchnorm = kwargs.pop("decoder_use_batchnorm", None)
        if decoder_use_batchnorm is not None:
            warnings.warn(
                "The usage of decoder_use_batchnorm is deprecated. Please modify your code for use_norm",
                DeprecationWarning,
                stacklevel=2,
            )
            decoder_use_norm = decoder_use_batchnorm

        self.decoder = LinknetDecoder(
            encoder_channels=self.encoder.out_channels,
            n_blocks=encoder_depth,
            prefinal_channels=32,
            use_norm=decoder_use_norm,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=32, out_channels=classes, activation=activation, kernel_size=1
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(
                in_channels=self.encoder.out_channels[-1], **aux_params
            )
        else:
            self.classification_head = None

        self.name = "link-{}".format(encoder_name)
        self.initialize()
