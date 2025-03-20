from typing import Optional, Literal, Tuple
import os


class ModelConfig:
    """
    Configuration for model initialization and management.
    """

    def __init__(
        self,
        model_name: str,
        model_path: Optional[str] = None,
        precision: Literal["fp32", "fp16", "bf16"] = "fp16",
    ):
        """
        Initialize the model configuration.

        Args:
            model_name (str): Name of the pretrained model.
            model_path (Optional[str]): Path to custom model weights.
            precision (str): Model precision, default is 'fp16'.
        """
        # Todo: make model_name optional. If only model_path is provided then
        # we should par
        self.model_name = model_name
        self.model_path = model_path
        self.precision = precision
        self._validate()
        self.model_path = self._resolve_model_path(self.model_path)

    def _resolve_model_path(self, model_path: Optional[str]) -> str:
        """
        Resolve the model path by checking the YALIS_CACHE environment variable.

        Args:
            model_path (Optional[str]): The provided model path.

        Returns:
            str: The resolved model path.

        Raises:
            ValueError: If the resolved model path does not exist.
        """
        if model_path is None:
            # Default to the YALIS_CACHE environment variable or a fallback directory
            cache_dir = os.getenv("YALIS_CACHE", "~/.cache/yalis/")
            model_path = os.path.join(
                os.path.expanduser(cache_dir), "checkpoints", self.model_name
            )

        # Check if the directory exists
        if not os.path.exists(model_path):
            # ToDo: improve this error message. Ask the user to run download.py
            raise ValueError(f"Model path does not exist: {model_path}")

        return model_path

    def _validate(self):
        """
        Validate the configuration.
        """
        if self.model_path is None and self.model_name is None:
            raise ValueError("Either 'model_name' or 'model_path' must be provided.")

        if self.precision not in {"fp32", "fp16", "bf16"}:

            raise ValueError(
                f"Invalid precision: {self.precision}. Supported values are 'fp32', 'fp16', 'bf16'."
            )

    def __repr__(self):
        return (
            f"ModelConfig(model_name={self.model_name}, model_path={self.model_path}, "
            f"precision={self.precision}"
        )


class InferenceConfig:
    """
    Configuration for inference parameters.
    """

    def __init__(
        self,
        batch_size: int = 1,
        max_length_of_generated_sequences: int = 1024,
        top_k: Optional[int] = None,
        top_p: Optional[float] = 1.0,
        temperature: Optional[float] = 1.0,
        metrics: bool = False,
        tp_dims: Optional[Tuple[int, int, int]] = None,
        use_intra_head_parallelism: bool = False
    ):
        """
        Initialize the inference configuration.

        Args:
            batch_size (int): Number of inputs processed in parallel.
            max_length_of_generated_sequences (int): Maximum length of the generated sequences.
            decoding_strategy (str): Decoding strategy, default is 'greedy'.
            num_beams (Optional[int]): Number of beams for beam search.
            temperature (Optional[float]): Sampling temperature.
            top_k (Optional[int]): Top-k sampling limit.
            top_p (Optional[float]): Nucleus sampling probability.
            metrics (bool): Enable real-time metrics collection.
        """
        self.batch_size = batch_size
        # ToDo - default max_length should be none. If it is none, we should set it
        # from the model config
        # anyway this arg isn't being used right now. KV Cache is defaulting to the model
        # max sequence length
        self.max_length = max_length_of_generated_sequences
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.metrics = metrics
        self.tp_dims = tp_dims
        self.use_intra_head_parallelism = use_intra_head_parallelism

        self._validate()

    def _validate(self):
        """
        Validate the configuration.
        """
        if self.batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")

        if self.max_length <= 0:
            raise ValueError("max_length must be a positive integer.")


        if self.temperature is not None and (self.temperature < 0.0):
            raise ValueError("temperature must be >=0.0.")

        if self.top_k is not None and self.top_k <= 0:
            raise ValueError("top_k must be a positive integer.")

        if self.top_p is not None and (self.top_p <= 0.0 or self.top_p > 1.0):
            raise ValueError("top_p must be in the range (0.0, 1.0].")

        if self.tp_dims is not None and (type(self.tp_dims) != tuple or len(self.tp_dims) != 3):
            raise ValueError("tp_dims must be a 3-dimensional tuple.")


    def __repr__(self):
        return (
            f"InferenceConfig(batch_size={self.batch_size}, max_length={self.max_length}, "
            f"decoding_strategy={self.decoding_strategy}, num_beams={self.num_beams}, "
            f"temperature={self.temperature}, top_k={self.top_k}, top_p={self.top_p}, metrics={self.metrics})"
        )
