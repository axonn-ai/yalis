from typing import Optional, Literal, Tuple, List
import os
from packaging.version import Version
from importlib.metadata import version, PackageNotFoundError
from yalis.attention.backends import AttentionBackend


# Valid component names for CPU offloading
# - "mlp": MLP layers (experts-only for MoE, full MLP for dense models)
# - "attn": Attention layers
# - "norm": Normalization layers
# Note: For MoE models, the router (gate) always stays on GPU
VALID_OFFLOAD_COMPONENTS = ["mlp", "attn", "norm"]


class ModelConfig:
    """
    Configuration for model initialization and management.
    """

    def __init__(
        self,
        model_name: str,
        model_path: Optional[str] = None,
        precision: Literal["fp32", "fp16", "bf16"] = "fp16",
        disable_tp: bool = False,
    ):
        """
        Initialize the model configuration.

        Args:
            model_name (str): Name of the pretrained model.
            model_path (Optional[str]): Path to custom model weights.
            precision (str): Model precision, default is 'fp16'.
            disable_tp (bool): Disable tensor parallelism. Need model-specific
                                TP for speculative decoding.
        """
        # Todo: make model_name optional. If only model_path is provided then
        # we should par
        self.model_name = model_name
        self.model_path = model_path
        self.precision = precision
        self.disable_tp = disable_tp
        self._validate()
        self.model_path = self._resolve_model_path(self.model_path)

    def _resolve_model_path(self, model_path: Optional[str]) -> str:
        """
        Resolve the model path by checking the YALIS_CACHE environment variable

        Args:
            model_path (Optional[str]): The provided model path.

        Returns:
            str: The resolved model path.

        Raises:
            ValueError: If the resolved model path does not exist.
        """
        if model_path is None:
            # Default to the YALIS_CACHE env variable or a fallback directory
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
            raise ValueError(
                "Either 'model_name' or 'model_path' must be provided."
            )

        if self.precision not in {"fp32", "fp16", "bf16"}:

            raise ValueError(
                f"Invalid precision: {self.precision}. Supported values are 'fp32', 'fp16', 'bf16'."  # noqa: E501
            )

    def __repr__(self):
        return (
            f"ModelConfig(model_name={self.model_name}, model_path={self.model_path}, "  # noqa: E501
            f"precision={self.precision}"
        )


class InferenceConfig:
    """
    Configuration for inference parameters.
    """

    def __init__(
        self,
        max_batch_size: int = 1,
        max_length_of_generated_sequences: int = 1024,
        top_k: Optional[int] = None,
        top_p: Optional[float] = 1.0,
        temperature: Optional[float] = 1.0,
        metrics: bool = False,
        tp_dims: Optional[Tuple[int, int, int]] = None,
        attention_backend: str = "flash",
        use_intra_head_parallelism: bool = False,
        use_paged_kv_caching: bool = False,
        prestore_kv_cache: bool = True,
        symmetric_allreduce_strategy: Optional[
            Literal["one-shot", "two-shot", "nvshmem"]
        ] = None,
        # CPU Offloading options
        use_cpu_offloading: bool = False,
        cpu_offload_mode: Optional[Literal["all", "rows", "inline"]] = "all",
        cpu_offload_num_prefetch_layers: int = 1,
        cpu_offload_pin_memory: bool = True,
        cpu_offload_use_preallocated_buffers: bool = False,
        cpu_offload_components: Optional[List[str]] = None,
        use_prefetched: bool = False,
        prefetch_default_vect_path: Optional[str] = None,
    ):
        """
        Initialize the inference configuration.

        Args:
            max_batch_size (int): Maximum number of inputs processed in
                            parallel. The model will allocate KV cache
                            for this many sequences. During inference,
                            any batch size <= max_batch_size can be used.
                            This enables dynamic batching for efficient
                            resource utilization.
            max_length_of_generated_sequences (int): Max generated seq length
            decoding_strategy (str): Decoding strategy, default is 'greedy'.
            num_beams (Optional[int]): Number of beams for beam search.
            temperature (Optional[float]): Sampling temperature.
            top_k (Optional[int]): Top-k sampling limit.
            top_p (Optional[float]): Nucleus sampling probability.
            tp_dims (Optional[Tuple[int, int, int]]): Tensor parallel dims.
                            If None, all GPUs are used in the first dimension.
            metrics (bool): Enable real-time metrics collection.
            attention_backend (str): Attention backend to use.
                            Options are 'flash', 'flex', or 'sdpa'.
            use_intra_head_parallelism (bool): Use intra-head parallelism.
            use_paged_kv_caching (bool): Use paged k/v caching for attention.
            prestore_kv_cache (bool): Pre-store k/v cache before attention.
            use_cpu_offloading (bool): Enable CPU offloading for memory
                            efficiency. Keeps model on CPU and streams
                            layers to GPU on demand.
            cpu_offload_num_prefetch_layers (int): Number of layers to
                            prefetch when CPU offloading is enabled.
            cpu_offload_pin_memory (bool): Pin CPU memory for faster
                            CPU->GPU transfers during offloading.
            cpu_offload_use_preallocated_buffers (bool): Use fixed GPU 
                            buffers with .copy_() instead of .to() for
                            zero-allocation transfers.
            cpu_offload_components (List[str]): Components to offload/prefetch.
                            Options: "mlp", "attn", "norm"
                            Default: ["mlp", "attn", "norm"] (full layer)
                            Note: For MoE models, "mlp" means experts only
                                  (router stays on GPU permanently)
            use_prefetched: bool = False,
            prefetch_default_vect_path: Optional[str] = None,
        """
        self.max_batch_size = max_batch_size
        # TODO - default max_length should be none.
        # If it is none, we should set it from the model config
        self.max_length = max_length_of_generated_sequences
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.metrics = metrics
        self.tp_dims = tp_dims
        self.use_intra_head_parallelism = use_intra_head_parallelism
        self.use_paged_kv_caching = use_paged_kv_caching
        self.prestore_kv_cache = prestore_kv_cache
        self.symmetric_allreduce_strategy = symmetric_allreduce_strategy
        # CPU Offloading
        self.use_cpu_offloading = use_cpu_offloading
        self.cpu_offload_mode = cpu_offload_mode
        self.cpu_offload_num_prefetch_layers = cpu_offload_num_prefetch_layers
        self.cpu_offload_pin_memory = cpu_offload_pin_memory
        self.cpu_offload_use_preallocated_buffers = cpu_offload_use_preallocated_buffers
        
        # Validate and store offload components
        if cpu_offload_components is None:
            self.cpu_offload_components = ["mlp", "attn", "norm"]  # Full layer
        else:
            for comp in cpu_offload_components:
                if comp not in VALID_OFFLOAD_COMPONENTS:
                    raise ValueError(
                        f"Invalid component '{comp}' in cpu_offload_components. "
                        f"Valid options: {VALID_OFFLOAD_COMPONENTS}"
                    )
            self.cpu_offload_components = cpu_offload_components
        self.use_prefetched = use_prefetched
        self.prefetch_default_vect_path = prefetch_default_vect_path

        if attention_backend not in ["flash", "sdpa", "flex"]:
            raise ValueError(
                f"Invalid attention backend: {attention_backend}. Supported values are 'flash', 'sdpa', 'flex'."  # noqa: E501
            )
        self.attention_backend = AttentionBackend(attention_backend)
        try:
            pkg_ver = version("torch")
        except PackageNotFoundError:
            raise RuntimeError("torch isn’t installed")
        if Version(pkg_ver) < Version("2.6.0"):
            raise RuntimeError(f"torch >= 2.6.0 required (found {pkg_ver})")

        self._validate()

    def _validate(self):
        """
        Validate the configuration.
        """
        if self.max_batch_size <= 0:
            raise ValueError("max_batch_size must be a positive integer.")

        if self.max_length <= 0:
            raise ValueError("max_length must be a positive integer.")

        if self.temperature is not None and (self.temperature < 0.0):
            raise ValueError("temperature must be >=0.0.")

        if self.top_k is not None and self.top_k <= 0:
            raise ValueError("top_k must be a positive integer.")

        if self.top_p is not None and (self.top_p < 0.0 or self.top_p > 1.0):
            raise ValueError("top_p must be in the range [0.0, 1.0].")

        if self.tp_dims is not None and (
            not isinstance(self.tp_dims, tuple) or len(self.tp_dims) != 3
        ):
            raise ValueError("tp_dims must be a 3-dimensional tuple.")

        if (
            self.use_paged_kv_caching
            and not self.attention_backend == AttentionBackend.FLASH
        ):
            raise ValueError(
                "use_paged_kv_caching requires attention_backend=flash"
            )

        if (
            self.use_intra_head_parallelism
            and not self.attention_backend == AttentionBackend.SDPA
        ):
            raise ValueError(
                "use_intra_head_parallelism requires attention_backend=sdpa"
            )

        if (
            self.symmetric_allreduce_strategy is not None
            and self.symmetric_allreduce_strategy
            not in ["one-shot", "two-shot", "nvshmem"]
        ):
            raise ValueError(
                "symmetric_allreduce_strategy must be one of"
                " 'one-shot', 'two-shot', 'nvshmem', or None."
            )

        # CPU Offloading validation
        if self.use_cpu_offloading:
            if self.cpu_offload_num_prefetch_layers < 1:
                raise ValueError(
                    "cpu_offload_num_prefetch_layers must be >= 1"
                )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"  max_batch_size={self.max_batch_size},\n"
            f"  max_length_of_generated_sequences={self.max_length},\n"
            f"  top_k={self.top_k},\n"
            f"  top_p={self.top_p},\n"
            f"  temperature={self.temperature},\n"
            f"  metrics={self.metrics},\n"
            f"  tp_dims={self.tp_dims},\n"
            f"  use_intra_head_parallelism={self.use_intra_head_parallelism},\n"
            f"  attention_backend={self.attention_backend.value},\n"
            f"  use_paged_kv_caching={self.use_paged_kv_caching},\n"
            f"  prestore_kv_cache={self.prestore_kv_cache},\n"
            f"  use_cpu_offloading={self.use_cpu_offloading},\n"
            f"  cpu_offload_num_prefetch_layers={self.cpu_offload_num_prefetch_layers},\n"
            f"  cpu_offload_pin_memory={self.cpu_offload_pin_memory},\n"
            f"  cpu_offload_use_preallocated_buffers={self.cpu_offload_use_preallocated_buffers},\n"
            f"  cpu_offload_components={self.cpu_offload_components}\n"
            f")"
        )
