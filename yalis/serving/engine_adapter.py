import asyncio
from typing import Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import torch

from yalis.engine import prefill, generate
from yalis.constants import EnginePhase
from yalis import ModelConfig, InferenceConfig, LLMEngine
from .logger import get_logger

logger = get_logger("engine_adapter")

class AsyncEngineAdapter:
    """
    Async facade over the synchronous LLMEngine and owner of its lifecycle.
    Uses a single-step lock to serialize model steps (shared KV/cache).
    """

    def __init__(self, model_config: ModelConfig, inference_config: InferenceConfig) -> None:
      self.model_config = model_config
      self.inference_config = inference_config 
      self._engine: Optional[LLMEngine] = None
      self._step_lock = asyncio.Lock()
      self._executor: Optional[ThreadPoolExecutor] = None

    async def start(self) -> None:
      if self._engine is not None:

        return
      # Create a dedicated single-thread executor so all model work runs on one thread.
      self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="yalis-engine")
      def _build():
        return LLMEngine(model_config=self.model_config, inference_config=self.inference_config)
      loop = asyncio.get_running_loop()
      self._engine = await loop.run_in_executor(self._executor, _build)

    async def stop(self) -> None:
      # Teardown the executor
      if self._executor is not None:
        self._executor.shutdown(wait=True, cancel_futures=False)
        self._executor = None
      return

    @property
    def engine(self) -> LLMEngine:
        assert self._engine is not None, "AsyncEngineAdapter.start() must be called before accessing engine"
        return self._engine

    @property
    def tokenizer(self):
        return self.engine.tokenizer

    async def prefill(self, tokens: torch.Tensor, unpadded_prompt_lengths: torch.Tensor, block_table: torch.Tensor, token_counter: torch.Tensor):
      async with self._step_lock:
        assert self._executor is not None
        logger.info(f"Prefill block_table: {block_table}")
        loop = asyncio.get_running_loop()
        fn = partial(
          prefill,
          self.engine.model,
          tokens,
          unpadded_prompt_lengths,
          self.engine.inference_config.temperature,
          self.engine.inference_config.top_k,
          self.engine.inference_config.top_p,
          False,
          EnginePhase.PREFILL,
          block_table,
          token_counter,
        )
        return await loop.run_in_executor(self._executor, fn)

    async def decode(self, last_tokens: torch.Tensor, block_table: torch.Tensor, token_counter: torch.Tensor):
      async with self._step_lock:
        assert self._executor is not None
        loop = asyncio.get_running_loop()
        fn = partial(
          generate,
          self.engine.model,
          last_tokens,
          self.engine.inference_config.temperature,
          self.engine.inference_config.top_k,
          self.engine.inference_config.top_p,
          False,
          EnginePhase.DECODE_SINGLE,
          block_table,
          token_counter,
        )
        return await loop.run_in_executor(self._executor, fn)


