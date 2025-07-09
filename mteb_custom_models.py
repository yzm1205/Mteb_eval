# mteb/models/mteb_custom_models.py

from __future__ import annotations
import warnings
from functools import partial
from typing import List, Union, Any
from numpy.typing import NDArray
from pathlib import Path
import abc

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from transformers.modeling_outputs import (
    CausalLMOutputWithPast,
    BaseModelOutputWithPast,
)

from mteb.encoder_interface import Encoder, PromptType
from mteb.model_meta import ModelMeta
from mteb.models.wrapper import Wrapper
from mteb.models.utils import batched # MTEB provides its own batched utility
from tqdm import tqdm


def get_device(device_input: Union[str, bool]) -> str:
    """Determine the device to use (cuda or cpu)."""
    if isinstance(device_input, str):
        if device_input == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device_input
    elif isinstance(device_input, bool) and device_input:
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cpu"


def mean_pooling(model_output: torch.Tensor, attention_mask: torch.Tensor):
    """Performs mean pooling over token embeddings."""
    token_embeddings = model_output
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


class BaseAutoEncoderWrapper(Wrapper): # Renamed to Wrapper and added __init__
    """
    Base wrapper for AutoModel and AutoModelForCausalLM to be used with MTEB.
    This class handles the common loading and encoding logic.
    """
    def __init__(
        self,
        model_name_or_path: str,
        auto_model_cls: Union[AutoModel, AutoModelForCausalLM],
        revision: str | None = None,
        device: Union[str, bool] = "auto",
        torch_dtype: torch.dtype = torch.float32,
        cache_dir: Path | str | None = None,
        max_length: int = 1024,
        **kwargs: Any,
    ):
        # super().__init__(model_name_or_path, revision, **kwargs) # Pass to MTEB Wrapper
        self.model_name = model_name_or_path
        self.device = device
        self.max_length = max_length

        if self.device == "auto" and torch.cuda.device_count() < 2:
            # For multi-GPU, device_map="auto" is generally recommended
            device_map_arg = "cuda"
        else:
            device_map_arg = self.device # For single device or CPU

        self.model = (
            auto_model_cls.from_pretrained(
                model_name_or_path,
                revision=revision,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
                device_map=device_map_arg,
                cache_dir=cache_dir,
                **kwargs, # Pass additional kwargs to from_pretrained
            )
            .eval()
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, revision=revision, trust_remote_code=True, cache_dir=cache_dir
        )
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            warnings.warn(
                "Tokenizer does not have a pad token, setting pad_token to eos_token. "
                "This might not be optimal for all models."
            )

    @abc.abstractmethod
    def _get_embeddings_from_model_output(
        self, output: Union[BaseModelOutputWithPast, CausalLMOutputWithPast]
    ) -> torch.Tensor:
        """Abstract method to extract token embeddings from model output."""
        pass

    def encode(
        self,
        sentences: List[str],
        *,
        task_name: str,
        prompt_type: PromptType | None = None,
        batch_size: int = 32,
        **kwargs: Any,
    ) -> NDArray:
        all_embeddings = []
        total_batches = (len(sentences) + batch_size - 1) // batch_size
        for batch in tqdm(batched(sentences, batch_size),desc=f"Encoding {task_name}",
        total=total_batches,
        leave=True
    ):
            pass
        #     sent_encode = self.tokenizer(
        #         batch,
        #         return_tensors="pt",
        #         padding=True,
        #         truncation=True,
        #         max_length=self.max_length,
        #         return_token_type_ids=False,
        #     )
        #     sent_encode = {k: v.to(self.model.device) for k, v in sent_encode.items()}
        #     with torch.no_grad():
        #         # Pass through the model to get hidden states
        #         output = self.model(**sent_encode, use_cache=False, output_hidden_states=True)

        #     embeddings = self._get_embeddings_from_model_output(output)
        #     model_embedding = mean_pooling(embeddings, sent_encode["attention_mask"].to(embeddings.device))
        #     all_embeddings.append(model_embedding.cpu().numpy())

        # return np.concatenate(all_embeddings, axis=0)


class AutoModelForCausalLMEncoderWrapper(BaseAutoEncoderWrapper):
    """Wrapper for AutoModelForCausalLM models."""
    def __init__(self, model_name_or_path: str, revision: str | None = None, **kwargs: Any):
        super().__init__(
            model_name_or_path=model_name_or_path,
            auto_model_cls=AutoModelForCausalLM,
            revision=revision,
            **kwargs,
        )

    def _get_embeddings_from_model_output(self, output: CausalLMOutputWithPast) -> torch.Tensor:
        # For Causal LMs, the last hidden state from `output_hidden_states=True` is typically used.
        return output.hidden_states[-1]


class AutoModelEncoderWrapper(BaseAutoEncoderWrapper):
    """Wrapper for AutoModel models."""
    def __init__(self, model_name_or_path: str, revision: str | None = None, **kwargs: Any):
        super().__init__(
            model_name_or_path=model_name_or_path,
            auto_model_cls=AutoModel,
            revision=revision,
            **kwargs,
        )

    def _get_embeddings_from_model_output(self, output: BaseModelOutputWithPast) -> torch.Tensor:
        # For AutoModel, `last_hidden_state` is typically the pooled output or what you'd pool from.
        return output.last_hidden_state


class OpenELMEncoderWrapper(AutoModelForCausalLMEncoderWrapper):
    """
    OpenELM Encoder adapted for MTEB.
    Note: OpenELM uses a Llama-2 tokenizer.
    """
    def __init__(
        self,
        model_name_or_path: str = "/data/shared/models--apple--OpenELM-3B",
        revision: str | None = None,
        device: Union[str, bool] = "auto",
        cache_dir: Path | str | None = "/data/shared/",  # Default cache directory
        torch_dtype: torch.dtype = torch.float32,
        max_length: int = 1024,
        **kwargs: Any,
    ):
        self.device = device
        # Override tokenizer_id if different from model_id for from_pretrained
        tokenizer_id = "meta-llama/Llama-2-7b-hf"

        # Load model using the specific OpenELM ID, but tokenizer from Llama-2
        if self.device == "auto" and torch.cuda.device_count() < 2:
            # For multi-GPU, device_map="auto" is generally recommended
            device_map_arg = "cuda"
        else:
            device_map_arg = self.device # For single device or CPU
            
        print("Model on Device:", device_map_arg)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            cache_dir=cache_dir,
            device_map=device_map_arg,
            torch_dtype=torch_dtype,
            **kwargs,
        ).eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_id,
            cache_dir=cache_dir,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            warnings.warn(
                "Tokenizer for OpenELM does not have a pad token, setting pad_token to eos_token."
            )
        
        self.model_name = model_name_or_path
        self.revision = revision
        self.device = get_device(device) # Store the resolved device
        self.max_length = max_length


    def encode(
        self,
        sentences: List[str],
        *,
        task_name: str,
        prompt_type: PromptType | None = None,
        batch_size: int = 32,
        **kwargs: Any,
    ) -> NDArray:
        all_embeddings = []
        total_batches = (len(sentences) + batch_size - 1) // batch_size
        for batch in tqdm(batched(sentences, batch_size),desc=f"Encoding {task_name}",
        total=total_batches,
        leave=True
            ):
            sent_encode = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_token_type_ids=False,
            ) 
            sent_encode = {k: v.to(self.model.device) for k, v in sent_encode.items()}
            with torch.no_grad():
                output = self.model(
                    **sent_encode, output_hidden_states=True, use_cache=False
                )

            # For OpenELM, similar to other CausalLMs, use the last hidden state
            embeddings = output.hidden_states[-1]
            model_embedding = mean_pooling(embeddings, sent_encode["attention_mask"].to(embeddings.device))
            all_embeddings.append(model_embedding.cpu().numpy())

        return np.concatenate(all_embeddings, axis=0)


class OLMoEncoderWrapper(AutoModelForCausalLMEncoderWrapper):
    """OLMo Encoder adapted for MTEB."""

    def __init__(
        self,
        model_name_or_path: str = "/data/shared/olmo/OLMo-7B_shard_size_2GB", # Local path or HF ID
        revision: str | None = None,
        device: Union[str, bool] = "auto",
        cache_dir: Path | str | None = "/data/shared/",
        torch_dtype: torch.dtype = torch.float16,
        max_length: int = 1024,
        **kwargs: Any,
    ):
        super().__init__(
            model_name_or_path=model_name_or_path,
            revision=revision,
            device=device,
            cache_dir=cache_dir,
            torch_dtype=torch_dtype,
            max_length=max_length,
            **kwargs,
        )
        
    def encode(
        self,
        sentences: List[str],
        *,
        task_name: str,
        prompt_type: PromptType | None = None,
        batch_size: int = 32,
        **kwargs: Any,
    ) -> NDArray:
        all_embeddings = []
        total_batches = (len(sentences) + batch_size - 1) // batch_size
        for batch in tqdm(batched(sentences, batch_size),desc=f"Encoding {task_name}",
        total=total_batches,
        leave=True
            ):
            sent_encode = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_token_type_ids=False,
            ) 
            sent_encode = {k: v.to(self.model.device) for k, v in sent_encode.items()}
            with torch.no_grad():
                output = self.model(
                    **sent_encode, output_hidden_states=True, use_cache=False
                )

            # For OpenELM, similar to other CausalLMs, use the last hidden state
            embeddings = output.hidden_states[-1]
            model_embedding = mean_pooling(embeddings, sent_encode["attention_mask"].to(embeddings.device))
            all_embeddings.append(model_embedding.cpu().numpy())

        return np.concatenate(all_embeddings, axis=0)


# --- ModelMeta Definitions ---

olmo_7b_base = ModelMeta(
    loader=partial(
        OLMoEncoderWrapper,
        model_name_or_path="/data/shared/olmo/OLMo-7B_shard_size_2GB", # Path to your OLMo model
        # revision="<OLMO_REVISION_HASH>", # Add if you have a specific revision
        revision="local-2025-06-25-olmo-v1",
        torch_dtype=torch.float32,
        max_length=1024,
        cache_dir="/data/shared/", # Default cache directory
    ),
    name="Bridge-AI/OLMo-7B-Base-MTEB", # A distinctive name
    languages=["eng-Latn"], # Assuming English
    open_weights=True, # Assuming weights are public
    # revision="<OLMO_REVISION_HASH>", # Add if you have a specific revision
    revision="local-2025-06-25-olmo-v1",
    release_date="2024-06-25", # Current date as placeholder
    n_parameters=7_000_000_000, # Approx 7B
    # You might need to run `ModelMeta.calculate_memory_usage_mb(loader())` to get this
    memory_usage_mb=28000, # Placeholder, will depend on float16 and model size
    max_tokens=1024, # From your code
    embed_dim=4096, # Common for 7B models, verify with actual model
    license="apache-2.0", # OLMo is Apache 2.0
    reference="https://allenai.org/olmo", # Official OLMo reference
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    use_instructions=False, # These models are not instruction-tuned for embeddings like LLM2Vec
    public_training_code="https://github.com/allenai/OLMo",
    training_datasets=None, # Provide if publicly available
    public_training_data=None, # Provide if publicly available
)


openelm_3b_base = ModelMeta(
    loader=partial(
        OpenELMEncoderWrapper,
        model_name_or_path="apple/OpenELM-3B", # Hugging Face Model ID
        revision="local-2025-06-25-openelm-v1",# Or a specific commit hash
        torch_dtype=torch.float32,
        max_length=1024,
        cache_dir="/data/shared/", # Default cache directory
    ),
    name="Bridge-AI/OpenELM-3B-Base-MTEB", # A distinctive name
    languages=["eng-Latn"], # Assuming English
    open_weights=True,
    revision="local-2025-06-25-openelm-v1",
    release_date="2024-04-24", # OpenELM 3B release date (approx)
    n_parameters=3_000_000_000, # Approx 3B
    # You might need to run `ModelMeta.calculate_memory_usage_mb(loader())` to get this
    memory_usage_mb=12000, # Placeholder, depends on float16 and model size
    max_tokens=1024,
    embed_dim=2560, # OpenELM-3B has 2560 hidden size, verify
    license="apache-2.0", # OpenELM is Apache 2.0
    reference="https://huggingface.co/apple/OpenELM-3B", # Official HF reference
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    use_instructions=False, # These models are not instruction-tuned for embeddings
    public_training_code="https://github.com/apple/OpenELM",
    training_datasets=None, # Provide if publicly available
    public_training_data=None, # Provide if publicly available
)