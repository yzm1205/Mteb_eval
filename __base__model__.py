import abc
import warnings
from typing import List, Union
from numpy.typing import NDArray
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from transformers.modeling_outputs import (
    CausalLMOutputWithPast,
    BaseModelOutputWithPast,
)
from utils import full_path
from mteb.encoder_interface import PromptType
import torch
def get_device(device: bool) -> str:
    if device and torch.cuda.is_available():
        return device if isinstance(device, str) else "cuda"
    return "cpu"


def mean_pooling(model_output, attention_mask):
    token_embeddings = (
        model_output  # First element of model_output contains all token embeddings
    )
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )
    
    
class Encoder(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def encode(self, sentences: List[str], batch_size: int) -> NDArray:
        pass

    def cpu(self):
        if hasattr(self, "model"):
            self.model.cpu()
        else:
            warnings.warn("Model object does not exist.")


class BaseAutoEncoder(Encoder, abc.ABC):
    def __init__(
            self,
            auto_model_cls: Union[AutoModel, AutoModelForCausalLM],
            shard_model_pt: Path,
            device: bool = False,
    ):
        self.device = get_device(device)
        self.model = (
            auto_model_cls.from_pretrained(
                shard_model_pt, trust_remote_code=True, torch_dtype=torch.float16,
                device_map=self.device
            )
            .eval()
            
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            shard_model_pt, trust_remote_code=True
        )
        # Note: Depending on the model, you might have to add pad_token to the tokenizer

    @abc.abstractmethod
    def get_embeddings_from_model_output(
            self, output: Union[BaseModelOutputWithPast, CausalLMOutputWithPast]
    ):
        pass

    @abc.abstractmethod
    def get_model_output(
            self, sent_encode
    ) -> Union[BaseModelOutputWithPast, CausalLMOutputWithPast]:
        pass

    def encode(self, sentences: List[str], task_name: str, prompt_type: PromptType | None = None,  batch_size: int = 32, **kwargs) -> NDArray:
        sent_encode = self.tokenizer(
            sentences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
            return_token_type_ids=False,
        ).to(self.device)

        output = self.get_model_output(sent_encode)

        embeddings = self.get_embeddings_from_model_output(output)
        model_embedding = mean_pooling(embeddings, sent_encode["attention_mask"])
        return model_embedding.cpu().numpy()
            
class AutoModelForCausalLMEncoder(BaseAutoEncoder):
    def __init__(self, shard_model_pt: Path, device: bool = False):
        super().__init__(
            auto_model_cls=AutoModelForCausalLM,
            shard_model_pt=shard_model_pt,
            device=device,
        )

    def get_embeddings_from_model_output(self, output: CausalLMOutputWithPast):
        return output.hidden_states[-1]

    def get_model_output(self, sent_encode) -> CausalLMOutputWithPast:
        with torch.no_grad():
            return self.model(**sent_encode, use_cache=False, output_hidden_states=True)
        

class AutoModelEncoder(BaseAutoEncoder):
    def __init__(self, shard_model_pt: Path, device: bool = False):
        super().__init__(
            auto_model_cls=AutoModel, shard_model_pt=shard_model_pt, device=device
        )

    def get_embeddings_from_model_output(self, output: BaseModelOutputWithPast):
        return output.last_hidden_state

    def get_model_output(self, sent_encode) -> BaseModelOutputWithPast:
        with torch.no_grad():
            return self.model(**sent_encode, use_cache=False)
