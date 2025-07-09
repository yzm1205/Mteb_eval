from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np
from typing import Union, List
from mteb.encoder_interface import PromptType

class LLMEmbeddingWrapper:
    def __init__(self, model_dir: str, model_name: str, device: str = "auto"):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            device_map=device,
            low_cpu_mem_usage=True,
            cache_dir = "/data/shared/"
        )
        
        if model_name == "apple/OpenELM-3B":
            self.tokenizer = AutoTokenizer.from_pretrained(
                'meta-llama/Llama-2-7b-hf',
                add_bos_token=True,
                trust_remote_code=True,
                add_eos_token=True,
                return_tensors='pt'
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "right"
        self.model.eval()

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def encode(self, sentence: Union[str, List[str]], task_name: str, prompt_type: PromptType | None = None, **kwargs) -> np.ndarray:
        encoded_input = self.tokenizer(
            sentence,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512,
            return_token_type_ids=False
        ).to(self.model.device)
        
        # with torch.inference_mode():
        with torch.no_grad():
            model_output = self.model(
                **encoded_input,
                output_hidden_states=True,
                use_cache=False
            )
        sentence_embeddings = self.mean_pooling(model_output.hidden_states[-1], encoded_input['attention_mask']).squeeze()
        return sentence_embeddings.cpu().numpy()
    
if __name__ == "__main__":
    
    model = LLMEmbeddingWrapper("apple/OpenELM-3B", "apple/OpenELM-3B", device="auto")
    
    sentences = ["This is a test sentence.", "Here is another one."]
    embeddings = model.encode(sentences, task_name="", prompt_type=None)
    print("Embeddings shape:", embeddings.shape)