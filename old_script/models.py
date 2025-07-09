from pathlib import Path
from typing import List
import torch
from numpy.typing import NDArray
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
import sys
sys.path.insert(0,"./")
from utils import full_path
from __base__model__ import (
    Encoder,
    AutoModelEncoder,
    AutoModelForCausalLMEncoder,
    get_device,
    mean_pooling,
    
)
from mteb.encoder_interface import PromptType

class OpenELMEncoder(Encoder):
    """OpenELM Encoder"""

    def __init__(
            self, device: bool = False, cache_dir: Path = full_path("~/.cache"), **kwargs
    ):
        # self.device = get_device(device)
        if device and torch.cuda.is_available():
            self.device = device if isinstance(device, str) else "cuda"
            print(f"Targeting model load to specific device: {self.device}")
        else:
            self.device = torch.device('cpu')
            print(f"CUDA not available. Using CPU.")
            
        model_id = "apple/OpenELM-3B"
        tokenizer_id = "meta-llama/Llama-2-7b-hf"
        self.model = AutoModelForCausalLM.from_pretrained(
                model_id, trust_remote_code=True, cache_dir=cache_dir,
                device_map=self.device
            )
        # self.model = AutoModel.from_pretrained(
        #         model_id, trust_remote_code=True, cache_dir=cache_dir,
        #         device_map=self.device
        #     )
        
        self.model.eval()
            
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_id, cache_dir=cache_dir
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        

    def encode(self, sentences: List[str], task_name: str, prompt_type: PromptType | None = None,  batch_size: int = 32, **kwargs) -> NDArray:
        # check model layers device
        # for name, param in self.model.named_parameters():
        #     print(f"{name:40s} -> {param.device}")
        
        # sent_encode = self.tokenizer(
        #     sentences,
        #     return_tensors="pt",
        #     padding=True,
        #     truncation=True,
        #     max_length=1024,
        #     return_token_type_ids=False,
        #       # Ensure tensors are on the correct device
        # )
        # sent_encode = {k: v.to(self.model.device) for k, v in sent_encode.items()}
        
        
        
        # with torch.no_grad():
        #     output = self.model(
        #         **sent_encode, output_hidden_states=True, use_cache=False
        #     )

        # embeddings = output.hidden_states[-1]

        # model_embedding = mean_pooling(embeddings, sent_encode["attention_mask"])
        # return model_embedding.cpu().numpy()
        all_embeddings = []
        # Iterate over sentences in batches
        for i in range(0, len(sentences), batch_size):
            batch_sentences = sentences[i:i + batch_size]

            # Tokenize the current batch of sentences
            encoded_input = self.tokenizer(
                batch_sentences,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024, # Maximum sequence length (adjust if OpenELM has a different limit)
                return_token_type_ids=False, # Often not needed for causal LMs
            )
            
            # Manually move input tensors to the model's device
            encoded_input = {k: v.to(self.model.device) for k, v in encoded_input.items()}

            with torch.no_grad():
                # Pass the inputs to the model
                output = self.model(
                    **encoded_input,
                    output_hidden_states=True, # Ensure hidden states are returned
                    use_cache=False, # Disable cache for inference if not explicitly needed
                    **kwargs # Pass through any extra kwargs
                )

            # Extract the last hidden states for pooling
            embeddings = output.hidden_states[-1]

            # Perform mean pooling using the helper function
            attention_mask_batch = encoded_input["attention_mask"] # Use the attention mask from the batch
            model_embedding_batch = mean_pooling(embeddings, attention_mask_batch)

            # Move batch embeddings to CPU and convert to NumPy array, then append
            all_embeddings.append(model_embedding_batch.cpu().numpy())

        # Concatenate all batch embeddings into a single NumPy array
        return np.concatenate(all_embeddings, axis=0)

        

    
    
class OLMoEncoder(AutoModelForCausalLMEncoder):
    """OLMo Encoder"""

    def __init__(self, device: bool = False, **kwargs):
        shard_model_pt = full_path("/data/shared/olmo/OLMo-7B_shard_size_2GB")
        super().__init__(shard_model_pt=shard_model_pt, device=device)
    
    def encode(self, sentences: List[str], task_name: str, prompt_type: PromptType | None = None,  batch_size: int = 32, **kwargs) -> NDArray:
        
        all_embeddings = []
        # Iterate over sentences in batches
        for i in range(0, len(sentences), batch_size):
            batch_sentences = sentences[i:i + batch_size]

            # Tokenize the current batch of sentences
            encoded_input = self.tokenizer(
                batch_sentences,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024, # Maximum sequence length (adjust if OpenELM has a different limit)
                return_token_type_ids=False, # Often not needed for causal LMs
            )
            
            # Manually move input tensors to the model's device
            encoded_input = {k: v.to(self.model.device) for k, v in encoded_input.items()}

            with torch.no_grad():
                # Pass the inputs to the model
                output = self.model(
                    **encoded_input,
                    output_hidden_states=True, # Ensure hidden states are returned
                    use_cache=False, # Disable cache for inference if not explicitly needed
                    **kwargs # Pass through any extra kwargs
                )

            # Extract the last hidden states for pooling
            embeddings = output.hidden_states[-1]

            # Perform mean pooling using the helper function
            attention_mask_batch = encoded_input["attention_mask"] # Use the attention mask from the batch
            model_embedding_batch = mean_pooling(embeddings, attention_mask_batch)

            # Move batch embeddings to CPU and convert to NumPy array, then append
            all_embeddings.append(model_embedding_batch.cpu().numpy())

        # Concatenate all batch embeddings into a single NumPy array
        return np.concatenate(all_embeddings, axis=0)
        
    
if __name__ == "__main__":
    
    # model = OpenELMEncoder(device="auto",cache_dir="/data/shared/")
    model = OLMoEncoder(device="auto", cache_dir="/data/shared/")
    
    sentences = ["This is a test sentence.", "Here is another one."] * 10
    embeddings = model.encode(sentences,task_name="",batch_size=10)
    print("Embeddings shape:", embeddings.shape)