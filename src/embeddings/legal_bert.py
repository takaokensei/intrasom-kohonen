import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from typing import List, Union

class LegalBertEmbedder:
    def __init__(self, model_name: str = 'nlpaueb/legal-bert-base-uncased', max_length: int = 512, device: str = None):
        self.model_name = model_name
        self.max_length = max_length
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def get_sentence_embedding_dimension(self) -> int:
        return self.model.config.hidden_size

    def encode(self, sentences: Union[str, List[str]], batch_size: int = 32, show_progress_bar: bool = False, normalize_embeddings: bool = True, convert_to_numpy: bool = True) -> np.ndarray:
        if isinstance(sentences, str): sentences = [sentences]
        all_embs = []
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i : i + batch_size]
            encoded = self.tokenizer(batch, padding=True, truncation=True, max_length=self.max_length, return_tensors='pt').to(self.device)
            with torch.no_grad():
                outputs = self.model(**encoded)
                attention_mask = encoded['attention_mask'].unsqueeze(-1)
                token_embeddings = outputs.last_hidden_state
                input_mask_expanded = attention_mask.expand(token_embeddings.size()).float()
                sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                batch_embs = sum_embeddings / sum_mask
                if normalize_embeddings: batch_embs = torch.nn.functional.normalize(batch_embs, p=2, dim=1)
                all_embs.append(batch_embs.cpu().numpy())
        res = np.vstack(all_embs)
        return res if convert_to_numpy else torch.from_numpy(res)
