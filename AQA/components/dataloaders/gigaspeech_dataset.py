import os
import torch
import torchaudio
from torch.utils.data import Dataset
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Any
from collections import Counter
from utils.glossary import normalize_word


@dataclass
class AudioTextBatch:
    """Container for a batch of audio and text data."""
    # Audio features
    audio: torch.Tensor  # Shape: [batch_size, channels, time]
    audio_lengths: torch.Tensor  # Shape: [batch_size]
    sample_rate: int
    
    # Text features
    input_ids: torch.Tensor  # Shape: [batch_size, max_seq_length]
    attention_mask: torch.Tensor  # Shape: [batch_size, max_seq_length]
    
    # Metadata
    segment_ids: List[str]
    raw_text: List[str]
    
    def to(self, device):
        """Move batch tensors to specified device"""
        return AudioTextBatch(
            audio=self.audio.to(device),
            audio_lengths=self.audio_lengths.to(device),
            sample_rate=self.sample_rate,
            input_ids=self.input_ids.to(device),
            attention_mask=self.attention_mask.to(device),
            segment_ids=self.segment_ids,
            raw_text=self.raw_text
        )


class WordTokenizer:
    def __init__(self, vocab_size=10000):
        # Special tokens
        self.PAD_token = 0
        self.CLS_token = 1
        self.UNK_token = 2
        self.vocab_size = vocab_size + 3
        
        # Initialize vocabulary dictionaries
        self.word_to_id = {
            '[PAD]': self.PAD_token,
            '[CLS]': self.CLS_token,
            '[UNK]': self.UNK_token
        }
        self.id_to_word = {
            self.PAD_token: '[PAD]',
            self.CLS_token: '[CLS]',
            self.UNK_token: '[UNK]'
        }
        
        # The original normalization code should be imported and used directly
        self.special_tokens = set(['[PAD]', '[CLS]', '[UNK]'])

    def process_text(self, text):
        if not text or not isinstance(text, str):
            return ""
            
        words = []
        current_phrase = []
        
        for word in text.split():
            if word in self.special_tokens:
                # Process any accumulated phrase first
                if current_phrase:
                    normalized = normalize_word(" ".join(current_phrase))
                    if normalized:
                        words.extend(normalized.split())
                    current_phrase = []
                words.append(word)
            else:
                current_phrase.append(word)
        
        # Process any remaining phrase
        if current_phrase:
            normalized = normalize_word(" ".join(current_phrase))
            if normalized:
                words.extend(normalized.split())
                
        return " ".join(words)

    def build_vocab(self, texts):
        word_counts = Counter()
        
        for text in texts:
            processed_text = self.process_text(text)
            words = processed_text.split()
            word_counts.update(words)
        
        # Filter out special tokens from counts
        for token in self.special_tokens:
            word_counts.pop(token, None)
            
        # Select most common words
        common_words = sorted(word_counts.items(), key=lambda x: (-x[1], x[0]))
        selected_words = common_words[:self.vocab_size - len(self.special_tokens)]
        
        # Add words to vocabulary
        for idx, (word, _) in enumerate(selected_words):
            token_id = idx + len(self.special_tokens)
            self.word_to_id[word] = token_id
            self.id_to_word[token_id] = word
                
        print(f"Vocabulary size: {len(self.word_to_id)} words")

    def encode(self, text, max_length=None):
        if not text or not isinstance(text, str):
            return {
                'input_ids': torch.tensor([self.CLS_token], dtype=torch.long),
                'attention_mask': torch.tensor([1], dtype=torch.long)
            }
        
        processed_text = self.process_text(text)
        words = processed_text.split()
        
        # Start with CLS token
        token_ids = [self.CLS_token]
        
        # Truncate words if max_length provided
        if max_length is not None:
            words = words[:max_length - 1]  # Reserve one position for CLS
        
        # Convert words to token IDs
        for word in words:
            token_id = self.word_to_id.get(word, self.UNK_token)
            token_ids.append(token_id)
        
        # Create attention mask
        attention_mask = [1] * len(token_ids)
        
        return {
            'input_ids': torch.tensor(token_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
        }
    
    def decode(self, token_ids):
        """
        Convert token IDs back to text.
        
        Args:
            token_ids: List or tensor of token IDs
            
        Returns:
            Decoded text string
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
            
        words = []
        for token_id in token_ids:
            word = self.id_to_word.get(token_id)
            if word and word not in self.special_tokens:
                words.append(word)
                
        return " ".join(words)


class GigaSpeechDataset(Dataset):
    """GigaSpeech dataset with simplified tokenization"""
    
    def __init__(
        self,
        root_dir: str,
        subset: str = "xs",
        transform: Optional[Any] = None,
        max_text_length: int = 256,
        vocab_size: int = 10000
    ):
        """
        Initialize the dataset.
        
        Args:
            root_dir: Root directory containing the downloaded dataset
            subset: Which subset to load ('xs', 's', 'm', 'l', 'xl', 'dev', 'test')
            transform: Optional transform to be applied to the audio
            max_text_length: Maximum length for text sequences (in words)
            vocab_size: Size of vocabulary for the tokenizer
        """
        self.root_dir = root_dir
        self.subset = subset
        self.transform = transform
        self.max_text_length = max_text_length
        
        # Load metadata
        print("Loading metadata...")
        self.metadata = self._load_metadata()
        
        # Initialize and train tokenizer
        print("Building vocabulary...")
        self.tokenizer = WordTokenizer(vocab_size=vocab_size)
        self.tokenizer.build_vocab(self.metadata['text'].tolist())
        
        # Create category mapping
        self.category_to_idx = {
            cat: idx for idx, cat in enumerate(sorted(self.metadata['category'].unique()))
        }
    
    def _load_metadata(self):
        """Load and process metadata"""
        meta_dir = os.path.join(self.root_dir, "metadata")
        
        dfs = []
        for file in os.listdir(meta_dir):
            if file.startswith(f"{self.subset}_chunks_") and file.endswith("_metadata.csv"):
                df = pd.read_csv(os.path.join(meta_dir, file))
                dfs.append(df)
        
        df = pd.concat(dfs, ignore_index=True)
        df = df.rename(columns={
            'sid': 'segment_id',
            'aid': 'audio_id',
            'text_tn': 'text',
            'path': 'original_full_path'
        })
        
        return df
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        """Get a single example from the dataset"""
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Get metadata for this example
        row = self.metadata.iloc[idx]
        
        # Construct audio path and load audio
        audio_path = os.path.join(
            self.root_dir,
            "audio",
            f"{self.subset}_chunks_{int(row['segment_id'].split('_')[1]):04d}",
            f"{row['segment_id']}.wav"
        )
        
        waveform, sample_rate = torchaudio.load(audio_path)
        if self.transform:
            waveform = self.transform(waveform)
        
        # Process text
        text_features = self.tokenizer.encode(
            row['text'], 
            max_length=self.max_text_length
        )
        
        return {
            'audio': waveform,
            'audio_length': waveform.shape[1],
            'sample_rate': sample_rate,
            'text_features': text_features,
            'raw_text': row['text'],
            'segment_id': row['segment_id'],
            'speaker': row['speaker'],
            'category': self.category_to_idx[row['category']],
            'source': row['source']
        }


def collate_fn(batch: List[Dict]) -> AudioTextBatch:
    """
    Custom collate function to handle batching of audio and text.
    Pads sequences to the maximum length in the batch.
    """
    # Get max lengths for padding
    max_audio_length = max(s['audio_length'] for s in batch)
    max_text_length = max(len(s['text_features']['input_ids']) for s in batch)
    
    # Initialize tensors
    batch_size = len(batch)
    audio_lengths = torch.zeros(batch_size, dtype=torch.long)
    padded_audio = torch.zeros(batch_size, batch[0]['audio'].shape[0], max_audio_length)
    
    # Text tensors
    input_ids = torch.full((batch_size, max_text_length), 0, dtype=torch.long)  # 0 is PAD token
    attention_mask = torch.zeros(batch_size, max_text_length, dtype=torch.long)
    
    # Metadata lists
    segment_ids = []
    raw_texts = []
    
    # Fill in batch tensors
    for i, sample in enumerate(batch):
        # Handle audio
        audio = sample['audio']
        audio_length = sample['audio_length']
        padded_audio[i, :, :audio_length] = audio
        audio_lengths[i] = audio_length
        
        # Handle text
        text_features = sample['text_features']
        text_length = len(text_features['input_ids'])
        
        input_ids[i, :text_length] = text_features['input_ids']
        attention_mask[i, :text_length] = text_features['attention_mask']
        
        # Store metadata
        segment_ids.append(sample['segment_id'])
        raw_texts.append(sample['raw_text'])
    
    return AudioTextBatch(
        audio=padded_audio,
        audio_lengths=audio_lengths,
        sample_rate=batch[0]['sample_rate'],
        input_ids=input_ids,
        attention_mask=attention_mask,
        segment_ids=segment_ids,
        raw_text=raw_texts
    )