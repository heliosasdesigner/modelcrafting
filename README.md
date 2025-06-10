# ModelCrafting: Learning GPT from Scratch

This project is a hands-on implementation of language models, starting from the basics and building up to more complex architectures. It's based on Andrej Karpathy's excellent tutorial series "Let's build GPT: from scratch, in code, spelled out" (https://www.youtube.com/watch?v=kCc8FmEb1nY).

## 🎯 Project Structure

The project follows a progressive learning path:

1. **Bigram Language Model** (`nanogpt/bigram.py`)
   - Basic character-level language model
   - Token embeddings and positional embeddings
   - Simple training loop with evaluation
   - Text generation capabilities

## 🚀 Getting Started

### Prerequisites

- Python 3.x
- PyTorch
- CUDA (optional, for GPU acceleration)

### Installation

1. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install torch
   ```

3. **Prepare the dataset:**
   - Place your training text file in the `data` directory
   - The current implementation uses `tiny_shakespeare.txt`

## 📚 Implementation Details

### Bigram Model (`nanogpt/bigram.py`)

The bigram model is our starting point, implementing:
- Character-level tokenization
- Token and positional embeddings
- Simple forward pass with cross-entropy loss
- Text generation with temperature sampling

Key components:
- `BigramLanguageModel`: The main model class
- `get_batch`: Data loader for training
- `estimate_loss`: Evaluation function
- `train_model`: Training loop implementation

## 🎓 Learning Path

This project follows the learning path from Karpathy's video series:

1. **Basics of Language Modeling**
   - Character-level tokenization
   - Train/validation splits
   - Simple bigram model

2. **Self-Attention Mechanism** (Coming Soon)
   - Matrix multiplication for weighted aggregation
   - Scaled attention
   - Multi-head attention

3. **Transformer Architecture** (Coming Soon)
   - Encoder/Decoder blocks
   - Feed-forward networks
   - Residual connections
   - Layer normalization

## 🔧 Usage

### Training the Bigram Model

```python
from nanogpt.bigram import create_model, train_model

# Create model
model = create_model(device="cuda" if torch.cuda.is_available() else "cpu")

# Train model
model, training_records = train_model(
    model=model,
    train_data=train_data,
    val_data=val_data,
    max_iters=3000,
    eval_interval=300,
    eval_iters=200,
    block_size=8,
    batch_size=32,
    learning_rate=1e-2
)
```

### Generating Text

```python
# Generate text
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated = model.generate(context, max_new_tokens=100)
print(decode(generated[0].tolist()))
```

## 📚 References

- [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- [Attention is All You Need paper](https://arxiv.org/abs/1706.03762)
- [OpenAI GPT-3 paper](https://arxiv.org/abs/2005.14165)
- [nanoGPT repository](https://github.com/karpathy/nanoGPT)

## 🎯 Next Steps

1. Implement multi-head self-attention
2. Add transformer blocks
3. Scale up the model architecture
4. Add more sophisticated training features
5. Implement proper model checkpointing

## 📝 Notes

- The current implementation is focused on educational purposes
- The model is trained on a small dataset for demonstration
- GPU acceleration is recommended for faster training
- The code includes detailed comments explaining each component

