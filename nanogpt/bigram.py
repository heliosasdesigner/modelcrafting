import torch
import torch.nn as nn
import torch.nn.functional as F


batch_size = 32  # How many independent sequences will we process in parallel?
block_size = 8  # What is the maximum context length for predictions?
max_iters = 3000  # How many iterations to train for?
eval_interval = 300  # How often to evaluate the loss?
learning_rate = 1e-2
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
n_embd = 32  # Number of embedding dimension

# ------------------------------------------------------------

torch.manual_seed(1337)
torch.cuda.manual_seed(1337)


# Import the data
with open("data/tiny_shakespeare.txt", "r", encoding="utf-8") as file:
    text = file.read()


# Here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Create a mapping from characters to integers
s_to_i = {ch: i for i, ch in enumerate(chars)}
i_to_s = {i: ch for i, ch in enumerate(chars)}
# The encode function takes a string and turns each character into its corresponding number, creating a list of numbers.
encode = lambda string: [s_to_i[character] for character in string]
# The decode function takes a list of numbers and turns each one back into its matching character, creating a string.
decode = lambda indices: "".join([i_to_s[ind] for ind in indices])

# ------------------------------------------------------------

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# ------------------------------------------------------------


# Data Loader
def get_batch(data, block_size, batch_size, split, train_data, val_data):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x, y


# ------------------------------------------------------------


# Estimate the loss
@torch.no_grad()
def estimate_loss(model, eval_iters, block_size, batch_size, train_data, val_data):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(None, block_size, batch_size, split, train_data, val_data)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def get_structure(self):
        return {
            "First Layer": "Token Embedding",
            "Second Layer": "Linear",
        }

    def forward(self, idx, targets=None):

        # idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx)  # (B,T,C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            print(
                f"idx shape: {idx.shape}, device: {idx.device}, type: {type(idx)},idx:{idx}"
            )
            logits, loss = self(idx)
            print(
                f"1.logits min: {logits.min()}, max: {logits.max()}, shape: {logits.shape}"
            )

            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            print(
                f"2.logits min: {logits.min()}, max: {logits.max()}, shape: {logits.shape}"
            )
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            print(
                f"3.probs min: {probs.min()}, max: {probs.max()}, shape: {probs.shape}"
            )
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            print(
                f"4.idx_next min: {idx_next.min()}, max: {idx_next.max()}, shape: {idx_next.shape}"
            )
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
            print(f"5.idx min: {idx.min()}, max: {idx.max()}, shape: {idx.shape}")
        return idx


def create_Bigram_model(device="cuda"):
    model = BigramLanguageModel()
    model = model.to(device)
    return model


def ensure_device(data, device):
    """Ensure data is on the specified device."""
    if data.device != device:
        return data.to(device)
    return data


def train_model(
    model,
    train_data,
    val_data,
    max_iters,
    eval_interval,
    eval_iters,
    block_size,
    batch_size,
    learning_rate,
):
    # Get the device from the model
    device = next(model.parameters()).device

    # Check model parameters device
    print("Model device check:")
    for name, param in model.named_parameters():
        print(f"Parameter {name} is on device: {param.device}")

    # Check and move input data to correct device
    print("\nInput data device check:")
    train_data = ensure_device(train_data, device)
    val_data = ensure_device(val_data, device)
    print(f"train_data device: {train_data.device}")
    print(f"val_data device: {val_data.device}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    training_records = []

    for iter in range(max_iters):
        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0:
            losses = estimate_loss(
                model, eval_iters, block_size, batch_size, train_data, val_data
            )
            # Convert losses to CPU and Python numbers
            train_loss = losses["train"].cpu().item()
            val_loss = losses["val"].cpu().item()
            training_records.append(
                {
                    "step": iter,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                }
            )

        # sample a batch of data
        xb, yb = get_batch(None, block_size, batch_size, "train", train_data, val_data)

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    return model, training_records


model = create_Bigram_model(device)

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
