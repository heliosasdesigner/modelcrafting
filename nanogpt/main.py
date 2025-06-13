import torch
import streamlit as st
import pandas as pd
import altair as alt

# Set page configuration for wider layout
st.set_page_config(
    page_title="NanoGPT", layout="wide", initial_sidebar_state="expanded"
)

from bigram import (
    BigramLanguageModel,
    get_batch,
    estimate_loss,
    create_Bigram_model,
    train_model,
    batch_size,
    block_size,
    max_iters,
    eval_interval,
    learning_rate,
    eval_iters,
    n_embd,
)

st.title("NanoGPT")

st.write("This is a replicate project of NanoGPT.")
youtube_url = "https://www.youtube.com/watch?v=kCc8FmEb1nY"
st.write(
    f"Source: [Let's build GPT: from scratch, in code, spelled out.]({youtube_url})"
)

st.divider()


tab1, tab2 = st.tabs(["Bigram Language Model", "Get into GPT"])

with tab1:
    st.subheader("Bigram Language Model")
    st.write(
        "Bigram Language Model is a simple language model that predicts the next token based on the current token."
    )
    st.markdown("#### Let's start with the data.")
    # open the data file
    st.write("Open the data file, preview the first 1000 characters:")
    with open("data/tiny_shakespeare.txt", "r", encoding="utf-8") as file:
        text = file.read()
    with st.container(border=True):
        # print the first 1000 characters
        st.write(text[:1000])

    col1, col2 = st.columns(2)

    with col1:
        st.write("Total number of characters:")
        st.markdown(f"**{len(text)}**")

    with col2:
        st.write("Total number of unique characters:")
        st.markdown(f"**{len(set(text))}**")

    st.markdown("#### Unique Characters occur in the data:")
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    col1, col2 = st.columns(2)
    with col2:
        st.write("Vocabulary size:")
        st.markdown(f"**{vocab_size}**")
    with col1:
        st.write("Vocabulary:")
        st.markdown(chars)

    # Tokenize the data
    st.divider()
    st.markdown("#### Tokenize the data:")

    # Create a mapping from characters to integers
    st.write("Create a mapping from characters to integers:")
    s_to_i = {ch: i for i, ch in enumerate(chars)}
    i_to_s = {i: ch for i, ch in enumerate(chars)}

    # Encode the data
    # The very simple Tokenizer
    # The encode function takes a string and turns each character into its corresponding number, creating a list of numbers.
    encode = lambda string: [s_to_i[character] for character in string]
    # The decode function takes a list of numbers and turns each one back into its matching character, creating a string.
    decode = lambda indices: "".join([i_to_s[ind] for ind in indices])
    # Encode the data
    data = encode(text)
    col1, col2 = st.columns(2)
    with col1:
        st.write("Encoded data:")
        st.write(data[:10])
    with col2:
        st.write("Decoded data:")
        st.write(decode(data[:10]))

    st.divider()

    # Import the torch library
    st.markdown("#### Import the torch library")

    # torch.long means a 64-bit integer (int64)
    data = torch.tensor(encode(text), dtype=torch.long)
    col1, col2 = st.columns(2)
    with col1:
        st.write("Data Shape:")
        st.write(data.shape)
    with col2:
        st.write("Data Type:")
        st.write(data.dtype)

    st.write("Data Preview:")
    st.write(data[:100])

    st.markdown("##### Split the data into training and validation sets")

    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    col1, col2 = st.columns(2)
    with col1:
        st.write("Training data:")
        st.write(train_data[:100])
        st.write(f"Decoding:\n {decode(train_data[:100].tolist())}")
    with col2:
        st.write("Validation data:")
        st.write(val_data[:100])
        st.write(f"Decoding:\n {decode(val_data[:100].tolist())}")

    st.divider()
    st.markdown("#### Create the target data")

    # Define the block size ( or context length)
    block_size = 8
    st.markdown(
        f"""
        <div style="
        font-size: 1.2rem;
        ">
            <p>Define the block size ( or context length ): <strong>{block_size}</strong> with offset by 1</p>
        """,
        unsafe_allow_html=True,
    )

    # offset by 1
    x = train_data[:block_size]
    y = train_data[1 : block_size + 1]

    with st.container(border=True):

        for t in range(block_size):
            context = x[: t + 1]
            target = y[t]
            st.write(f"When input is {context} the target is {target}")

    st.markdown(
        f"""
        <div style="
            padding: 1rem;
            border-radius: 0.5rem;
            background-color: #222;
            border-left: 4px solid #4CAF50;
            margin: 1rem 0;
        ">
            <h4>📝 Key Concepts</h4>
            <h5>Content Window (Context Length, ctx_len)</h5>
            <ul>
                <li><strong>Definition</strong>: The maximum number of tokens the model "sees" at once during training or inference.</li>
                <li><strong>Typical variable names</strong>: context_window, context_length, ctx_len (in config).</li>
                <li><strong>Example</strong>: If your context_window is 128, each training sample or prediction only attends to 128 tokens at a time.</li>
            </ul>
            <h5>Block Size</h5>
            <ul>
                <li><strong>Definition</strong>: Usually refers to the length of token sequences processed in one forward/backward pass during training.</li>
                <li><strong>Typical usage</strong>: In code, block_size often matches the context length (ctx_len), especially in GPT-style models.</li>
            </ul>
            
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()
    st.markdown("#### Batching the data")

    torch.manual_seed(1337)

    batch_size = 4  # how many independent sequences will we process in parallel?
    block_size = 8  # what is the maximum context length for predictions?

    # Helper function to get a batch of data
    def get_batch(split):
        data = train_data if split == "train" else val_data
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i : i + block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
        return x, y

    xb, yb = get_batch("train")
    col1, col2 = st.columns(2)
    with col1:
        st.write("inputs:")
        st.write(f"xb shape: **{xb.shape}**")
        st.write(f"xb: **{xb}**")
    with col2:
        st.write("targets:")
        st.write(f"yb shape: **{yb.shape}**")
        st.write(f"yb: **{yb}**")

    # Create a container for batch information
    with st.container():
        st.write("**Batch Information:**")
        for b in range(batch_size):
            with st.expander(f"Batch {b}"):
                for t in range(block_size):
                    context = xb[b, : t + 1]
                    target = yb[b, t]
                    st.write(f"When input is {context.tolist()} the target is {target}")

    st.divider()
    st.markdown("#### Let's try to input the data into Bigram Language Model")

    import torch.nn as nn
    from torch.nn import functional as F

    torch.manual_seed(1337)

    st.write(
        f"Variables: B: batch_size = {batch_size}, T: context_length = {block_size}, C: vocab_size = {vocab_size}"
    )
    st.write(
        f"The current Logits shape: **torch.Size([{batch_size}, {block_size}, {vocab_size}])**, B x T x C"
    )
    st.write(
        f"and because Pytorch expects the channel dimension to be the second dimension, we need to reshape the logits and targets (B*T,C) -> torch.Size([{batch_size * block_size}, {vocab_size}])"
    )
    st.write("Expected Loss in first batch: -log(1/C) = -log(1/65) = 4.17")

    st.markdown(
        f"""
                <div style="
                    padding: 1rem;
                    border-radius: 0.5rem;
                    background-color: #222;
                    border-left: 4px solid #4CAF50;
                    margin: 1rem 0;
                ">
                    <h4>📊 Understanding Logits</h4>
                    <p>Logits are the raw, unnormalized predictions from the model before any activation function is applied. In this Bigram Language Model:</p>
                    <ul>
                        <li><strong>Shape</strong>: (B,T,C) where:
                            <ul>
                                <li>B = Batch size (sequences processed in parallel)</li>
                                <li>T = Time steps (context length)</li>
                                <li>C = Vocabulary size (possible tokens)</li>
                            </ul>
                        </li>
                        <li><strong>Meaning</strong>: Each value represents the model's raw score for how likely a token is to come next</li>
                        <li><strong>Example</strong>: If logits are [2.1, -1.3, 0.5] for tokens [A,B,C], it means:
                            <ul>
                                <li>Token A has highest likelihood (2.1)</li>
                                <li>Token B has lowest likelihood (-1.3)</li>
                                <li>Token C has medium likelihood (0.5)</li>
                            </ul>
                        </li>
                    </ul>
                </div>
                """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
                <div style="
                    padding: 1rem;
                    border-radius: 0.5rem;
                    background-color: #222;
                    border-left: 4px solid #4CAF50;
                    margin: 1rem 0;
                ">
                    <h4>🤖 Bigram Language Model</h4>
                    <p>A simple but powerful language model that predicts the next token based on the current token:</p>
                    <ul>
                        <li><strong>Core Components</strong>:
                            <ul>
                                <li>Token Embedding Table: Maps each token to a vector of logits</li>
                                <li>Forward Pass: Processes input tokens to predict next tokens</li>
                                <li>Generation: Creates new text by sampling from predictions</li>
                            </ul>
                        </li>
                        <li><strong>How it Works</strong>:
                            <ul>
                                <li>Input: Takes a sequence of tokens (B,T) where B=batch size, T=context length</li>
                                <li>Embedding: Converts tokens to logits using the embedding table</li>
                                <li>Prediction: Uses these logits to predict the next token</li>
                                <li>Loss: Calculates cross-entropy loss between predictions and actual next tokens</li>
                            </ul>
                        </li>
                        <li><strong>Generation Process</strong>:
                            <ul>
                                <li>Starts with an initial context</li>
                                <li>Predicts next token using softmax probabilities</li>
                                <li>Samples from these probabilities</li>
                                <li>Appends new token and repeats</li>
                            </ul>
                        </li>
                    </ul>
                </div>
                """,
        unsafe_allow_html=True,
    )

    class BigramLanguageModelSimple(nn.Module):

        def __init__(self, vocab_size):
            super().__init__()
            # Each token directly reads off the logits for the next token from a lookup table
            self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

        def forward(self, idx, targets=None):

            # idx and targets are both (B,T) where B is batch size and T is context length
            logits = self.token_embedding_table(
                idx
            )  # (B,T,C), in this case C is vocab_size

            if targets is None:
                loss = None
            else:

                # (B,T,C) -> (B*T,C)
                B, T, C = logits.shape
                logits = logits.view(B * T, C)
                # (B,T) -> (B*T)
                targets = targets.view(B * T)

                loss = F.cross_entropy(logits, targets)

            return logits, loss

        def generate(self, idx, max_new_tokens):
            # idx is (B,T) array of indices in the current context
            for _ in range(max_new_tokens):
                # Get the Predictions
                logits, loss = self(idx)
                # focus only on the last time step
                logits = logits[:, -1, :]  # becomes (B,C)
                # apply softmax to get probabilities
                probs = F.softmax(logits, dim=-1)  # (B,C)
                # sample from the distribution
                idx_next = torch.multinomial(probs, num_samples=1)  # (B,1)
                # append sampled index to the running sequence
                idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
            return idx

    m = BigramLanguageModelSimple(vocab_size)
    out, loss = m(xb, yb)

    st.write("##### Output ")
    st.write(f"Output shape: **{out.shape}**")
    st.write(f"Loss: **{loss.item() if loss is not None else None}**")

    st.write("##### Generate the text")

    # Move idx to the same device for generation
    idx = torch.zeros((1, 1), dtype=torch.long, device=out.device)
    max_new_tokens = 100
    st.write(
        f"To Kick off the generation, we need to provide an initial idx: **{idx}** with max_new_tokens: **{max_new_tokens}**"
    )

    st.write(f"Generated text: ")
    st.write(
        f"{decode(m.generate(idx, max_new_tokens=max_new_tokens)[0].cpu().tolist())}"
    )

    st.divider()
    st.write("#### Let's create the PyTorch optimizer")
    optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)
    st.write(f"Optimizer:")
    st.write(f" **{optimizer}**")

    batch_size = 32
    training_steps = 1000
    st.write(f"Batch size: **{batch_size}**")

    training_records = []
    with st.container():
        with st.expander("Training Records"):
            for steps in range(training_steps):

                # sample a batch of data
                xb, yb = get_batch("train")
                # evaluate the loss
                logits, loss = m(xb, yb)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                if steps % 5 == 0:
                    training_records.append({"step": steps, "loss": loss.item()})
                    st.write(f"Step {steps} loss: {loss.item()}")
                    print(f"Step {steps} loss: {loss.item()}")

    st.write("**Training Progress:**")

    # Create a line chart with axis labels and limits
    import pandas as pd

    example_df = pd.DataFrame(training_records)

    # Create Altair chart with axis limits
    chart_for_example = (
        alt.Chart(example_df)
        .mark_line()
        .encode(
            x=alt.X("step", title="Step", scale=alt.Scale(domain=[0, training_steps])),
            y=alt.Y(
                "loss", title="Loss", scale=alt.Scale(domain=[0, 8])
            ),  # Adjust these limits based on your loss range
        )
        .properties(width="container", height=400)
    )

    st.altair_chart(chart_for_example, use_container_width=True)

    # Add a description
    st.caption(
        "Training loss over time - lower values indicate better model performance"
    )

    st.write("##### Output")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Steps number:")
        st.write(f"**{training_steps}**")
        st.write("Loss:")
        st.write(loss.item())
    with col2:
        st.write("Generate the text")
        st.write(f"{decode(m.generate(idx, max_new_tokens=500)[0].cpu().tolist())}")

    # ------------------------------------------------------------------
    st.divider()
    st.markdown("### Let's put one more layer in the model")

    # Hyperparameters
    batch_size = 32  # How many independent sequences will we process in parallel?
    block_size = 8  # What is the maximum context length for predictions?
    max_iters = 3000  # How many iterations to train for?
    eval_interval = 300  # How often to evaluate the loss?
    learning_rate = 1e-2
    device = "cuda" if torch.cuda.is_available() else "cpu"
    eval_iters = 200
    n_embd = 32  # Number of embedding dimension

    # import all the functions from bigram.py
    from bigram import (
        BigramLanguageModel,
        get_batch,
        estimate_loss,
        create_Bigram_model,
        train_model,
    )

    # create the model
    model = create_Bigram_model()

    st.markdown("#### Model Configuration")
    col1, col2 = st.columns(2)
    with col1:
        st.write("##### Hyperparameters")
        st.write(f"Batch size: {batch_size}")
        st.write(f"Block size: {block_size}")
        st.write(f"Max iters: {max_iters}")
        st.write(f"Eval interval: {eval_interval}")
        st.write(f"Learning rate: {learning_rate}")
        st.write(f"Device: {device}")
        st.write(f"Eval iters: {eval_iters}")
    with col2:
        # Print the model structure
        st.write("##### Model Structure")
        st.write(model.get_structure())

    st.markdown("#### Initial Model Evaluation")
    # Move input tensors to the same device as the model
    xb = xb.to(device)
    yb = yb.to(device)

    out, loss = model(xb, yb)

    st.write(f"Output shape: **{out.shape}**")
    st.write(f"Loss: **{loss.item() if loss is not None else None}**")

    # Move idx to the same device for generation
    idx = torch.zeros((1, 1), dtype=torch.long, device=device)
    max_new_tokens = 100

    st.write(f"Initial idx: **{idx}** with max_new_tokens: **{max_new_tokens}**")

    st.write(f"Generated text: ")
    generated = model.generate(idx, max_new_tokens=max_new_tokens)
    st.write(f"Generated shape: **{generated.shape}**")
    st.write(f"Generated: **{generated}**")
    st.write(
        f"{decode(model.generate(idx, max_new_tokens=max_new_tokens)[0].cpu().tolist())}"
    )

    # ------------------------------------------------------------------

    st.markdown("### Output")
    model, training_records = train_model(
        model,
        train_data,
        val_data,
        max_iters,
        eval_interval,
        eval_iters,
        block_size,
        batch_size,
        learning_rate,
    )

    # st.markdown("#### Training Results")
    # Convert training records to DataFrame
    training_records_df = pd.DataFrame(training_records)

    # Create Altair chart with axis limits
    chart_for_training_records = (
        alt.Chart(training_records_df)
        .mark_line()
        .encode(
            x=alt.X("step", title="Step", scale=alt.Scale(domain=[0, max_iters])),
            y=alt.Y(
                "train_loss", title="Training Loss", scale=alt.Scale(domain=[0, 8])
            ),
        )
        .properties(width="container", height=400)
    )

    # Add validation loss line
    val_line = (
        alt.Chart(training_records_df)
        .mark_line(color="red")
        .encode(
            x="step",
            y="val_loss",
        )
    )

    # Combine the charts
    final_chart = chart_for_training_records + val_line

    st.altair_chart(final_chart, use_container_width=True)

    # Add a description
    st.caption(
        "Training and validation loss over time - lower values indicate better model performance"
    )

    col1, col2 = st.columns(2)
    with col1:
        st.write("Steps number:")
        st.write(f"**{max_iters}**")
        st.write("Loss:")
        st.write(loss.item())
    with col2:
        st.write("Generate the text")
        st.write(f"{decode(model.generate(idx, max_new_tokens=500)[0].cpu().tolist())}")

    # ------------------------------------------------------------------
    st.divider()
with tab2:
    st.markdown("## Let's get into the Generative Pre-trained Transformer (GPT)")

    st.markdown("### The Mathematical Trick in self-attention")

    # consider the following toy example
    torch.manual_seed(42)
    B, T, C = 4, 8, 2
    x = torch.randn(B, T, C)

    # print the shape of the tensor
    st.write(f"x shape: **{x.shape}**")

    st.markdown(
        """
                ##### We want x[b,t] = mean_{i<=t} x[b,i]
                ```python 
                xbow = torch.zeros((B, T, C))
                for b in range(B):
                    for t in range(T):
                        xprev = x[b,:t+1] #(t,C)
                        xbow[b,t] = torch.mean(xprev, 0)
                ```
                """
    )
    xbow = torch.zeros((B, T, C))
    for b in range(B):
        for t in range(T):
            xprev = x[b, : t + 1]  # (t,C)
            xbow[b, t] = torch.mean(xprev, 0)

    st.write(
        "The value in the xbow(bag of words) is the mean of the previous tokens. so the -0.0894 in xbow[0][1] is the mean of 0.1808 and -0.3596 in the x[0][0] and 0.0000 in the x[0][1]"
    )
    col1, col2 = st.columns(2)
    with col1:
        st.write("x[0]:")
        st.write(x[0])
    with col2:
        st.write("xbow[0]:")
        st.write(xbow[0])

    st.write(
        "but this is not efficient, we can use the matrix multiplication to get the same result"
    )
    st.subheader("Let's introduce the masked matrix by `torch.tril`")
    st.markdown(
        """
        ```python
        torch.manual_seed(1337)
        a = torch.ones(3,3)
        b = torch.randint(0,10,(3,2)).float()
        c = a @ b
        
        ```
        """
    )
    a = torch.ones(3, 3)
    b = torch.randint(0, 10, (3, 2)).float()
    c = a @ b

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.write("a:")
        st.write(a)
    with col_b:
        st.write("b:")
        st.write(b)
    with col_c:
        st.write("c:")
        st.write(c)

    st.write("If we using torch.tril to mask the upper triangle of the matrix:")

    st.markdown(
        """
        ```python
        torch.manual_seed(1337)
        a = torch.tril(torch.ones(3,3))
        a = a / torch.sum(a,1,keepdim=True) <- to get the mean of the previous tokens
        b = torch.randint(0,10,(3,2)).float()
        c = a @ b
        
        ```
        """
    )
    a = torch.tril(torch.ones(3, 3))
    a = a / torch.sum(a, 1, keepdim=True)
    b = torch.randint(0, 10, (3, 2)).float()
    c = a @ b

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.write("a:")
        st.write(a)
    with col_b:
        st.write("b:")
        st.write(b)
    with col_c:
        st.write("c:")
        st.write(c)

    st.markdown("#### If we using torch.tril to mask the upper triangle of the matrix:")

    st.markdown(
        """
        ```python
        weights = torch.tril(torch.ones(T,T))
        weights = weights / torch.sum(weights,1,keepdim=True)
        
        xbow2 = weights @ x 
        ```
        """
    )
    weights = torch.tril(torch.ones(T, T))
    weights = weights / torch.sum(weights, 1, keepdim=True)
    st.write("weights:")
    st.write(weights)

    st.write(
        "Now we can use the masked matrix to get the same result as the previous example"
    )

    xbow2 = weights @ x  # (T, T) @ (B, T, C) -> (B, T, C)
    col1, col2 = st.columns(2)
    with col1:
        st.write("xbow2:")
        st.write(xbow2[0])
    with col2:
        st.write("xbow:")
        st.write(xbow[0])

    st.write(
        f"Checking by torch.allclose(xbow, xbow2): **{torch.allclose(xbow, xbow2)}**"
    )

    st.markdown("#### Softmax also did the job:")

    st.markdown(
        """
        ```python
        weights = torch.zeros((T,T))
        weights = torch.tril(weights)
        weights = weights.masked_fill(weights == 0, float('-inf'))
        weights = F.softmax(weights, dim=-1)

        ```
        """
    )

    source_weights = torch.ones((T, T))
    pre_weights1 = torch.tril(source_weights)
    pre_weights2 = pre_weights1.masked_fill(pre_weights1 == 0, float("-inf"))
    weights3 = F.softmax(pre_weights2, dim=-1)

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.write("Masked Matrix:")
        st.write(pre_weights1)
    with col_b:
        st.write("Converted to -inf:")
        st.write(pre_weights2)
    with col_c:
        st.write("Softmax:")
        st.write(weights3)

    xbow3 = weights3 @ x  # (T, T) @ (B, T, C) -> (B, T, C)
    col1, col2 = st.columns(2)
    with col1:
        st.write("xbow3:")
        st.write(xbow3[0])
    with col2:
        st.write("xbow2:")
        st.write(xbow2[0])

    st.write(
        f"Checking by torch.allclose(xbow2, xbow3): **{torch.allclose(xbow2, xbow3)}**"
    )

    st.markdown("#### Final Version: Self-Attention")

    st.markdown(
        """
        <div style="
            padding: 1.5rem;
            border-radius: 0.5rem;
            background-color: #1E1E1E;
            border-left: 4px solid #FFD700;
            margin: 1rem 0;
        ">
            <h3 style="color: #FFD700; margin-top: 0;">🧠 Understanding Self-Attention</h3>
            <h4 style="color: #4CAF50;">What is Self-Attention?</h4>
            <p>Self-attention is a mechanism that allows a model to weigh the importance of different words in a sequence when making predictions. Think of it as the model's ability to "pay attention" to different parts of the input when processing each word.</p>
            <h4 style="color: #4CAF50;">Key Components:</h4>
            <ul>
                <li><strong>Query (Q)</strong>: What we're looking for</li>
                <li><strong>Key (K)</strong>: What we're matching against</li>
                <li><strong>Value (V)</strong>: The actual content we want to retrieve</li>
            </ul>
            <h4 style="color: #4CAF50;">How it Works:</h4>
            <ol>
                <li>Each word gets transformed into three vectors: Query, Key, and Value</li>
                <li>Query vectors are compared with Key vectors to get attention scores</li>
                <li>Softmax is applied to get attention weights (probabilities)</li>
                <li>These weights are used to combine Value vectors</li>
            </ol>
            <h4 style="color: #4CAF50;">Mathematical Process:</h4>
            <p>1. Calculate attention scores: Q × K^T</p>
            <p>2. Apply softmax to get probabilities</p>
            <p>3. Weight and sum the values: softmax(Q × K^T) × V</p>
            <h4 style="color: #4CAF50;">Benefits:</h4>
            <ul>
                <li>Captures long-range dependencies</li>
                <li>Parallel processing of all words</li>
                <li>Dynamic weighting of importance</li>
                <li>No fixed window size limitation</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("Previous Version:")
    st.markdown(
        """
        ```python
        
        weights = torch.zeros((T,T)) <- we don't want the probability distribution is uniform
        weights = torch.tril(weights)
        weights = weights.masked_fill(weights == 0, float('-inf'))
        weights = F.softmax(weights, dim=1)

        ```
        """
    )
    st.write("Single Head Self-Attention:")

    st.markdown(
        """
        ```python
        torch.manual_seed(1337)
        B,T,C = 4,8,32 # batch, time-steps (sequence length), channels (embedding dim)
        x = torch.randn(B,T,C)
        
        # Single Head Self-Attention
                    similarity
        query(t) ─────────────► value(0...t)


        head_size = 16 # aka d_k
        key   = nn.Linear(C, head_size, bias=False)
        query = nn.Linear(C, head_size, bias=False)
        value = nn.Linear(C, head_size, bias=False)
        
        k = key(x) # (B,T,16)
        q = query(x) # (B,T,16)
        v = value(x) # (B,T,16)
        
        weights = q @ k.transpose(-2,-1) # (B,T,16) @ (B,16,T) -> (B,T,T)
        weights = weights * (head_size ** -0.5) # scale the weights
        tril = torch.tril(torch.ones(T, T))
        weights = weights.masked_fill(tril == 0, float('-inf'))
        weights = F.softmax(weights, dim=-1)
        
        # Times the weights by the value
        output = weights @ v # (B,T,T) @ (B,T,16) -> (B,T,16)

        ```
        """
    )
    st.markdown(
        """
        Why still pass `C`?
        - `nn.Linear(in_features=C, out_features=head_size)` says: "each token's C-dimensional embedding comes in, and I will project it to head_size." These three projections are learned and different – that's why we create three layers.

        `head_size`
        - The dimension per attention head (d_k in the paper). In multi-head attention the model splits the full embedding `(C = n_heads × head_size`) into several smaller subspaces, runs attention in parallel, then concatenates the heads back together. Here you are implementing one head with `d_k = 16`.

        `bias=False`
        - Removes the additive bias vector. It slightly simplifies the math and matches many reference implementations. (Nothing stops you from leaving `bias=True`; it usually does not change quality much.)
        
        """,
        unsafe_allow_html=True,
    )
    torch.manual_seed(1337)
    B, T, C = 4, 8, 32  # batch, time-steps (sequence length), channels (embedding dim)
    x = torch.randn(B, T, C)

    # Single Head Self-Attention
    head_size = 16
    key = nn.Linear(C, head_size, bias=False)
    query = nn.Linear(C, head_size, bias=False)
    value = nn.Linear(C, head_size, bias=False)

    k = key(x)  # (B,T,16)
    q = query(x)  # (B,T,16)
    v = value(x)  # (B,T,16)

    import math

    weights_1 = q @ k.transpose(-2, -1)  # (B,T,16) @ (B,16,T) -> (B,T,T)
    weights_2 = weights_1 * (head_size**-0.5)  # scale the weights
    tril = torch.tril(torch.ones(T, T))
    weights_3 = weights_2.masked_fill(tril == 0, float("-inf"))
    weights_4 = F.softmax(weights_3, dim=-1)

    output = weights_4 @ v  # (B,T,T) @ (B,T,16) -> (B,T,16)

    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        st.write("q @ k.transpose(-2, -1):")
        st.write(f"shape: {weights_1.shape}")
        st.write(weights_1[0])
    with col_b:
        st.write("scale:")
        st.write(f"shape: {weights_2.shape}")
        st.write(weights_2[0])
    with col_c:
        st.write("masked_fill:")
        st.write(f"shape: {weights_3.shape}")
        st.write(weights_3[0])
    with col_d:
        st.write("softmax:")
        st.write(f"shape: {weights_4.shape}")
        st.write(weights_4[0])

    st.write("output:")
    st.write(f"shape: {output.shape}")
    st.write(output[0])
    st.markdown(
        """
        #### Notes from [Andrej Karpathy's Notebook](https://colab.research.google.com/drive/1JMLa53HDuA-i7ZBmqV7ZnA3c_fvtXnx-?usp=sharing&authuser=2#scrollTo=h5hjCcLDr2WC):
        - **Attention is a communication mechanism. Can be seen as nodes in a directed graph looking at each other and aggregating information with a weighted sum from all nodes that point to them, with data-dependent weights.**
            - Attention = message-passing on a directed graph
            
            Think of the sequence positions (tokens, image patches, audio frames …) as nodes.
            For every query token tᵩ we compute weights to every key token tₖ.
            softmax turns those weights into a probability distribution, and the output for tᵩ is a weighted sum of the value vectors it "listens to".
            Graphically:

            ```
            t0  t1  t2  t3
            |↘ ↓  ↙|
            |  ↘↓↙ |
            t0' t1' t2' t3'   (primes are the new, context-aware vectors)
            ```
            Because the weights depend on the input itself, the graph is data-dependent and can change from batch to batch or even from token to token.
        - **There is no notion of space. Attention simply acts over a set of vectors. This is why we need to positionally encode tokens.**
            - Unlike a convolution, attention does not come with baked-in locality or a fixed coordinate system.
            All tokens are treated as an unordered set of vectors of length C.
            That's enormously flexible (you can connect any two positions), but the model now has no idea which token is first, second, ….
        - **Each example across batch dimension is of course processed completely independently and never "talk" to each other**
        - **In an "encoder" attention block just delete the single line that does masking with `tril`, allowing all tokens to communicate. This block here is called a "decoder" attention block because it has triangular masking, and is usually used in autoregressive settings, like language modeling.**
            - When do we want tokens to "talk to each other"?
            
                - **Analysis / understanding tasks**

                Sentiment analysis, question answering, image classification, protein-fold prediction …

                Full-context encoder attention captures long-distance cues (negation, object relationships, residue interactions).

                - **Generative tasks**

                Language modelling, audio generation, code completion …

                **Need causal masking to uphold the autoregressive factorisation**
                $$
                P(x₁,…,xₜ) = ∏ₜ P(xₜ | x₁…xₜ₋₁).
                $$

                - **Hybrid tasks**

                Translation, summarisation use an encoder (full context) feeding a decoder (causal) with an extra cross-attention bridge so the generated output can still consult the entire source.
        - **"self-attention" just means that the keys and values are produced from the same source as queries. In "cross-attention", the queries still get produced from x, but the keys and values come from some other, external source (e.g. an encoder module)**
            - Self-attention vs. cross-attention
            
            Self-attention Queries, keys, and values all originate from the same sequence (the encoder's input, or the decoder's own previous tokens).
            Cross-attention Queries come from one place, keys & values from another.
            Example (translation):
            ```
            Encoder:  src tokens ───► self-attention (no mask) ───► memory
            Decoder:  tgt tokens ───► masked self-attention ──┐
                                                            ↓  queries
                                cross-attention keys&values ◄─┘
                                ```
            Cross-attention lets each partially generated target word decide where to look in the source sentence.


        - **"Scaled" attention additional divides `weights` by `1/sqrt(head_size)`. This makes it so when input Q,K are unit variance, weights will be unit variance too and Softmax will stay diffuse and not saturate too much. Illustration below**
        
        
        """,
        unsafe_allow_html=True,
    )
    # ------------------------------------------------------------------
    st.divider()
    st.markdown("### Study Notes on Self-Attention")

    st.markdown(
        """
    #### 1. Token → Embedding (Static, Pre-Context)
    ```
    tokens (ints) ──► lookup table 𝔼[vocab_size × d_model] ──► x_embed ∈ ℝ^{B×T×d_model}
    ```
    - Each token id (e.g. "Tower" → 50227) is mapped to the same row vector every time
    - Before the model "sees" context
    - Think of the embedding matrix 𝔼 as a learned dictionary

    #### 2. Positional Encoding – Adding Order to a Set
    Self-attention treats the input as an unordered bag of vectors. Inject order by addition:
    ```
    x₀ = Embed(TokenIDs) + PosEnc(0…T−1)
    ```

    ##### Popular Positional Encoding Methods:
    | Method | Pros | Cons |
    |--------|------|------|
    | Sinusoidal (sin/cos) | Extrapolates to longer seqs | Fixed, no tuning |
    | Learned absolute | Simple, best with trunc. len | Cannot extrapolate |
    | Rotary (RoPE) | Encodes relative distance, good for long seqs | Implementation trickier |

    #### 3. Single-Head Self-Attention – "One Table, One Vote"

    ##### 3.1 Parameter Shapes
    | Symbol | Shape | Meaning |
    |--------|-------|---------|
    | `W_Q`, `W_K`, `W_V` | (`d_model`, `d_k`) | Separate linear layers per head |
    | `Q`, `K`,` V` | (B, T, d_k) | Projected token vectors |
    | `d_k` | = head_size | Sub-space dimension (e.g. 64) |

    Rule of thumb: d_model = n_heads × d_k

    ##### 3.2 Forward Pass
    ```python
    Q = X @ W_Q          # (B, T, d_k)
    K = X @ W_K
    V = X @ W_V
    scores = (Q @ K.transpose(-2, -1)) / sqrt(d_k)    # (B, T, T)
    scores = mask(scores)         # causal or full
    attn_w = softmax(scores, dim=-1)
    Z = attn_w @ V           # weighted sum (B, T, d_k)
    ```
    
    ##### 3.3 Attention Score Functions
    Below are four widely-used "compatibility" (similarity) functions that can replace the plain dot-product q @ kᵀ in a self-attention block.

    ###### Common Score Functions
    | Name | Score Formula s(q,k) | Notes |
    |------|---------------------|-------|
    | Scaled dot-product (baseline) | $q·k / √dₖ$ | Fast, kernel-friendly; what Transformer papers mean by "self-attention" |
    | Cosine similarity | $(q·k) / (‖q‖‖k‖)$ | Keeps scores in [-1, 1]; often multiplied by a "temperature" hyper-parameter |
    | Additive / Bahdanau | $vᵀ tanh(W₁ q + W₂ k)$ | Extra learnable matrices; popular in seq-to-seq RNNs |
    | Bilinear / General | $qᵀ W k$ | Lets the model learn how dimensions interact; covers Bilinear Attention Networks, Luong's "multiplicative" attention |

    > Note: "Attention" is the whole process: scores → mask → softmax → weighted sum.
    > The part you're swapping out here is the score (or "compatibility / alignment") function.

    ###### A. Cosine-similarity Attention
    ```python
    import torch, torch.nn.functional as F

    def cosine_attention(q, k, v, mask=None, eps=1e-8, temperature=1.0):
        # q,k,v: (B, T, D)
        q_norm = F.normalize(q, p=2, dim=-1, eps=eps)
        k_norm = F.normalize(k, p=2, dim=-1, eps=eps)
        scores = (q_norm @ k_norm.transpose(-2, -1)) * temperature  # (B,T,T)
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        return attn @ v
    ```
    - Because the range is bounded, you usually multiply by a learnable or fixed temperature (e.g. 10) instead of dividing by √dₖ

    ###### B. Additive (Bahdanau) Attention
    ```python
    import torch.nn as nn, torch.nn.functional as F

    class AdditiveAttention(nn.Module):
        def __init__(self, embed_dim, hidden_dim):
            super().__init__()
            self.w_q = nn.Linear(embed_dim, hidden_dim, bias=False)
            self.w_k = nn.Linear(embed_dim, hidden_dim, bias=False)
            self.v   = nn.Linear(hidden_dim, 1, bias=False)

        def forward(self, q, k, v, mask=None):          # all (B,T,E)
            q_proj = self.w_q(q).unsqueeze(2)           # (B,T,1,H)
            k_proj = self.w_k(k).unsqueeze(1)           # (B,1,T,H)
            scores = self.v(torch.tanh(q_proj + k_proj)).squeeze(-1)  # (B,T,T)
            if mask is not None:
                scores = scores.masked_fill(~mask, float('-inf'))
            attn = F.softmax(scores, dim=-1)
            return attn @ v
    ```
    - Introduced for RNN enc-dec models
    - Still useful when you want extra capacity without increasing head size

    ###### C. Bilinear (General / Luong-style) Attention
    ```python
    class BilinearAttention(nn.Module):
        def __init__(self, embed_dim):
            super().__init__()
            self.W = nn.Parameter(torch.empty(embed_dim, embed_dim))
            nn.init.xavier_uniform_(self.W)

        def forward(self, q, k, v, mask=None):
            scores = (q @ self.W) @ k.transpose(-2, -1)   # (B,T,T)
            if mask is not None:
                scores = scores.masked_fill(~mask, float('-inf'))
            attn = F.softmax(scores, dim=-1)
            return attn @ v
    ```
    - Reduces to the plain dot-product when W = I
    - Can learn cross-dimensional correlations

    ###### D. Negative-squared-distance (RBF / Euclidean) Attention
    ```python
    def rbf_attention(q, k, v, mask=None, sigma=1.0):
        # ‖q - k‖² is broadcast automatically
        scores = -((q.unsqueeze(2) - k.unsqueeze(1))**2).sum(-1) / (2*sigma**2)  # (B,T,T)
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        return attn @ v
    ```
    - Interpretable as putting a Gaussian kernel over the embedding space
    - Rarely used in large Transformers because the triple broadcast is memory-heavy

    ###### When to Use Each Function
    | Scenario | Good Choice | Why |
    |----------|-------------|-----|
    | Huge models / GPU kernels | Scaled dot-product | Fused, numerically stable, minimal parameters |
    | Need scale-invariant similarity | Cosine | Normalisation removes magnitude |
    | Small-/mid-size RNNs, cross-modal problems | Additive | Extra MLP captures non-linear relations, works well at low dims |
    | Want learnable cross-dim interactions | Bilinear | Matrix W learns which dimensions should match |
    | Kernel-methods intuition / proximity search | RBF | Attention falls off smoothly with Euclidean distance |

    > All of them fit into the same "mask → softmax → weight values" pipeline, so you can swap the score block without touching the rest of your Transformer logic.

    #### 4. Scaling by √dₖ – Variance Perspective
    - If Q and K have i.i.d. entries ∼𝒩(0, 1), their dot-product variance ≈ d_k
    - Dividing by √d_k normalises logits to 𝒪(1) ⇒ softmax stays diffuse
    - Avoids near-one-hot rows, keeps gradients alive
    - Modern kernels (torchSDPA, FlashAttention) often pre-scale Q once to save FLOPs

    ##### Why the dot product needs that exact scale
    - Dot product ∼ Σ qᵢ kᵢ
    - If each component has unit variance, the variance of the sum grows linearly with the dimension dₖ
    - Softmax is sensitive to large spreads: logits ≫ 0 drive probabilities toward one-hot and kill gradients
    - Dividing by √dₖ normalises the variance back to ≈ 1, so softmax outputs remain diffuse

    ##### What happens with other similarity functions?
    | Compatibility | Raw range | Common "temperature" fix | Notes |
    |--------------|-----------|--------------------------|-------|
    | Cosine (q·k)/(‖q‖‖k‖) | [-1, 1] | Multiply by a learnable or fixed τ (e.g. τ = 10) | Norms already cancel the √dₖ effect |
    | Additive / Bahdanau vᵀ tanh(W₁q + W₂k) | Depends on ‖v‖ | No mandatory scale; can learn v's magnitude | Hidden layer keeps values bounded by tanh |
    | Bilinear qᵀWk | Var ∝ dₖ,‖W‖ | Often divide by √dₖ or add LayerNorm | Same growth problem as plain dot product |
    | Squared-distance -‖q−k‖² / 2σ² | Controlled by σ | Choose σ or learn 1/σ² | Acts like a Gaussian kernel; σ plays role of temperature |

    > Key idea: every scoring formula benefits from a temperature knob that keeps logits in a range where softmax neither saturates nor becomes too flat. Dot-product's temperature just happens to be the simple analytic value √dₖ.

    ##### Practical take-aways
    - If you use PyTorch's `scaled_dot_product_attention`, `nn.MultiheadAttention`, Flash-Attention, etc. the √dₖ divide is applied for you
    - If you switch to cosine or additive attention drop the divide-by-√dₖ, but consider adding a learnable scalar multiplier τ
    - If you invent a new score watch the logit statistics during training—exploding or vanishing gradients usually mean you need a temperature or normalisation step

    > So, the √dₖ rule is a special case of a broader principle: keep your attention logits in a safe numeric band; the correct scale depends on the similarity you choose.

    #### 5. Mask Types
    | Use-case | Mask | Meaning |
    |----------|------|---------|
    | Encoder (BERT, ViT) | none | All tokens attend all |
    | Decoder (causal) (GPT) | lower-triangular | Token t sees ≤ t |
    | Padding mask | 0 for PAD | Prevents junk attention to padding |
    | Arbitrary (e.g., band, local) | custom | Limits window to save memory |

    #### 6. Residual & LayerNorm
    ```
    X          (input)
         │  Self-Attention
         
    Z = Attn(X)
    
         │  + Dropout
         
    X1 = X + Z          # residual
    X2 = LayerNorm(X1)
    
         │  MLP (≈ 4×d_model hidden)
         
    X3 = MLP(X2)
    Out = LayerNorm(X2 + X3)
    ```
    - Residual: pass original signal forward—stabilises very deep stacks
    - LayerNorm: normalise per token ensuring scale/shift invariance

    #### 7. Position-wise Feed-Forward Network (MLP)
    ```
    MLP(x) = W₂ GELU(W₁x + b₁) + b₂
    ```
    - Applies identical tiny MLP to every sequence position independently
    - Role: inject non-linearity & mix dimensions
    - Does not store distances

    #### 8. Multi-Head Attention – "Ensembled Views"

    ##### 8.1 Why Multiple Heads?
    Different heads can specialise in:
    - Syntax vs. semantics
    - Long-range vs. local
    - Programming structure, etc.

    ##### 8.2 Tensor Implementation (PyTorch)
    ```python
    # X: (B,T,d_model)   d_model = nH * d_k
    QKV = X @ W_qkv      # (B,T,3*d_model)
    QKV = QKV.view(B, T, nH, 3*d_k).transpose(1,2)  # (B, nH, T, 3*d_k)
    Q, K, V = QKV.split(d_k, dim=-1)                # each (B,nH,T,d_k)
    ...
    context = (attn @ V).transpose(1,2).reshape(B,T,d_model)
    ```

    #### 9. Cross-Attention – Talking to Another Sequence
    Used in encoder–decoder models, diffusion, multimodal (text ↔ image):
    ```
    Queries  ← decoder hidden (masked self-attn output)
    Keys/Val ← encoder output  (full context)

    scores = Q · Kᵀ / √d_k   # (B, T_dec, T_enc)
    ```
    Gives each decoder step a dynamic, content-based window over the entire source sequence.

    #### 10. Training Flow & Gradients
    - Loss (e.g. next-token cross-entropy) computed on topmost logits
    - Back-prop drills through:
        - LM head → last block output
        - Each block (LayerNorm, MLP, Attention, projections)
        - Embeddings & positional encodings
    - Network learns Q/K/V sub-spaces from scratch—no manual design

    #### 11. Common Hyper-Parameters Cheat-Sheet
    | Model | d_model | n_heads | d_k | Blocks | MLP hidden | Params |
    |-------|---------|---------|-----|--------|------------|--------|
    | GPT-2 small | 768 | 12 | 64 | 12 | 3072 | 117 M |
    | BERT-Base | 768 | 12 | 64 | 12 | 3072 | 110 M |
    | ViT-B/16 | 768 | 12 | 64 | 12 | 3072 | 86 M |
    | Llama-3-8B | 4096 | 32 | 128 | 32 | 11008 | 8 B |

    MLP hidden ≈ 4× d_model (MLP-ratio)

    #### 12. FAQ & Pitfalls
    | Question | Answer |
    |----------|--------|
    | Do embeddings carry context? | Not before the first attention layer |
    | Is Q · K the only similarity? | No – cosine, additive (Bahdanau), bilinear… but scaled dot-product is fastest |
    | Do W_Q/W_K share weights with embedding? | No, they're independent nn.Linears |
    | Does the MLP memorise distances? | No. It's a per-token non-linear mixer |
    | Why does training diverge without √dₖ? | Softmax saturates ⇒ vanishing gradients |
    | How big should d_k be? | Usually 64 ≤ d_k ≤ 128. Too small → capacity bottleneck; too large → memory |
    | What happens if I remove the residual? | Deep (>3) layers become hard to train; performance drops sharply |
    | Can attention learn 'Eiffel Tower' as a span? | Yes, but only if doing so lowers loss—no direct supervision |

    #### 13. Conceptual Recap Diagram
    ```
    token ids
      │  embed + pos
      ▼
    ┌──────────────────── Transformer Block ──────────────────────┐
    │  ┌── Self-Attention (n_heads) ──┐                           │
    │  │  Q = XW_Q                    │                           │
    │  │  K = XW_K                    │  scores= QKᵀ/√d_k → mask  │
    │  │  V = XW_V                    │           → softmax       │
    │  │  Z = softmax ⊗ V             │                           │
    │  └───────────────┬──────────────┘                           │
    │                  ▼   (residual)                             │
    │             X1 = X + Z                                      │
    │             X2 = LayerNorm(X1)                              │
    │                  │                                          │
    │        ┌─────────┴── Feed-Forward (MLP) ─────────┐          │
    │        │ h = GELU(X2W1 + b1)                     │          │
    │        │ y = hW2 + b2                            │          │
    │        └─────────┬───────────────────────────────┘          │
    │                  ▼   (residual)                             │
    │             Out = LayerNorm(X2 + y)                         │
    └─────────────────────────────────────────────────────────────┘
    ```
    Stack N such blocks → project to logits → softmax → loss.
    """
    )

    st.divider()
    st.markdown("### Let's train the model: Bigram with Self-Attention")

    # Hyperparameters
    batch_size = 32  # How many independent sequences will we process in parallel?
    block_size = 8  # What is the maximum context length for predictions?
    max_iters = 5000  # How many iterations to train for?
    eval_interval = 500  # How often to evaluate the loss?
    learning_rate = 1e-4
    device = "cuda" if torch.cuda.is_available() else "cpu"
    eval_iters = 200
    n_embd = 32

    # import all the functions from gpt.py
    from gpt import (
        load_data,
        get_batch,
        estimate_loss,
        create_Bigram_with_SelfAttention_model,
        train_model,
    )

    # load the data
    train_data, val_data, vocab_size, encode, decode = load_data(
        "data/tiny_shakespeare.txt"
    )

    # get the batch
    xb, yb = get_batch(block_size, batch_size, "train", train_data, val_data)

    # create the model
    bigram_with_self_attention_model, params = create_Bigram_with_SelfAttention_model(
        vocab_size, n_embd, block_size
    )

    st.markdown("#### Model Configuration")
    col1, col2 = st.columns(2)
    with col1:
        st.write("##### Hyperparameters")
        st.write(f"Batch size: `{batch_size}`")
        st.write(f"Block size: `{block_size}`")
        st.write(f"Max iters: `{max_iters}`")
        st.write(f"Eval interval: `{eval_interval}`")
        st.write(f"Learning rate: `{learning_rate}`")
        st.write(f"Device: `{device}`")
        st.write(f"Eval iters: `{eval_iters}`")
        st.write(f"Vocab size: `{vocab_size}`")
        st.write(f"n_embd: `{n_embd}`")

    with col2:
        # Print the model structure
        st.write("##### Model Structure")
        st.write(bigram_with_self_attention_model.get_structure())
        st.write(f"Number of parameters: `{params}`")

    st.divider()

    st.markdown("#### Re-prepare the Data")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"Train data size: `{len(train_data)}`")
        st.write(f"Val data size: `{len(val_data)}`")
        st.write(f"Vocab size: `{vocab_size}`")
        st.write(f"Encode: `{encode}`")
        st.write(f"Decode: `{decode}`")
    with col2:
        st.write(f"xb: `{xb[:4]}`")
        st.write(f"yb: `{yb[:4]}`")
        st.write(f"xb shape: `{xb.shape}`")
        st.write(f"yb shape: `{yb.shape}`")
    # Move input tensors to the same device as the model
    xb = xb.to(device)
    yb = yb.to(device)

    st.divider()

    st.markdown("#### Initial Model Evaluation")
    out, loss = bigram_with_self_attention_model.forward(xb, yb)

    st.write(f"Output shape: `{out.shape}`")
    st.write(f"Loss: `{loss.item() if loss is not None else None}`")

    # Move idx to the same device for generation
    idx = torch.zeros((1, 1), dtype=torch.long, device=device)
    max_new_tokens = 100

    st.write(f"Initial idx: **{idx}** with max_new_tokens: **{max_new_tokens}**")

    st.write(f"Generated text: ")
    # generated = model.generate(idx, max_new_tokens=max_new_tokens)
    # st.write(f"Generated shape: **{generated.shape}**")
    # st.write(f"Generated: **{generated}**")
    # st.write(
    #     f"{decode(model.generate(idx, max_new_tokens=max_new_tokens)[0].cpu().tolist())}"
    # )

    # # ------------------------------------------------------------------

    # st.markdown("### Final Model Output")
    # model, training_records = train_model(
    #     model,
    #     train_data,
    #     val_data,
    #     max_iters,
    #     eval_interval,
    #     eval_iters,
    #     block_size,
    #     batch_size,
    #     learning_rate,
    # )

    # # st.markdown("#### Training Results")
    # # Convert training records to DataFrame
    # training_records_df = pd.DataFrame(training_records)

    # # Create Altair chart with axis limits
    # chart_for_training_records = (
    #     alt.Chart(training_records_df)
    #     .mark_line()
    #     .encode(
    #         x=alt.X("step", title="Step", scale=alt.Scale(domain=[0, max_iters])),
    #         y=alt.Y(
    #             "train_loss", title="Training Loss", scale=alt.Scale(domain=[0, 8])
    #         ),
    #     )
    #     .properties(width="container", height=400)
    # )

    # # Add validation loss line
    # val_line = (
    #     alt.Chart(training_records_df)
    #     .mark_line(color="red")
    #     .encode(
    #         x="step",
    #         y="val_loss",
    #     )
    # )

    # # Combine the charts
    # final_chart = chart_for_training_records + val_line

    # st.altair_chart(final_chart, use_container_width=True)

    # # Add a description
    # st.caption(
    #     "Training and validation loss over time - lower values indicate better model performance"
    # )

    # col1, col2 = st.columns(2)
    # with col1:
    #     st.write("Steps number:")
    #     st.write(f"**{max_iters}**")
    #     st.write("Loss:")
    #     st.write(loss.item())
    # with col2:
    #     st.write("Generate the text")
    #     # st.write(f"{decode(model.generate(idx, max_new_tokens=500)[0].cpu().tolist())}")

    # # ------------------------------------------------------------------
    # st.markdown("### Text Generation")
    # # Prompt the user to enter a prompt
    # st.divider()

    # st.write("Enter a prompt to generate text:")
    # prompt = st.text_area("Prompt")

    # if st.button("Generate"):
    #     if prompt:
    #         # Encode the prompt
    #         context = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
    #         # Generate new tokens
    #         generated = model.generate(context, max_new_tokens=500)
    #         # Decode and display
    #         st.write(f"Generated text: {decode(generated[0].tolist())}")
    #     else:
    #         st.write("Please enter a prompt first.")
