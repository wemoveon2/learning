# Tutorial 6: Transformers and Multi Head Attention

- Attention is a mechanism which takes a weighted average of a sequence where the weights are determined dynamically based on an input query and the elements' keys.
  - Goal is to average over features of multiple elements, but with the weight dependent on the values of the elements.
- Four components:
  1. Query: The query is a feature vector that describes what we are looking for in the sequence.
  2. Keys: Each input element has a key, which is another feature vector. This feature vector describes what the element offers or what might be important.
  3. Values: Each input element also has a value vector, this feature vector is the one we want to average over.
  4. Score function: Used to rate which element to pay attention to. Takes in a query and key and returns the score of the pair.
  - The weights for the average are calculated by a softmax over all score functions.
- Transformers use self attention. Each element has a query, key, and value. For each element, we use its query and compare it to every other key to get a averaged value vector for each element.
  - We assign a higher weight to value vectors with keys similar to the query. 
  - In **self attention**, we obtain the key, query, and value vectors from each element in the input sequence.

## Scaled Dot Product Attention

- We want an attention mechanism where any element can be compared to any other element while still being efficient.
- The dot product attention takes a set of queries $Q \in \mathbb{R}^{T \times d_k}$, keys $K \in \mathbb{R}^{T \times d_k}, and values $V \in \mathbb{R}^{T \times d_v}. 
  - $T$ is the sequence length, $d_k$ and $d_v$ are the hidden dimensions for keys/queries and values.
  - The attention value from element $i$ to $j$ is the similarity of the query $Q_i$ and key $K_j$, essentially the proportion of $Q$ that lies in the direction of $K$. 
  - $QK^T$ performs the dot product for every pair of queries and keys, resulting in a matrix with shape $T\times T$.
    - Each row of this matrix represents the attention logits for a specific element $i$ to all other elements.
  - Softmax is applied to the resulting matrix and the result is multiplied with the value matrix to obtain the weighted mean. 
- The *scaling factor* of $\frac{1}{\sqrt{d_k}}$ is to maintain appropriate variance of attention values. Since performing dot product over two vectors with variance $\sigma^2$ results in a scalar with $d_k$ times higher variance,we must scale the variance back.
- The optional *mask* is for when we batch multiple sequences of unequal length. We must pad the shorter sequences and mask out the padding tokens during calculation of the attention values (usually by setting their attention logits to a very low value).

```python
def scaled_dot_product(q, k, v, mask=None):
  d_k = q.shape[-1]
  attn_logits = torch.matmul(q, k.transpose(-2, -1))
  attn_logits = atten_logits / math.sqrt(d_k)
  if mask is not None:
    attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
  attention = F.softmax(attn_logits, dim=-1)
  values = torch.matmul(attention, v)
  return values, attention
```

## Multi-Head Attention

- There might be different aspects of a sequence to attend to, as such, taking a single weighted average won't suffice. MHA splits the query, key, value triplets into subsets which undergo scaled dot product attention independently before their results are concatenated into a final matrix.




