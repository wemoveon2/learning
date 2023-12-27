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
