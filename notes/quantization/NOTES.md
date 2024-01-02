# quantization

## [AutoGPTQ](https://arxiv.org/pdf/2210.17323.pdf)

### Paper Notes

-  **Layer-Wise Quantization**: 
	- GPTQ folows this approach, atleast from a high level.
  - Going layer by layer, find the weights which produces the most similar outputs to the original weights, want to find:

$$\small \underset{\hat{W}}{\arg\min}} ||WX - \hat{W}X||^2_2 \tag{1}$$

- $X$ is the expectation over the layer inputs, can be approximated by taking the mean over a small set of $N$ input examples.

- **Optimal Brain Quantization (OBQ)**:
  - Method for solving $(1)$, GPTQ paper modifies the original method for performance.
    - Inspired by [Optimal Brain Surgeon](https://www.babak.caltech.edu/pubs/conferences/00298572.pdf). 
  - OBQ:
    - Select the layer to update, $w_q$:
      $$\underset{w_q}{\arg\min}\frac{(\text{quant}(w_q)-w_q)^2}{[H^{-1}_F]_{qq}} \tag{2.1}$$
    - Compute corresponding update to remaining full precision weights, $F$:
    $$\delta_F=-\frac{w_q-\text{quant}(w_q)}{[H_F^{-1}]_{qq}}\cdot (H_F^{-1})_{:,q} \tag{2.2}$$
    - Update is to mitigate errors from quantizing a single weight (layer?).
    - Layer to update chosen greedily, picks the weights to quantize based off quantization error.
  - Cubic runtime wrt to input dimensions, $O(d_{row}\cdot d_{col}^3)$
    - > Tries to avoid a full recomputation of $H^{1}$ by removing the *qth* row and column after quantizing $w_q$ directly in the inverse via one step of Gaussian elimination

- **GPTQ**
  - **Arbitrary order insight**:
    - Choosing the layer to quantize based off quantization error only slightly improves performance vs. arbitrary order.
      - Especially so for large models with heavily parameterized layers.
      - Due to layers with large quantization weights being updated last in OBQ, when there are less unquantized weights available to compensate for the large error.
    - Rather than independently updating each row of $W$ in a specific order defined by the error, GPTQ updates all rows in the same order. 
      - Since the order is fixed, unquantized $H^{-1}$ and $W$ is constant over all rows. Reduces runtime to $O(max(d_{row}\cdot d_{col}^2,d_{col}^3))$
  - **Lazy batch updates**:
    - Direct implementation of the algorithm has a low compute to memory access ratio as it needs to update a large matrix where each element requires few FLOPs.
    - Rounding decisions for column $i$ depends only on the updates performed on that column, later updates are irrelevant. So given a chosen block size $B$, only update columns within that block and only perform a global update of $H^{-1}$ and $W$ once the block is fully processed.
  - **Cholesky Reformulation**:
    - So $H_F^{-1}$ has a tendency to become "indefinite" (0, since we divide by values in $\mathbf{H}$) due to accumulation of numerical errors during the update, this is obviously not good and causes the algorithm to update the remaining weights in the wrong direction.
      - More likely to occur in larger models and almost certainly occurs in layers with billions of parameters.
    - Can mitigate this for smaller models by adding a small constant to the diagonal elements of $H$.
    - For larger models, precompute the needed elements of $H_{F_{q}}^{-1}$ via a more numerically stable method (we only need elemenst after the diagonal in row $q$ of $H_{F_{q}}^{-1}$ when quantizing weight $q$).
      - *Cholesky kernels*?
    

Algorithm 1 Quantize $\small W$ given inverse Hessian $\small H^{-1} = (2XX^T + \lambda I)^{-1}$ and blocksize B.
```python
Q ← 0_{d_row x d_col}                     // quantized output
E ← 0_{d_row x B}                        // block quantization errors
H^{-1} ← Cholesky(H^{-1})^T           // Hessian inverse information

for i = 0, B, 2B, ... do
    for j = i, ..., i + B - 1 do
        Q_j,; ← quant(W_j,;)
        E_j,; ← (W_j,; - Q_j,;) / H^{-1}_jj
        W_j,;(i+j+B) ← W_j,;(i+B) - E_j,; * H^{-1}_j,;(i+B)
    end for
    W_;(i+B) ← W_;(i+B) - E * H^{-1}_;(i+B);(i+B)
end for
```

- *Iterate*: $\large\mathscr{i}$ from 0 to $d_{col}$ in steps of size $B$
    - *Iterate*: $\large\mathscr{j}$ from $\mathscr{i}$ to $\mathscr{i} + B -1$
        - Quantize column $\large \mathscr{j}$ of $W$
        - Compute the quantization error w.r.t the full precision weights, normalize with the $\mathscr{j}^{th}$ diagonal element of $H$
        -  Scale the error we just computed with the corresponding row of $H$, taken starting from the $\mathscr{j}^{th}$ diagonal element to the end of the current block
        - Update all rows of $W$ in the current block by subtracting the scaled error from the weights.
    - Update the remaining unquantized blocks in $W$

In a nutshell, we:

1) Compute the Hessian matrix.
    - The 2nd derivative of $(1)$, $\mathbf{H}=\mathbf{X}\mathbf{X^T}$
2) Compute the inverse of that Hessian matrix.
    - In OBQ, we iterate over each row and select the weight that introduces the least error when quantized to quantize. We then update the remaining unquantized weight with our error to offset the error we just introduced by quantizing. While this worked, authors of GPTQ found that this essentially didn't matter and if we iterated in a fixed order, we can avoid recomputing the inverse Hessian matrix and reduce the runtime from $O(d_{row}\cdot d_{col}^3)$ to $O(\max(d_{row}\cdot d_{col}^2, \, d_{col}^3))$
    - One of the key contributions from GPTQ is precomputing this via Cholensky factorization.
3) Iterate over the columns of $W$ in chunks of $B$, `blocksize`
    - This is another contribution from GPTQ, OBQ authors observed one only had to update the weights in the current block rather than the entire set of weights. 
4) Quantize the weights
5) Compute the error by comparing it to the full precision weights, scaled by the value of our inverse Hessian matrix on the 
6) Update all remaining unquantized weights based on our error, essentially we're adjusting the unquantized weights to make up for the error we introduced by quantizing the weights

## [QUIK](https://arxiv.org/abs/2310.09259)




