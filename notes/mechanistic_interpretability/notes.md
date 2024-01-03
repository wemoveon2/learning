# [Concrete Steps to Get Started in Transformer Mechanistic Interpretability](https://www.neelnanda.io/mechanistic-interpretability/getting-started)

## [Mechanistic Interpretability](https://dynalist.io/d/n2ZWtnoYHrU1s4vnFSAQ519J#z=eL6tFQqNwd4LbYlO1DVIen8K)

### Definitions

- **Mechanistic Interpretability** - The study of reverse engineering NN from the learned weights to the human interpretable algorithms.
  - Analogous to reverse engineering compiled program binary back to source code.
- **Feature** - Some property of the input to a model. 
  - Such as a specific token in a prompt for a LLM or a patch of an image.
    Fuzzy definition, usually used to describe property of the example which has an internal representation in the NN. The property is usually something generalized across inputs rather than the property of a specific input.
- **Decomposability** - The degree to which a network representation can be described in terms of independently understandable features.
- **Circuit** - The subset of the model's weights and non-linearities used to map a set of earlier features to a set of later features.
  - Also a fuzzy term, intuitively it means the components of a model which performs the computations that produces some interpretable features from prior interpretable features.
- **Intervening on** or **editing** - Refers to the process of editing or replacing activations once they've been generated but not used in the next layer. 
- **Neuron Splitting** - When a feature is decomposed into several features.
- **Universality** - Hypothesis that the same circuits will show up in different models. 
- **Microscope AI** - Idea that when we achieve ASI, rather than needing to use it, we can reverse engineer it to learn what it has learned about the world, and use this knowledge instead.

### Representation of features

- The activations and weights of a NN live in high dimensional space, which is hard to reason about. 
- To understand what the model is doing, we decompose the model's internal activations into *features*, and the features into *circuits*.
  - Decomposability is crucial to resolve the curse of dimensionality, features might not be decomposable if it is highly correlated with another feature.
  - The features should be useful for computing the outputs (mutual information between the feature and correct output) and can be recovered from the activations.

- The **linear representation hypthoesis** is a hypothesis that the features are represented in the model as **directions** in activation space. This would mean the features can be **recovered** by projecting onto the relevant directions.
  - Main thing models do is linear algebra, including projecting onto certain directions. As such, it's a natural way for a NN to represent features. 
    - If a later layer wanted a feature, it can project onto that feature's direction.
    - Multiple features can be represented and combined, they can vary independently, and strength of feature can be represented as the magnitude in a particular direction.

- **Interpretable Basis** - Set of directions in activation space where each direction corresponds to an interpretable feature.
  - We don't necessarily know what the directions are, we just know what the features are.
- **Priviledged basis** - Set of vectors which have some meaning, but may not be interpretable.
  - A space can have an interpretable basis without having a priviledged basis.
- **Bottleneck Activation** - These are lower dimensional intermediate activations located between input and output activations of higher dimensionality.
- **Features as Neurons** - Hypothesis that not only do features correspond to directions, but that each feature corresponds to a specific neuron. The activation of the neuron is the strength of the feature in the input.

#### Superposition

- **Superposition** is when a model represents more than **n** features in an **n** dimensional activation space.
  - Features still correspond to directions, but the set of interpretable directions is larger than the number of dimensions.
  - Model is simulating a larger model.
  - If superposition is being used, there *cannot* be an interpretable basis, features as neurons cannot perfectly hold.
  - We cannot perfectly recover features as they cannot all be orthogonal. 
- In a transformer, there are two kinds of superposition:
  - **Bottleneck Superposition** - If there are 50k tokens in the vocab and 768 dimensions in the *residual stream*, there almost has to be more features than dimensions. This type of superposition is used for storage.
  - **Neuron Superposition** - When there are more features represented than there are neurons. Using n non-linearities to do some processing that outputs more than n features.
- **Neuron Polysemanticity** - Idea that a single neuron activation corresponds to multiple features, such as when the neuron activates on multiple clusters of unrelated things.
  - Neurons are **monosemantic** if it corresponds to a single feature.
  - Saying a neuron is polysemantic is equivalent to saying the standard basis isn't interpretable.

- Superposition is essentially a form of lossy compression, the model tries to represent more features at the cost of adding noise and interference between features.
  - There has to be some optimal point balancing the two, plausible that the optima isn't zero superposition.
- Two key aspects of a feature: its importance (usefulness in getting lower loss, more expensive to interfere with important features) and its sparsity (how frequently it appears in the input, the sparser it is the less it interferes with other features). 
  - Problems with sparse, unimportant features will show significant superposition.

# [Transformers (From a MI Perspective)](https://dynalist.io/d/n2ZWtnoYHrU1s4vnFSAQ519J#z=pndoEIqJ6GPvC1yENQkEfZYR)

# [Fact Finding: Attempting to Reverse-Engineer Factual Recall on the Neuron Level](https://www.alignmentforum.org/posts/iGuwZTHWb6DFY3sKB/fact-finding-attempting-to-reverse-engineer-factual-recall)






