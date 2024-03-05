# Goals

- Pre-train a sLLM

  - Small only because I'm poor, $100 budget <1.1b-3b params?
  - Can reuse code for modeling and training
  - What's the cheapest $/param achievable?

- Resulting model should perform well

  - Might be interesting to play with some synthetic dataset generation methods to generate data off one of the benchmarks.
    - [Garrulus](https://huggingface.co/udkai/Garrulus) wrote a paper off contaminating [NeuralMarcoro](https://huggingface.co/mlabonne/NeuralMarcoro14-7B) with benchmark data
      - Trained on Winogrande, benchmarked on TruthfulQA, HellaSwag, and ARC.
    - [Turdus](https://huggingface.co/udkai/Turdus) is a "less contaminated ver"
      - 0.2% increase in performance on 5 other benchmarks with only 1 epoch on 1200 examples.
  - There was research showing training on coding improves reasoning benchmarks (find citation, probably some coding model).

- The model should be uncensored
  - Dolphin
    - Indicates this is achievable at the fine-tune stage by filtering refusals.

# TODOs

- [ ] Determine architecture for model
- [ ] Get data for training and eval
- [ ] Figure out compute requirements
- [ ] Determine hardware

# Research

- [ ] Orca
- [ ] Phi
- [ ] tiny llama

- [ ] Creating good synthetic datasets
- [ ] training optimizations
- [ ] architecture
  - efficiency, performance

## Training

- LLM369 models
  - [Cerebras Model Zoo](https://github.com/Cerebras/modelzoo)
    - [CrystalCoder](https://github.com/LLM360/crystalcoder-train)
      - This model just used the gpt3 that came with the model zoo.
  - [Amber](https://github.com/LLM360/amber-train)
    - Uses `flash_attn` lib
    - Ran on 56 nodes each with 4 GPUs and 16 CPUs for 7b model.
      - Uses [Lightning Fabric](https://lightning.ai/docs/fabric/stable/) for parallelization
        - A more flexible `lightning.Trainer`.
    - Uses `fire` to generate a one liner CLI off the parameters of `main`, `fire.Fire(main)`
