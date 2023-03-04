# Thin Graphene layers detection.

This is the code of a reproduction of the approach using UNET. The orignal code can be found [here](https://github.com/Hui-Ying/Graphene-automatic-detection).

Changes to the original code:
* Changed color model from SVM (which didn't use GPU) to two-layer FCNN achieving significant performance increase without loss of quality.
* Added logging using [Weights & Biases](https://wandb.ai/site).
* Refined the whole thing: removed copypaste, made it simpler overall.
* And, of course, had to write my own class holding dataset.
