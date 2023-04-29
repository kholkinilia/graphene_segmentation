## Unet Approach

This is the code of a reproduction of the approach using UNET. The original code and a paper can be found [here](https://github.com/Hui-Ying/Graphene-automatic-detection).

Changes to the original code:
* Changed color model from SVM (which didn't use GPU) to two-layer Fully Connected NN achieving significant performance increase without loss of quality.
* Added logging using [Weights & Biases](https://wandb.ai/site).
* Refined the code: removed copy-pasting, made it simpler overall.
* And, of course, created a custom dataset class.
