# README

Simple Image Classifier Model.

- Trained on 1 CPU, 1 GPU (Cuda and MacOS) and multiple GPUs (Cuda).
- Using Pytorch and Pytorch Lightning


# Learnings

- GPUs are slower than CPUs unless there's enough data for parallel threads to make a difference.


# Experiments
Training time with a batch size of 4 on single CPU: ~55s
Training time with a batch size of 4 on single GPU: ~1m54s
Accuracy ~ 54%

Training time with a batch size of 1000 on single CPU: ~49s
Training time with a batch size of 1000 on single GPU: ~28s
Accuracy < 15%
*A couple category had high accuracy (>75%) while all the others had 0%.

Training time with a batch size of 4 and wider hidden layers (20 nodes between conv layers) on single CPU: ~1m3s
Training time with a batch size of 4 and wider hidden layers (20 nodes between conv layers) on single GPU: ~1m53s
Accuracy ~ 58%

# Questions
- Can pin_memory improve speed of the GPU? hypothesis: is the memory transfer causing a slowdown?

# References
[Training a Classifier — PyTorch Tutorials 2.0.1+cu117 documentation](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
[Optional: Data Parallelism — PyTorch Tutorials 2.0.1+cu117 documentation](https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html)