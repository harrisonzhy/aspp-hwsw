# aspp-hwsw

## Introduction
This project explores AI accelerators for structured weight sparsity in dilated Atrous convolutions, which used in semantic segmentation to enhance receptive field without reducing the spatial dimension.
This is a capstone project for 6.5930 (Hardware Architecture for Deep Learning) at MIT.
The codebase is based off of the publicly-available
`timeloop-accelergy-exercises` repository, which contains example designs,
workloads, and tutorials for using Timeloop and Accelergy.

## Getting Started
Update the container, and then start with `docker-compose up`. 
```
cd aspp-hwsw
export DOCKER_ARCH=arm64
docker-compose pull
docker-compose up
```

## Performance Evaluation
See `workspace/README.md` for details on how to model performance across different model architectures and different accelerator architectures.

##  Related reading
 - [Timeloop/Accelergy documentation](https://timeloop.csail.mit.edu/v4)
 - [Timeloop/Accelergy tutorial](http://accelergy.mit.edu/tutorial.html)
 - [SparseLoop tutorial](https://accelergy.mit.edu/sparse_tutorial.html)
 - [eyeriss-like design](https://people.csail.mit.edu/emer/papers/2017.01.jssc.eyeriss_design.pdf)
 - [simba-like architecture](https://people.eecs.berkeley.edu/~ysshao/assets/papers/shao2019-micro.pdf)
