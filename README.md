# aspp-hwsw

## Introduction
This project explores AI accelerators for structured weight sparsity in dilated Atrous convolutions, which used in semantic segmentation to enhance receptive field without reducing the spatial dimension.
The codebase is based off of the publicly-available
`timeloop-accelergy-exercises` repository, which contains example designs,
workloads, and tutorials for using Timeloop and Accelergy.

### Environment
Update the container, and then start with `docker-compose up`. 
```
cd aspp-hwsw
export DOCKER_ARCH=arm64
docker-compose pull
docker-compose up
```

###  Related reading

 - [Timeloop/Accelergy documentation](https://timeloop.csail.mit.edu/v4)
 - [Timeloop/Accelergy tutorial](http://accelergy.mit.edu/tutorial.html)
 - [SparseLoop tutorial](https://accelergy.mit.edu/sparse_tutorial.html)
 - [eyeriss-like design](https://people.csail.mit.edu/emer/papers/2017.01.jssc.eyeriss_design.pdf)
 - [simba-like architecture](https://people.eecs.berkeley.edu/~ysshao/assets/papers/shao2019-micro.pdf)
