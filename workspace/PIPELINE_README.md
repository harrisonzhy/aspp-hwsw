# Pipeline for Cross-Model and Cross-Architecture Evaluation
This repository contains the codebase for the pipeline we created to easily compare deep neural network (DNN) accelerator performance 
across different DNNs. In our project, we compared the performance of different accelerators, Eyeriss, simple weight stationary, and
simple output stationary, on the DeepLabv3 DNN architecture. In particular, we were interested in how these accelerators perform
on Atrous convolutions, which introduce regular sparsity in the kernel (as opposed to, say, the ifmap).

The pipeline is currently set up to generate and visualize data for a DeepLabv3 model and an undilated equivalent. However, the pipeline
is totally generalizable to other models!

### 0. Reference directory
Note that directory and file paths referenced below are relative to `final-project` unless otherwise specified (such as in `.py` files).

### 1. Implement desired models using PyTorch
We compared the DeepLabv3 architecture and an undilated equivalent architecture, which we creatively called Undilated DeepLabv3. This example is included in `gen_yamls.ipynb`. You can either code your own implementation or import them from `torchvision.models`. Both methods work.

### 2. Generate `.yaml` files for your models
For this part, simply specify some directories, run the appropriate cell in `gen_yamls.ipynb`, and the `profiler` will take care of the rest. Run this for all the models you want to compare, making sure to specify the fields below.
```
    top_dir='example_designs/layer_shapes/CONV',
    sub_dir='<desired model name in CONV folder>',
    timeloop_dir='example_designs/example_designs/<desired architecture>'
```

### 3. Format the `.yaml`'s
We need to correct some inconsistent naming in the `.yaml` files to be able to feed them into Timeloop. Navigate to 
`example_designs/rename.py`, specify the folder path to be the subdirectory `subdir` in step (2). From `example_designs/main.ipynb`, run this for all models that you generate above.

### 4. Timeloop outputs
From `example_designs/main.ipynb`, run the following for the models you generated as well as the architectures whose performances you wish to compare.
```
!python3 run_example_designs.py --architecture <desired architecture as in (2)> --problem CONV/<desired model name as in (2)>
```
This will take a while, but at the end, you should have the desired summary statistics `{...}/timeloop-mapper.stats.txt` for all layers. Next, be sure to rename your output folders to a unique name for each architecture, e.g. `example_designs/example_designs/eyeriss_like/outputs_DeepLabv3` and `example_designs/example_designs/eyeriss_like/outputs_UndilatedDeepLabv3`. See (5) for a template.

### 5. Visualization
We can finally generate pretty graphs. Navigate to the next cell and specify `file_dir1` and `file_dir2` for the two architectures you wish to compare. For instance,
```
file_dir1 = 'example_designs/<desired architecture>/outputs_<first model name>/'
file_dir2 = 'example_designs/<desired architecture>/outputs_<second model name>/'

file_dir1 = 'example_designs/simple_output_stationary/outputs_DeepLabv3/'
file_dir2 = 'example_designs/simple_output_stationary/outputs_UndilatedDeepLabv3/'
```
Some metrics we extracted (which are common metrics to extract) across the architectures in `archs` include the following:
```
# fJ/compute metrics
strings = ["mac                                           ", "psum_spad                                     ",  "weights_spad                                  ", \
           "ifmap_spad                                    ", "shared_glb                                    ", "DRAM                                          ", \
           "Total                                         "]

strings = ["GFLOPs (@1GHz):", "Utilization:", "Cycles:", "Energy:", "EDP(J*cycle):"]
labels = ["GFLOPs (@1GHz)", "Utilization (%)", "Cycles", "Energy (uJ)", "EDP (J*cycle)"]
archs = ["eyeriss_like", "simple_weight_stationary", "simple_output_stationary"]
```

Simply specify the list of desired metrics, keeping in mind that the units for all of these metrics should be the same for the same plot, for instance, fJ/Compute. If you choose to customize the lists, they must consist of valid substrings in all instances of `{...}/timeloop-mapper.stats.txt`. Additionally, each specified substring should have a number on the same line. Regular expressions automatically parse the data and search for these indicators within the summary statistics files. The graphs are created using `matplotlib`. The `loop_archs_graph()` function visualizes the desired metrics on the same plot across all architectures specified in `archs`.
