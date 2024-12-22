# Simulation Stack for Cross-Architecture and Cross-Accelerator Evaluation
This subdirectory contains the stack we created to easily compare accelerator performance 
across different model architectures. This stack surveys the performance of different accelerators against baselines EyeRiss, generic (naive) weight stationary, and
generic output stationary, on the DeepLabv3 DNN architecture. In particular, we were interested in how these accelerators perform
on Atrous convolutions, which introduce structured kernel sparsity.

### 0. Reference directory
Directory and file paths referenced below are relative to `workspace/final-project` unless otherwise specified.

### 1. Implement desired models using PyTorch
We compared the DeepLabv3 architecture and an undilated equivalent architecture, which we creatively called Undilated DeepLabv3. This example is included in `gen_yamls.ipynb`. You can either code your own implementation or import them from `torchvision.models`.

### 2. Generate `.yaml` files for your models
Specify desired directories containing the model architecture in the desired format. Then, run the appropriate cell in `gen_yamls.ipynb` for all the models you want to compare, ensuring the fields below are specified.
```
    top_dir='example_designs/layer_shapes/CONV',
    sub_dir='<desired model name>',
    timeloop_dir='example_designs/example_designs/<desired architecture>'
```

### 3. Model Architecture Parsing
We need to correct some inconsistent naming to feed the model layers into Timeloop. Use 
`example_designs/rename.py` to specify the folder path to be the subdirectory `subdir` in step (2). 
From `example_designs/main.ipynb`, run this for all models you generated above.

### 4. Summary Statistic Generaton
From `example_designs/main.ipynb`, run the following for the models you generated as well as the architectures whose performances you wish to compare.
```
!python3 run_example_designs.py --architecture <desired architecture as in (2)> --problem CONV/<desired model name as in (2)>
```
This will take a while.
At the end, you should have the desired summary statistics `{...}/timeloop-mapper.stats.txt` for all layers.

### 5. Visualization
The simulation stack supports performance visualization. Navigate to the next cell and specify `file_dir1` and `file_dir2` for the two architectures you wish to compare.
```
file_dir1 = 'example_designs/<desired architecture>/outputs_<first model name>/'
file_dir2 = 'example_designs/<desired architecture>/outputs_<second model name>/'
```
For instance,
```
file_dir1 = 'example_designs/simple_output_stationary/outputs_DeepLabv3/'
file_dir2 = 'example_designs/simple_output_stationary/outputs_UndilatedDeepLabv3/'
```
Some recommended and common metrics (see `archs`) include the following:
```
# fJ/compute metrics
strings = ["mac                                           ", "psum_spad                                     ",  "weights_spad                                  ", \
           "ifmap_spad                                    ", "shared_glb                                    ", "DRAM                                          ", \
           "Total                                         "]

strings = ["GFLOPs (@1GHz):", "Utilization:", "Cycles:", "Energy:", "EDP(J*cycle):"]
labels = ["GFLOPs (@1GHz)", "Utilization (%)", "Cycles", "Energy (uJ)", "EDP (J*cycle)"]
archs = ["eyeriss_like", "simple_weight_stationary", "simple_output_stationary"]
```

Simply specify the list of desired metrics, ensuring that all metrics are consistent (e.g. `fJ/compute`). Regular expressions cleanly parse summary statistics and report the desired metrics.
