# symbolic regression
This subfolder contains an example on how perform symbolic regression in cuML (from C++)
There are two `CMakeLists.txt` in this folder:
1. `CMakeLists.txt` (default) which is included when building cuML
2. `CMakeLists_standalone.txt` as an example for a stand alone project linking to `libcuml.so`

## Build
`symreg_example` is built as a part of cuML. To build it as a standalone executable, do
```bash
$ cmake .. -DCUML_LIBRARY_DIR=/path/to/directory/with/libcuml.so -DCUML_INCLUDE_DIR=/path/to/cuml/headers
```
Then build with `make` or `ninja`
```
$ make
Scanning dependencies of target raft
[ 10%] Creating directories for 'raft'
[ 20%] Performing download step (git clone) for 'raft'
Cloning into 'raft'...
[ 30%] Performing update step for 'raft'
[ 40%] No patch step for 'raft'
[ 50%] No configure step for 'raft'
[ 60%] No build step for 'raft'
[ 70%] No install step for 'raft'
[ 80%] Completed 'raft'
[ 80%] Built target raft
Scanning dependencies of target symreg_example
[ 90%] Building CXX object CMakeFiles/symreg_example.dir/symreg_example.cpp.o
[100%] Linking CUDA executable symreg_example
[100%] Built target symreg_example
```
`CMakeLists_standalone.txt` also loads a minimal set of header dependencies(namely [raft](https://github.com/NVIDIA/raft) and [cub](https://github.com/NVIDIA/cub)) if they are not detected in the system.
## Run

1. Generate a toy training and test dataset
```
$ python prepare_input.py
Training set has n_rows=250 n_cols=2
Test set has n_rows=50 n_cols=2
Wrote 500 values to train_data.txt
Wrote 100 values to test_data.txt
Wrote 250 values to train_labels.txt
Wrote 50 values to test_labels.txt
```

2. Run the symbolic regressor using the 4 files as inputs. An example query is given below
```bash
$ ./symreg_example -n_cols 2                   \
                   -n_train_rows 250           \
                   -n_test_rows 50             \
                   -random_state 21            \
                   -population_size 4000       \
                   -generations 20             \
                   -stopping_criteria 0.01     \
                   -p_crossover 0.7            \
                   -p_subtree 0.1              \
                   -p_hoist 0.05               \
                   -p_point 0.1                \
                   -parsimony_coefficient 0.01
```

3. The corresponding output for the above query is given below :

```
Reading input with 250 rows and 2 columns from train_data.txt.
Reading input with 250 rows from train_labels.txt.
Reading input with 50 rows and 2 columns from test_data.txt.
Reading input with 50 rows from test_labels.txt.
***************************************
Allocating device memory...
Allocation time =   0.259072ms
***************************************
Beginning training on given dataset...
Finished training for 4 generations.
              Best AST index :      1855
              Best AST depth :         3
             Best AST length :        13
           Best AST equation :( add( sub( mult( X0, X0) , div( X1, X1) ) , sub( X1, mult( X1, X1) ) ) )
Training time =    626.658ms
***************************************
Beginning Inference on Test dataset...
Inference score on test set = 5.29271e-08
Inference time =    0.35248ms
Some Predicted test values:
-1.65061;-1.64081;-0.91711;-2.28976;-0.280688;
Corresponding Actual test values:
-1.65061;-1.64081;-0.91711;-2.28976;-0.280688;
```
