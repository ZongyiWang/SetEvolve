## Implementation of *SetEvolve*, CIKM 2019.

Please cite the following work if you find the code useful.

```
@inproceedings{yang2019setevolve,
	Author = {Yang, Carl and Gan, Lingrui and Wang, Zongyi and Shen, Jiaming and Xiao, Jinfeng and Han, Jiawei},
	Booktitle = {CIKM},
	Title = {Query-specific knowledge summarization with entity evolutionary networks},
	Year = {2019}
}
```
Contact: Carl Yang (yangji9181@gmail.com)

## Publication
Carl Yang, Lingrui Gan, Zongyi Wang, Jiaming Shen, Jinfeng Xiao, Jiawei Han, "Query-Specific Knowledge Summarization with Entity Evolutionary Networks", CIKM19.


## Content
   Files:
   - BaseGraphicalLasso.py -   Base class for graphical lasso
   - StaticGL.py           -   GL solver solving time independent graphical lasso problems
   - DataHandler.py        -   Results writer
   - penalty_functions.py  -   Penalty functions
   - SetEvolve.py          -   Main algorithm

   \*BaseGraphicalLasso.py, StaticGL.py, DataHandler.py, penalty_functions.py are adopted from:  
    Time-Varying Graphical Lasso, https://github.com/tpetaja1/tvgl

   Folders:
   - case_study            -   input folder, time dependent observations
   - network_results       -   output folder from the algorithm, network inferences
   

## Deployment

Implemented in Python2, with numpy, scipy and multiprocessing


## Demo

  Run the following command

  ```
  python SetEvolve.py case_study/bio_1991_2003.csv 2 4 20 0
  ```

## Parameter

  1. filename
  2. discrete param (discrete=2, otherwise=1)
  3. number of blocks
  4. lambda
  5. output result matrix(optional, 0 as default), x means the xth network inference will be output as adjacency list


## Input

  See folder case_study, an input file contains:
  1. one line of entities
  2. followed by multiple lines of time dependent observations (the first column is just timestamp and not used for computation)


## Output

   - result_draw.json      -   the specified network inference output, for the purpose of easy drawing
   - network_results       -   output folder from the algorithm, network inferences


## Contact
If you have any questions about the code or data, please feel free to contact me.

