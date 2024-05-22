# LearnSPN

## Overview

This codebase implements the structure learning algorithm for the sum-product networks (SPNs).

## Structure Learning

Under the project directory, the codebase can be compiled and run as follows:

```bash
cd scripts/learnspn
./setup.bash
```

Please refer to the content of `scripts/learnspn/setup.bash` and ensure that all relevant parameters are properly set. Additionally, the custom dataset symlinks under `data` should be updated to point to the correct location of the target datasets. The generated SPN are stored under `output/learnspn`.

## Original README

README from the original authors starts here:

LearnSPN Version 1.0
6/17/13 Robert Gens rcg@cs.washington.edu

This is raw, unoptimized research code.  We provide the subroutines as described in the paper: pairwise independence (G-test) and online hard EM over a naive Bayes mixture model.  We will likely expand this code with other subroutines.

Below are several commands to run experiments as in the paper (run from the main directory).  Java bytecode is provided so you don't have to compile.

If you have any questions about the code or paper, don't hesitate to email me (rcg@cs.washington.edu).

**Run LearnSPN**
This command will run LearnSPN, iterate over a few smoothing values, and then output the SPN with highest validation log-likelihood to d12.spn .
java -cp bin exp.RunSLSPN DATA 12 GF 10 CP 0.6 INDEPINST 4 N d12.spn

**Compute the log-likelihood for the test set**
Shows the average LL and total time at the end.
java -cp bin exp.inference.SPNInfPLL DATA 12 N d12.spn

**Compute the pseudo log-likelihood for the test set**
Shows the average PLL and total time at the end.
java -cp bin exp.inference.SPNInfPLL DATA 12 N d12.spn

**Generate 1000 queries for each of 10 proportions from test set**
This will create a file for query settings (.q) and a corresponding file for evidence settings (.ev)
mkdir data/nltcs/
java -cp bin exp.inference.GenQEV DATA 12

**Run inference over the set of queries with 30% query and 20% evidence**
Lists the CLL of each instance. Shows the average CLL and total time at the end.
java -cp bin exp.inference.SPNInf N d12.spn Q nltcs/VE_Q0.30_E0.20.q EV nltcs/VE_Q0.30_E0.20.ev

**Run same queries as previous example but with marginal inference (CMLL)**
java -cp bin exp.inference.SPNInfCMLL N d12.spn Q nltcs/VE_Q0.30_E0.20.q EV nltcs/VE_Q0.30_E0.20.ev


Parameters:

"N" is the filename of the saved/loaded SPN

"INDEPINST" is just to prevent the algorithm from needlessly running pairwise independence tests on matrices with this many or fewer instances (in which case all variables are independent)

Grid search in paper:
Cluster penalty "CP": {0.2, 0.4, 0.6, 0.8}
Significance threshold "GF": {10 , 15} (corresponding to p-values 0.0015 and 0.0001, respectively)

Datasets "DATA"
0  EachMovie
1  MSWeb
2  KDD
6  Audio
7  Book
9  Jester
10 MSNBC
11 Netflix
12 NLTCS
13 Plants
19 Accidents
20 Ad
21 BBC
22 C20NG
23 CWebKB
24 DNA
25 Kosarak
26 Retail
27 Pumsb_Star
28 CR52