# Lab 9
In this directory I store the code for lab9, I investigated the use of different approaches in reaching an optimal fitness for the black box problem presented:
- I tried with an evolution strategy, a genetic algorithm and two island models: a simple one and a hierarchical one.
- The crossover is a one cut crossover.
- The mutation modality is common to all my approaches, at mutation time every locus of the individual has a probability $p$ to mutate. I defaulted $p$ to 0.001 so that, in case of 1000 loci, the expected number of mutated loci is 1. I tried to adapt $p$ in such a way that it increases with a factor inversely proportional to the variance of the vector consisting of the maximum fitness of the last generations. In this way $p$ would increase in case the best solution is stuck on a local optimum. This approach didn't yield significant improvements.
- To reduce the number of fitness calls I attempted to calculate the fitness only in some generations, with children ineriting the fitness of their parents with some random noise. However I discarded this method since I think that a more sophisticated implicit evaluation system should be employed.
- I also tried to freeze some parameters that were common to top agents, however this turned out to be worsening the fitness. Presumably because of the complexity of the fitness landscape
