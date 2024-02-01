# Quixo
In this project I cooperated with Claudio Macaluso s317149. 
With him I discussed the drawbacks of QLearning and motivated me to follow other approaches.
## The approach
I employed q-learning, evolutionary strategy, island model and evolutionary bagging.
I got bad results with q-learning, while the other approaches worked well, with win rates above 85%.
Evolutionary bagging is a technique I developed to exploit the whole population to make decision, the intuition stems from random forests, which use an ensemble of weak learners to reduce their bias. Evolutionary bagging is faster to train and better than the other methods I tried.
## How to use the agents
In demo.ipynb there are the instructions to train, test, save and load the agents. Set the id attribute of the agent to 0 if testing by starting first or set it to 1 if testing by starting second.
