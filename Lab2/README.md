# Directory for the lab 2 of the computational intelligence course
The directory contains an algorithm for $(\mu, \lambda)$ and $(\mu + \lambda)$ evolutionary strategies.
In addition, the code sets a temperature parameter for transiting from an exploration-prevalent phase to a exploitation-prevalent phase.
In particular, exploitation with probability $P[U(0,1)<(\frac{\text{generation}}{\text{total generations}})^t]$, whereas exploration is performed with $P[U(0,1)>(\frac{\text{generation}}{\text{total generations}})^t]$.