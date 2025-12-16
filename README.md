# Heat Equation Solver (AI-Assisted)


This project contains a small Python program that solves the 1D heat equation
using an explicit finite difference (Forward-Time Central-Space, FTCS) scheme.

The initial implementation was created with the help of an RAG AI assistant (https://rai.uni-stuttgart.de/) and then manually reviewed, tested, and corrected. The solver reads
simulation parameters from a JSON configuration file, runs the time integration,
and writes the final temperature distribution to a CSV file and a plot.
