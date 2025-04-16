# AI From Scratch

This repository contains implementations of various AI and machine learning concepts, architectures, and exercises built from scratch. It serves as a learning resource for understanding the underlying principles of artificial intelligence and machine learning algorithms.

## Repository Structure

### Architectures

This directory contains implementations of different neural network architectures and machine learning models.

- **convolutional_neural_network_reinforcement_learning_monte_carlo_tree_search_selfplay_alphazero_tictactoe.ipynb**: Implementation of a convolutional neural network with reinforcement learning using Monte Carlo Tree Search and self-play, similar to AlphaZero, applied to Tic-Tac-Toe.
- **logistic_regression.ipynb**: Implementation of logistic regression algorithm.
- **lstm.py**: Long Short-Term Memory neural network implementation.
- **diffusion.py**: Diffusion model implementation.
- **linear_regression.py**: Linear regression implementation.
- **bigram.py**: Bigram language model implementation.
- **multiple_linear_polynomical_sinus_etc_regression_and_gradient_descent.ipynb**: Implementation of various regression techniques including multiple linear, polynomial, and sine regression with gradient descent.
- **reinforcement-learning-deep-q-learning**: Implementation of Deep Q-learning reinforcement learning and applied to Snake game.
- **grpo_group_relative_policy_optimization.ipynb**: Implementation of Group Relative Policy Optimization (GRPO), a reinforcement learning algorithm that enhances LLM's reasoning abilities by generating multiple responses to a given prompt, evaluating each using a reward function (solution correctness in math), and updating the model based on the relative performance of these responses within the group

#### Transformer

- **transformer.py**: Implementation of the transformer architecture.
- **einops.py**: Implementation of einops operations for tensor manipulations.

##### GPT-2

- **train_gpttwo.py**: Training script for a GPT-2 style model.
- **play.ipynb**: Interactive notebook for playing with the trained GPT-2 model.
- **fineweb.py**: Fine-tuning utilities for web data.
- **hellaswag.py**: Implementation for the HellaSwag benchmark.
- **input.txt**: Training data for the model.

### Physics-Inspired Neural Networks

- **cooling/**: A collection of notebooks and code exploring physics-based applications of machine learning:
  - **temp_pred.ipynb**: Neural network models and physics informed neural network model for predicting temperature dynamics in cooling systems, including implementation of L2 regularization techniques.
  - **regularisation_ex.ipynb**: Demonstrates the application of regularization techniques in machine learning models to prevent overfitting, with visualizations comparing regularized vs. non-regularized polynomial regression.
  - **network.py**: Basic neural network architecture for solving physics-based problems.
  - **diff_equations.py**: Implementation of fundamental cooling law equations and gradient calculation for physics-based machine learning.

### Exercises

- **hyperplane_classifier_of_clothes.py**: Exercise implementation of a hyperplane classifier for clothing items.

## Getting Started

To use this repository, clone it to your local machine and explore the different implementations. Each file is self-contained and includes the necessary code to understand and run the respective algorithm or model.

```bash
git clone https://github.com/yourusername/ai-from-scratch.git
cd ai-from-scratch
```

## Prerequisites

- Python 3.x
- NumPy
- PyTorch (for some implementations)
- Jupyter Notebook (for running .ipynb files)

## License

This project is available for educational purposes.

## Acknowledgments

- Inspired by the desire to understand AI and machine learning concepts from first principles. 