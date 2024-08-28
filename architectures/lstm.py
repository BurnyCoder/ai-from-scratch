import numpy as np


def lstm_cell_forward(xt, a_prev, c_prev, parameters):
    """
    Arguments:
    xt -- your input data at timestep "t", numpy array of shape (n_x, m).
    a_prev -- Hidden state at timestep "t-1", numpy array of shape (n_a, m)
    c_prev -- Memory state at timestep "t-1", numpy array of shape (n_a, m)
    parameters -- python dictionary containing:
                        Wf -- Weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
                        bf -- Bias of the forget gate, numpy array of shape (n_a, 1)
                        Wi -- Weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
                        bi -- Bias of the update gate, numpy array of shape (n_a, 1)
                        Wc -- Weight matrix of the first "tanh", numpy array of shape (n_a, n_a + n_x)
                        bc --  Bias of the first "tanh", numpy array of shape (n_a, 1)
                        Wo -- Weight matrix of the output gate, numpy array of shape (n_a, n_a + n_x)
                        bo --  Bias of the output gate, numpy array of shape (n_a, 1)
                        Wy -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)

    Returns:
    a_next -- next hidden state, of shape (n_a, m)
    c_next -- next memory state, of shape (n_a, m)
    yt_pred -- prediction at timestep "t", numpy array of shape (n_y, m)
    cache -- tuple of values needed for the backward pass, contains (a_next, c_next, a_prev, c_prev, xt, parameters)

    Note: ft/it/ot stand for the forget/update/output gates, cct stands for the candidate value (c tilde),
          c stands for the memory value
    """

    # Retrieve parameters from "parameters"
    Wf = parameters["Wf"]  # Weight matrix of the forget gate, shape (n_a, n_a + n_x)
    bf = parameters["bf"]  # Bias of the forget gate, shape (n_a, 1)
    Wi = parameters["Wi"]  # Weight matrix of the input/update gate, shape (n_a, n_a + n_x)
    bi = parameters["bi"]  # Bias of the input/update gate, shape (n_a, 1)
    Wc = parameters["Wc"]  # Weight matrix of the candidate memory cell, shape (n_a, n_a + n_x)
    bc = parameters["bc"]  # Bias of the candidate memory cell, shape (n_a, 1)
    Wo = parameters["Wo"]  # Weight matrix of the output gate, shape (n_a, n_a + n_x)
    bo = parameters["bo"]  # Bias of the output gate, shape (n_a, 1)
    Wy = parameters["Wy"]  # Weight matrix for output, shape (n_y, n_a)
    by = parameters["by"]  # Bias for output, shape (n_y, 1)

    # Retrieve dimensions from shapes of xt and Wy
    n_x, m = xt.shape  # n_x: input dimension, m: batch size
    n_y, n_a = Wy.shape  # n_y: output dimension, n_a: number of hidden units

    # Explanation:
    # This code is creating a concatenated input for the LSTM cell.
    # 1. We create a new array 'concat' with dimensions (n_a + n_x, m),
    #    where n_a is the number of hidden units, n_x is the input dimension,
    #    and m is the batch size.
    # 2. We fill the first n_a rows of 'concat' with a_prev, which is the
    #    previous hidden state. This allows the LSTM to consider its previous state.
    # 3. We fill the remaining rows (from index n_a onwards) with xt, which is
    #    the current input.
    # This concatenation combines the previous hidden state with the current input,
    # allowing the LSTM to process both simultaneously in its various gates.

    # Concatenate a_prev and xt
    concat = np.zeros((n_a + n_x, m))  # shape (n_a + n_x, m)
    # n_a: number of hidden units, n_x: input dimension, m: batch size
    concat[: n_a, :] = a_prev  # a_prev: previous hidden state, shape (n_a, m)
    concat[n_a:, :] = xt  # xt: current input, shape (n_x, m)

    # Compute values for ft, it, cct, c_next, ot, a_next
    # Example of np.matmul(Wf, concat) + bf:
    # If Wf.shape = (100, 150), concat.shape = (150, 64), and bf.shape = (100, 1)
    # Then np.matmul(Wf, concat).shape = (100, 64)
    # And (np.matmul(Wf, concat) + bf).shape = (100, 64)
    ft = sigmoid(np.matmul(Wf, concat) + bf)  # ft: forget gate, shape (n_a, m)
    # n_a: number of hidden units, m: batch size
    it = sigmoid(np.matmul(Wi, concat) + bi)  # it: input/update gate, shape (n_a, m)
    cct = np.tanh(np.matmul(Wc, concat) + bc)  # cct: candidate memory cell, shape (n_a, m)
    c_next = (ft * c_prev) + (it * cct)  # c_next: next cell state, shape (n_a, m)
    ot = sigmoid(np.matmul(Wo, concat) + bo)  # ot: output gate, shape (n_a, m)
    a_next = ot * np.tanh(c_next)  # a_next: next hidden state, shape (n_a, m)

    # Compute prediction of the LSTM cell
    yt_pred = softmax(np.matmul(Wy, a_next) + by)  # yt_pred: predicted output, shape (n_y, m)
    # n_y: output dimension, m: batch size

    # store values needed for backward propagation in cache
    cache = (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt, parameters)

    return a_next, c_next, yt_pred, cache


def lstm_cell_backward(da_next, dc_next, cache):
    """
    Arguments:
    da_next -- Gradients of next hidden state, of shape (n_a, m)
    dc_next -- Gradients of next cell state, of shape (n_a, m)
    cache -- cache storing information from the forward pass

    Returns:
    gradients -- python dictionary containing:
                        dxt -- Gradient of input data at time-step t, of shape (n_x, m)
                        da_prev -- Gradient w.r.t. the previous hidden state, numpy array of shape (n_a, m)
                        dc_prev -- Gradient w.r.t. the previous memory state, of shape (n_a, m, T_x)
                        dWf -- Gradient w.r.t. the weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
                        dWi -- Gradient w.r.t. the weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
                        dWc -- Gradient w.r.t. the weight matrix of the memory gate, numpy array of shape (n_a, n_a + n_x)
                        dWo -- Gradient w.r.t. the weight matrix of the output gate, numpy array of shape (n_a, n_a + n_x)
                        dbf -- Gradient w.r.t. biases of the forget gate, of shape (n_a, 1)
                        dbi -- Gradient w.r.t. biases of the update gate, of shape (n_a, 1)
                        dbc -- Gradient w.r.t. biases of the memory gate, of shape (n_a, 1)
                        dbo -- Gradient w.r.t. biases of the output gate, of shape (n_a, 1)
    """

    # Retrieve information from "cache"
    (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt, parameters) = cache

    # Retrieve dimensions from xt's and a_next's shape
    n_x, m = xt.shape
    n_a, m = a_next.shape

    # Compute gates related derivatives
    dot = da_next * np.tanh(c_next) * ot * (1 - ot)
    dcct = (dc_next * it + ot * (1 - np.square(np.tanh(c_next))) * it * da_next) * (1 - np.square(cct))
    dit = (dc_next * cct + ot * (1 - np.square(np.tanh(c_next))) * cct * da_next) * it * (1 - it)
    dft = (dc_next * c_prev + ot * (1 - np.square(np.tanh(c_next))) * c_prev * da_next) * ft * (1 - ft)

    concat = np.concatenate((a_prev, xt), axis=0)

    # Compute parameters related derivatives.
    dWf = np.dot(dft, concat.T)
    dWi = np.dot(dit, concat.T)
    dWc = np.dot(dcct, concat.T)
    dWo = np.dot(dot, concat.T)
    dbf = np.sum(dft, axis=1, keepdims=True)
    dbi = np.sum(dit, axis=1, keepdims=True)
    dbc = np.sum(dcct, axis=1, keepdims=True)
    dbo = np.sum(dot, axis=1, keepdims=True)

    # Compute derivatives w.r.t previous hidden state, previous memory state and input. Use equations (15)-(17). (≈3 lines)
    da_prev = np.dot(parameters['Wf'][:, :n_a].T, dft) + np.dot(parameters['Wi'][:, :n_a].T, dit) + np.dot(
        parameters['Wc'][:, :n_a].T, dcct) + np.dot(parameters['Wo'][:, :n_a].T, dot)
    dc_prev = dc_next * ft + ot * (1 - np.square(np.tanh(c_next))) * ft * da_next
    dxt = np.dot(parameters['Wf'][:, n_a:].T, dft) + np.dot(parameters['Wi'][:, n_a:].T, dit) + np.dot(
        parameters['Wc'][:, n_a:].T, dcct) + np.dot(parameters['Wo'][:, n_a:].T, dot)

    # Save gradients in dictionary
    gradients = {"dxt": dxt, "da_prev": da_prev, "dc_prev": dc_prev, "dWf": dWf, "dbf": dbf, "dWi": dWi, "dbi": dbi,
                 "dWc": dWc, "dbc": dbc, "dWo": dWo, "dbo": dbo}

    return gradients

def lstm_forward(x, a0, parameters):
    """
    Arguments:
    x -- Input data for every time-step, of shape (n_x, m, T_x).
    a0 -- Initial hidden state, of shape (n_a, m)
    parameters -- python dictionary containing:
                        Wf -- Weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
                        bf -- Bias of the forget gate, numpy array of shape (n_a, 1)
                        Wi -- Weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
                        bi -- Bias of the update gate, numpy array of shape (n_a, 1)
                        Wc -- Weight matrix of the first "tanh", numpy array of shape (n_a, n_a + n_x)
                        bc -- Bias of the first "tanh", numpy array of shape (n_a, 1)
                        Wo -- Weight matrix of the output gate, numpy array of shape (n_a, n_a + n_x)
                        bo -- Bias of the output gate, numpy array of shape (n_a, 1)
                        Wy -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)

    Returns:
    a -- Hidden states for every time-step, numpy array of shape (n_a, m, T_x)
    y -- Predictions for every time-step, numpy array of shape (n_y, m, T_x)
    caches -- tuple of values needed for the backward pass, contains (list of all the caches, x)
    """

    # Initialize "caches", which will track the list of all the caches
    caches = []

    # Retrieve dimensions from shapes of x and parameters['Wy']
    n_x, m, T_x = x.shape
    n_y, n_a = parameters["Wy"].shape

    # Initialize "a", "c" and "y" with zeros
    # a: hidden states, c: cell states, y: predictions
    # Example: If n_a = 100, m = 64, T_x = 10, n_y = 5
    # a.shape = (100, 64, 10), c.shape = (100, 64, 10), y.shape = (5, 64, 10)
    a = np.zeros((n_a, m, T_x))
    c = np.zeros_like(a)
    y = np.zeros((n_y, m, T_x))

    # Initialize a_next and c_next
    # a_next: next hidden state, c_next: next cell state
    # Example: If n_a = 100, m = 64
    # a_next.shape = (100, 64), c_next.shape = (100, 64)
    a_next = a0
    c_next = np.zeros_like(a_next)

    # Loop over all time-steps
    for t in range(T_x):
        # Update next hidden state (a_next), next memory state (c_next), compute the prediction (yt), get the cache
        # The next hidden state (a_next) is the output of the LSTM cell at this time step,
        # which will be used as input for the next time step and for making predictions.
        # The next memory state (c_next) is the internal cell state of the LSTM,
        # which carries long-term information and is updated at each time step.
        # While a_next is used directly in computations, c_next is only used internally by the LSTM cell.
        # x[:,:,t] is the input at time step t, shape (n_x, m)
        a_next, c_next, yt, cache = lstm_cell_forward(x[:, :, t], a_next, c_next, parameters)
        
        # Save the value of the new "next" hidden state in a
        a[:, :, t] = a_next
        # This line stores the current hidden state (a_next) in the 'a' array at time step t.
        # 'a' is a 3D array where each slice along the third dimension represents the hidden state at a particular time step.
        
        # Save the value of the prediction in y
        y[:, :, t] = yt
        # This line stores the current prediction (yt) in the 'y' array at time step t.
        # 'y' is a 3D array where each slice along the third dimension represents the prediction at a particular time step.
        
        # Save the value of the next cell state
        c[:, :, t] = c_next
        # This line stores the current cell state (c_next) in the 'c' array at time step t.
        # 'c' is a 3D array where each slice along the third dimension represents the cell state at a particular time step.
        
        # These operations are crucial for maintaining the state of the LSTM across all time steps.
        # They allow us to keep track of how the hidden state, predictions, and cell state evolve over time,
        # which is essential for both the forward pass and subsequent backward pass during training.
        
        # Append the cache into caches
        caches.append(cache)

    # Store values needed for backward propagation in cache
    caches = (caches, x)

    # Explanation of how LSTM predicts and the role of time:
    
    # 1. Sequential Processing:
    # LSTM processes input sequentially, one time step at a time. This is why we have a loop over T_x (total time steps).
    # At each time step t, the LSTM cell takes in:
    #   - The current input x[:,:,t]
    #   - The previous hidden state a_next (initially a0)
    #   - The previous cell state c_next (initially zeros)
    
    # 2. Hidden State and Cell State:
    # The LSTM maintains two states: hidden state (a) and cell state (c).
    # These states carry information through time, allowing the network to capture long-term dependencies.
    # a[:,:,t] represents the hidden state at time t
    # c[:,:,t] represents the cell state at time t
    
    # 3. Prediction at Each Time Step:
    # At each time step, the LSTM cell produces a prediction yt based on the current input and states.
    # y[:,:,t] = yt represents the prediction at time t
    # This allows the LSTM to make a separate prediction for each time step in the sequence.
    
    # 4. Time's Role in Prediction:
    # Time plays a crucial role in how information is processed and predictions are made:
    #   a. Past Context: Earlier time steps influence later ones through the evolving hidden and cell states.
    #   b. Sequence Order: The order of inputs matters; changing the order can lead to different predictions.
    #   c. Temporal Dependencies: LSTM can capture both short-term and long-term temporal dependencies in the data.
    
    # 5. Final Output:
    # After processing all time steps:
    #   - 'a' contains hidden states for all time steps: shape (n_a, m, T_x)
    #   - 'c' contains cell states for all time steps: shape (n_a, m, T_x)
    #   - 'y' contains predictions for all time steps: shape (n_y, m, T_x)
    
    # This allows for various use cases:
    # - Use the final prediction y[:,:,-1] for tasks that need a single output after seeing the whole sequence.
    # - Use the entire 'y' for tasks that require a prediction at each time step (e.g., time series forecasting).
    # - Use the final hidden state a[:,:,-1] as a fixed-length representation of the entire input sequence.
    # Explanation of how the context window works in LSTMs:
    
    # 1. Context Window Concept:
    # In LSTMs, the context window is not a fixed-size sliding window like in some other models.
    # Instead, it's a dynamic and adaptive window that can theoretically span the entire sequence.
    
    # 2. Hidden State and Cell State:
    # The hidden state (a) and cell state (c) act as the "context window" in LSTMs.
    # These states are updated at each time step and carry information from previous time steps.
    
    # 3. Adaptive Context:
    # - The LSTM can learn to keep or forget information in its states.
    # - This allows it to maintain relevant information for long periods and discard irrelevant information.
    # - The effective size of the "context window" can vary based on the data and what the model learns.
    
    # 4. Gating Mechanism:
    # - Forget gate (ft): Decides what information to discard from the cell state.
    # - Input gate (it): Decides which values to update in the cell state.
    # - Output gate (ot): Controls what parts of the cell state are output to the hidden state.
    # These gates allow the LSTM to control the flow of information through time.
    
    # 5. Long-term Dependencies:
    # Unlike fixed-size windows, LSTMs can theoretically capture dependencies across very long sequences.
    # The cell state can carry relevant information across many time steps if needed.
    
    # 6. Bidirectional LSTMs:
    # In bidirectional LSTMs, the context window effectively includes both past and future context,
    # as the model processes the sequence in both forward and backward directions.
    
    # 7. Practical Limitations:
    # While theoretically the context can span the entire sequence, in practice:
    # - Very long sequences might still pose challenges due to vanishing gradients.
    # - The model's capacity to retain information is finite and depends on its architecture and training.
    
    # In summary, the "context window" in LSTMs is not a fixed window but an adaptive mechanism
    # that allows the model to selectively retain and use information from across the entire sequence.
    return a, y, c, caches


def lstm_backward(da, caches):
    """
    Arguments:
    da -- Gradients w.r.t the hidden states, numpy-array of shape (n_a, m, T_x)
    dc -- Gradients w.r.t the memory states, numpy-array of shape (n_a, m, T_x)
    caches -- cache storing information from the forward pass (lstm_forward)

    Returns:
    gradients -- python dictionary containing:
                        dx -- Gradient of inputs, of shape (n_x, m, T_x)
                        da0 -- Gradient w.r.t. the previous hidden state, numpy array of shape (n_a, m)
                        dWf -- Gradient w.r.t. the weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
                        dWi -- Gradient w.r.t. the weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
                        dWc -- Gradient w.r.t. the weight matrix of the memory gate, numpy array of shape (n_a, n_a + n_x)
                        dWo -- Gradient w.r.t. the weight matrix of the save gate, numpy array of shape (n_a, n_a + n_x)
                        dbf -- Gradient w.r.t. biases of the forget gate, of shape (n_a, 1)
                        dbi -- Gradient w.r.t. biases of the update gate, of shape (n_a, 1)
                        dbc -- Gradient w.r.t. biases of the memory gate, of shape (n_a, 1)
                        dbo -- Gradient w.r.t. biases of the save gate, of shape (n_a, 1)
    """

    # Retrieve values from the first cache (t=1) of caches.
    (caches, x) = caches
    (a1, c1, a0, c0, f1, i1, cc1, o1, x1, parameters) = caches[0]

    # Retrieve dimensions from da's and x1's shapes
    n_a, m, T_x = da.shape
    n_x, m = x1.shape

    # initialize the gradients with the right sizes
    dx = np.zeros((n_x, m, T_x))
    da0 = np.zeros((n_a, m))
    da_prevt = np.zeros(da0.shape)
    dc_prevt = np.zeros(da0.shape)
    dWf = np.zeros((n_a, n_a + n_x))
    dWi = np.zeros(dWf.shape)
    dWc = np.zeros(dWf.shape)
    dWo = np.zeros(dWf.shape)
    dbf = np.zeros((n_a, 1))
    dbi = np.zeros(dbf.shape)
    dbc = np.zeros(dbf.shape)
    dbo = np.zeros(dbf.shape)

    # loop back over the whole sequence
    for t in reversed(range(T_x)):
        # Compute all gradients using lstm_cell_backward
        gradients = lstm_cell_backward(da[:, :, t], dc_prevt, caches[t])
        # Store or add the gradient to the parameters' previous step's gradient
        dx[:, :, t] = gradients["dxt"]
        dWf += gradients["dWf"]
        dWi += gradients["dWi"]
        dWc += gradients["dWc"]
        dWo += gradients["dWo"]
        dbf += gradients["dbf"]
        dbi += gradients["dbi"]
        dbc += gradients["dbc"]
        dbo += gradients["dbo"]
    # Set the first activation's gradient to the backpropagated gradient da_prev.
    da0 = gradients["da_prev"]

    # Store the gradients in a python dictionary
    gradients = {"dx": dx, "da0": da0, "dWf": dWf, "dbf": dbf, "dWi": dWi, "dbi": dbi,
                 "dWc": dWc, "dbc": dbc, "dWo": dWo, "dbo": dbo}

    return gradients

import numpy as np

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def initialize_adam(parameters) :
    """
    Initializes v and s as two python dictionaries with:
                - keys: "dW1", "db1", ..., "dWL", "dbL" 
                - values: numpy arrays of zeros of the same shape as the corresponding gradients/parameters.
    
    Arguments:
    parameters -- python dictionary containing your parameters.
                    parameters["W" + str(l)] = Wl
                    parameters["b" + str(l)] = bl
    
    Returns: 
    v -- python dictionary that will contain the exponentially weighted average of the gradient.
                    v["dW" + str(l)] = ...
                    v["db" + str(l)] = ...
    s -- python dictionary that will contain the exponentially weighted average of the squared gradient.
                    s["dW" + str(l)] = ...
                    s["db" + str(l)] = ...

    """
    
    L = len(parameters) // 2 # number of layers in the neural networks
    v = {}
    s = {}
    
    # Initialize v, s. Input: "parameters". Outputs: "v, s".
    for l in range(L):
        v["dW" + str(l+1)] = np.zeros(parameters["W" + str(l+1)].shape)
        v["db" + str(l+1)] = np.zeros(parameters["b" + str(l+1)].shape)
        s["dW" + str(l+1)] = np.zeros(parameters["W" + str(l+1)].shape)
        s["db" + str(l+1)] = np.zeros(parameters["b" + str(l+1)].shape)
    
    return v, s


def update_parameters_with_adam(parameters, grads, v, s, t, learning_rate = 0.01,
                                beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8):
    """
    Update parameters using Adam
    
    Arguments:
    parameters -- python dictionary containing your parameters:
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl
    grads -- python dictionary containing your gradients for each parameters:
                    grads['dW' + str(l)] = dWl
                    grads['db' + str(l)] = dbl
    v -- Adam variable, moving average of the first gradient, python dictionary
    s -- Adam variable, moving average of the squared gradient, python dictionary
    learning_rate -- the learning rate, scalar.
    beta1 -- Exponential decay hyperparameter for the first moment estimates 
    beta2 -- Exponential decay hyperparameter for the second moment estimates 
    epsilon -- hyperparameter preventing division by zero in Adam updates

    Returns:
    parameters -- python dictionary containing your updated parameters 
    v -- Adam variable, moving average of the first gradient, python dictionary
    s -- Adam variable, moving average of the squared gradient, python dictionary
    """
    
    L = len(parameters) // 2                 # number of layers in the neural networks
    v_corrected = {}                         # Initializing first moment estimate, python dictionary
    s_corrected = {}                         # Initializing second moment estimate, python dictionary
    
    # Perform Adam update on all parameters
    for l in range(L):
        # Moving average of the gradients. Inputs: "v, grads, beta1". Output: "v".
        v["dW" + str(l+1)] = beta1 * v["dW" + str(l+1)] + (1 - beta1) * grads["dW" + str(l+1)] 
        v["db" + str(l+1)] = beta1 * v["db" + str(l+1)] + (1 - beta1) * grads["db" + str(l+1)]

        # Compute bias-corrected first moment estimate. Inputs: "v, beta1, t". Output: "v_corrected".
        v_corrected["dW" + str(l+1)] = v["dW" + str(l+1)] / (1 - beta1**t)
        v_corrected["db" + str(l+1)] = v["db" + str(l+1)] / (1 - beta1**t)

        # Moving average of the squared gradients. Inputs: "s, grads, beta2". Output: "s".
        s["dW" + str(l+1)] = beta2 * s["dW" + str(l+1)] + (1 - beta2) * (grads["dW" + str(l+1)] ** 2)
        s["db" + str(l+1)] = beta2 * s["db" + str(l+1)] + (1 - beta2) * (grads["db" + str(l+1)] ** 2)

        # Compute bias-corrected second raw moment estimate. Inputs: "s, beta2, t". Output: "s_corrected".
        s_corrected["dW" + str(l+1)] = s["dW" + str(l+1)] / (1 - beta2 ** t)
        s_corrected["db" + str(l+1)] = s["db" + str(l+1)] / (1 - beta2 ** t)

        # Update parameters. Inputs: "parameters, learning_rate, v_corrected, s_corrected, epsilon". Output: "parameters".
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * v_corrected["dW" + str(l+1)] / np.sqrt(s_corrected["dW" + str(l+1)] + epsilon)
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * v_corrected["db" + str(l+1)] / np.sqrt(s_corrected["db" + str(l+1)] + epsilon)

    return parameters, v, s


def test_lstm():
    # Define LSTM parameters
    n_a = 64  # Number of units in the LSTM cell
    n_x = 10  # Input dimension
    n_y = 5   # Output dimension
    m = 32    # Batch size
    T_x = 15  # Sequence length

    # Initialize parameters
    np.random.seed(1)
    x = np.random.randn(n_x, m, T_x)
    a0 = np.random.randn(n_a, m)
    Wf = np.random.randn(n_a, n_a + n_x)
    bf = np.random.randn(n_a, 1)
    Wi = np.random.randn(n_a, n_a + n_x)
    bi = np.random.randn(n_a, 1)
    Wc = np.random.randn(n_a, n_a + n_x)
    bc = np.random.randn(n_a, 1)
    Wo = np.random.randn(n_a, n_a + n_x)
    bo = np.random.randn(n_a, 1)
    Wy = np.random.randn(n_y, n_a)
    by = np.random.randn(n_y, 1)

    parameters = {"Wf": Wf, "Wi": Wi, "Wo": Wo, "Wc": Wc, "Wy": Wy, "bf": bf, "bi": bi, "bo": bo, "bc": bc, "by": by}

    # Forward pass
    a, y, c, caches = lstm_forward(x, a0, parameters)

    # Print output shapes
    print("a.shape =", a.shape)
    print("y.shape =", y.shape)
    print("c.shape =", c.shape)

    # Backward pass
    da = np.random.randn(n_a, m, T_x)
    gradients = lstm_backward(da, caches)

    # Print gradients
    print("gradients[\"dx\"].shape =", gradients["dx"].shape)
    print("gradients[\"da0\"].shape =", gradients["da0"].shape)
    print("gradients[\"dWf\"].shape =", gradients["dWf"].shape)
    print("gradients[\"dWi\"].shape =", gradients["dWi"].shape)
    print("gradients[\"dWc\"].shape =", gradients["dWc"].shape)
    print("gradients[\"dWo\"].shape =", gradients["dWo"].shape)
    print("gradients[\"dbf\"].shape =", gradients["dbf"].shape)
    print("gradients[\"dbi\"].shape =", gradients["dbi"].shape)
    print("gradients[\"dbc\"].shape =", gradients["dbc"].shape)
    print("gradients[\"dbo\"].shape =", gradients["dbo"].shape)

    print("Test completed successfully!")

# Run the test
test_lstm()
