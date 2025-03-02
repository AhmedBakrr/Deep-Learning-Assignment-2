import numpy as np

X = [1, 2, 3]
W_f, W_hf, b_f = 0.5, 0.1, 0
W_i, W_hi, b_i = 0.6, 0.2, 0
W_c, W_hc, b_c = 0.7, 0.3, 0
W_o, W_ho, b_o = 0.8, 0.4, 0
h_prev, C_prev = 0, 0

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def tanh(x):
    return np.tanh(x)

for t in range(len(X)):
    x_t = X[t]
    f_t = sigmoid(W_f * x_t + W_hf * h_prev + b_f)
    i_t = sigmoid(W_i * x_t + W_hi * h_prev + b_i)
    C_tilde_t = tanh(W_c * x_t + W_hc * h_prev + b_c)
    C_t = f_t * C_prev + i_t * C_tilde_t
    o_t = sigmoid(W_o * x_t + W_ho * h_prev + b_o)
    h_t = o_t * tanh(C_t)   
    h_prev, C_prev = h_t, C_t
    
    print(f"Time Step {t+1}:")
    print(f"  Forget gate (f_t): {f_t:.3f}")
    print(f"  Input gate (i_t): {i_t:.3f}")
    print(f"  Candidate cell state (C_tilde_t): {C_tilde_t:.3f}")
    print(f"  Cell state (C_t): {C_t:.3f}")
    print(f"  Output gate (o_t): {o_t:.3f}")
    print(f"  Hidden state (h_t): {h_t:.3f}")
    print()

W_y, b_y = 4, 0
y_pred = W_y * h_t + b_y
print(f"Predicted next value: {y_pred:.3f}")