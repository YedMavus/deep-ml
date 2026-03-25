import numpy as np
from typing import Tuple

def compute_qkv(X: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Query, Key, and Value matrices.
    
    Args:
        X: Input matrix of shape (seq_len, d_model)
        W_q, W_k, W_v: Weight matrices of shape (d_model, d_model)
    
    Returns:
        Q, K, V matrices each of shape (seq_len, d_model)
    """
    # Your code here
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v
    return Q,K,V

def self_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Compute scaled dot-product self-attention.
    
    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_k)
    
    Returns:
        Attention output of shape (seq_len, d_k)
    """
    # Your code here
    d = Q.shape[1]
    scores = Q @ K.T /np.sqrt(d)
    scores = scores - np.max(scores,axis=1, keepdims = True)
    exp_scores = np.exp(scores)
    attn = exp_scores/np.sum(exp_scores,axis=1, keepdims = True)
    return attn @ V

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, n_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    
    Args:
        Q, K, V: Matrices of shape (seq_len, d_model)
        n_heads: Number of attention heads
    
    Returns:
        Attention output of shape (seq_len, d_model)
    """
    # Your code here
    seq_len, d_model = Q.shape
    d_k = d_model//n_heads
    Q = Q.reshape(seq_len, n_heads, d_k)
    Q = Q.transpose(1,0,2)
    K = K.reshape(seq_len, n_heads, d_k)
    K = K.transpose(1,0,2)
    V = V.reshape(seq_len, n_heads, d_k)
    V = V.transpose(1,0,2)

    output = []
    for h in range(n_heads):
        output.append(self_attention(Q[h],K[h],V[h]))

    out = np.stack(output, axis=0)
    out = out.transpose(1,0,2)
    out = out.reshape(seq_len, d_model)
    return out
    
