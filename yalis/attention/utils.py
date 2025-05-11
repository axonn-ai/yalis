import torch

# This function has been taken from HuggingFace's transformers library: https://github.com/huggingface/transformers/blob/953196a43dae6a3c474165fba7d215fcbc7b7730/src/transformers/integrations/sdpa_attention.py#L18
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


#@torch.compile()
def fit_powerlaw_linreg_torch(x: torch.Tensor, y: torch.Tensor):
    """
    Fits a power-law model using a linear regression on the logarithm of data.
    
    The model is of the form: y = a * x^b, where:
        log(y) = log(a) + b * log(x + 1)
    
    Parameters:
        x (torch.Tensor): Input tensor for x values.
        y (torch.Tensor): Input tensor for y values.
    
    Returns:
        a (torch.Tensor): Parameter 'a' of the power-law (after applying exp to the intercept).
        b (torch.Tensor): Parameter 'b' of the power-law (the slope).
        r2 (torch.Tensor): R² score of the regression fit.
    """
    x_dtype = x.dtype
    x = x.to(torch.float32)  # Ensure x is float32
    y = y.to(torch.float32)  # Ensure y is float32
    epsilon = 1e-8

    # Compute the log-transformed variables.
    # Here x is shifted by 1 to avoid log(0), and y is offset with epsilon.
    X = torch.log(x + 1)    # Shape: [B, H, N]
    Y = torch.log(y + epsilon)  # Shape: [B, H, N]

    #print (f"{X=}")
    #print (f"{Y=}")
    
    
    # Compute the means along the last dimension (i.e. across the N values)
    X_mean = torch.mean(X, dim=-1, keepdim=True)  # Shape: [B, H, 1]
    Y_mean = torch.mean(Y, dim=-1, keepdim=True)  # Shape: [B, H, 1]

    # Calculate the slope (b) using the formula:
    #   slope = Σ[(X - X_mean) * (Y - Y_mean)] / Σ[(X - X_mean)^2]
    numerator = torch.sum((X - X_mean) * (Y - Y_mean), dim=-1)  # Shape: [B, H]
    denominator = torch.sum((X - X_mean)**2, dim=-1)             # Shape: [B, H]
    #print (f"{numerator=}")
    #print (f"{denominator=}")


    slope = numerator / denominator                             # Shape: [B, H]
    #print (f"{slope=}")

    # Calculate the intercept in log-space using:
    #   intercept = Y_mean - slope * X_mean
    # Squeeze the last dimension since X_mean and Y_mean have shape [B, H, 1]
    intercept = Y_mean.squeeze(-1) - slope * X_mean.squeeze(-1)  # Shape: [B, H]
    #print (f"{intercept=}")
    
    # Use broadcasting to compute predictions: add a singleton dim for slope and intercept.
    intercept_unsq = intercept.unsqueeze(-1)  # Shape: [B, H, 1]
    slope_unsq = slope.unsqueeze(-1)          # Shape: [B, H, 1]
    y_pred = intercept_unsq + slope_unsq * X   # Shape: [B, H, N]
    
    # Calculate the residual sum of squares (SS_res) and total sum of squares (SS_tot)
    ss_res = torch.sum((Y - y_pred) ** 2, dim=-1)  # Shape: [B, H]
    ss_tot = torch.sum((Y - torch.mean(Y, dim=-1, keepdim=True))**2, dim=-1)  # Shape: [B, H]
    
    # Compute R² score safely, using torch.where to handle cases when ss_tot is zero.
    r2 = torch.where(ss_tot == 0, torch.ones_like(ss_tot), 1.0 - ss_res / ss_tot)
    
    # Convert the intercept back to coefficient a by exponentiating it.
    a = torch.exp(intercept)  # Shape: [B, H]
    b = slope                 # Shape: [B, H]

    return a.to(x_dtype), b.to(x_dtype), r2.to(x_dtype)