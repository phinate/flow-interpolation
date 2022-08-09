import torch


def train_step(batch, flow, optimizer, feature_scaler, context_scaler=None):
    features, weights, context = batch
    inputs = torch.tensor(feature_scaler.transform(features), dtype=torch.float32)
    weights = torch.tensor(weights, dtype=torch.float32)
    if context is not None:
        context = torch.tensor(context_scaler.transform(context), dtype=torch.float32)
    optimizer.zero_grad()
    loss = torch.mean(-flow.log_prob(inputs=inputs, context=context)*weights)
    loss.backward()
    optimizer.step()
    return loss