import torch

def nll_loss(flow, inputs, context, weights=None):
    if weights is not None:
        return torch.mean(-flow.log_prob(inputs=inputs, context=context)*weights)
    return torch.mean(-flow.log_prob(inputs=inputs, context=context))

def train_loop(num_iter, flow, optimizer, batch_iterator, feature_scaler, use_weights = False, valid_set = None, context_scaler=None, metric = None):
    
    if valid_set is not None:
        if use_weights:
            valid_features, valid_weights, valid_context = valid_set
        else: 
            valid_features, valid_context = valid_set

    for i in range(num_iter):
        if use_weights:
            features, weights, context = next(batch_iterator)
            weights = torch.tensor(weights, dtype=torch.float32)
        else:
            features, context = next(batch_iterator)
            weights = None

        inputs = torch.tensor(feature_scaler.transform(features), dtype=torch.float32)

        if context is not None:
            context = torch.tensor(context_scaler.transform(context), dtype=torch.float32)

        optimizer.zero_grad()
        loss = nll_loss(flow, inputs, context, weights)
        loss.backward()
        optimizer.step()

        if valid_set is not None:
            valid_inputs = torch.tensor(feature_scaler.transform(valid_features), dtype=torch.float32)
            if use_weights:
                valid_weights = torch.tensor(valid_weights, dtype=torch.float32)
            else:
                valid_weights = None
            if valid_context is not None:
                valid_context = torch.tensor(context_scaler.transform(valid_context), dtype=torch.float32)
            with torch.no_grad():
                valid_loss = nll_loss(flow, valid_inputs, valid_context, valid_weights)
        if i % 100 == 0:
            print(f'iteration {i}:')
            print(f'train loss = {loss}')
            if metric is not None:
                print(f'train metric = {metric(flow, inputs, context, weights)}')
            if valid_set is not None:
                print(f'valid loss = {valid_loss}')
                if metric is not None:
                    print(f'valid metric = {metric(flow, valid_inputs, valid_context, valid_weights)}')

    return flow.state_dict()