import torch

def nll_loss(flow, inputs, context, weights=None):
    if weights is not None:
        return torch.mean(-flow.log_prob(inputs=inputs, context=context)*weights)
    return torch.mean(-flow.log_prob(inputs=inputs, context=context))

def train_loop(num_iter, flow, optimizer, batch_iterator, feature_scaler, use_weights = False, valid_set = None, context_scaler=None, metric = None, print_every = 1000):
    
    best_metric = 99999
    best_flow = None
    best_iter = 0
    losses = [], []
    if valid_set is not None:
        if use_weights:
            valid_features, valid_weights, valid_context = valid_set
        else: 
            valid_features, valid_context = valid_set

        valid_inputs = torch.tensor(feature_scaler.transform(valid_features), dtype=torch.float32)
        if use_weights:
            valid_weights = torch.tensor(valid_weights, dtype=torch.float32)
        else:
            valid_weights = None
        if valid_context is not None:
            valid_context = torch.tensor(context_scaler.transform(valid_context), dtype=torch.float32)

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
        loss = torch.mean(-flow.log_prob(inputs=inputs, context=context))
        
        loss.backward()
        optimizer.step()

        if valid_set is not None:
            with torch.no_grad():
                valid_loss = torch.mean(-flow.log_prob(inputs=valid_inputs, context=valid_context))
                losses[1].append(valid_loss) 
                if metric is not None:
                    valid_metric = metric(flow, valid_inputs, valid_context, valid_weights)
                else:
                    valid_metric = valid_loss
                if valid_metric < best_metric:
                    best_metric = valid_metric
                    best_flow = flow.state_dict()
                    best_iter = i
        if (i == 0) or ((i+1) % print_every == 0):
            print(f'iteration {i}:')
            print(f'train loss = {loss}')
            if metric is not None:
                print(f'train metric = {metric(flow, inputs, context, weights)}')
            if valid_set is not None:
                print(f'valid loss = {valid_loss}')
                if metric is not None:
                    print(f'valid metric = {metric(flow, valid_inputs, valid_context, valid_weights)}')

    return best_flow, best_metric, best_iter, losses