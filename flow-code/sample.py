__all__ = ("hists_from_flows_with_errors",)

import numpy as np
import torch


def _sample_flow(flow, feature_scaler, context_scaler=None, context=None, num_samples=10000):
    if (context is not None) and (context_scaler is not None):
        context = context_scaler.transform(context).astype("float32")
    with torch.no_grad():
        samples = flow.sample(num_samples, context=context).detach()
        return feature_scaler.inverse_transform(samples.reshape(-1, feature_scaler.n_features_in_)).reshape(
            len(context), num_samples, feature_scaler.n_features_in_
        )


def hist2d_from_flows_with_error(
    flow_list,  x_bins, y_bins, feature_scaler, context_scaler=None, context=None, num_samples=10000, density=False
):
    flow_samples = np.array([_sample_flow(flow, feature_scaler, context_scaler, context, num_samples) for flow in flow_list])

    if (len(np.array(x_bins).shape)) > 1:
        hists = [
                [
                    np.histogram2d(data[:, 0], data[:, 1], bins=bins, density=density)[0]
                    for data, bins in zip(
                        flow_data, zip(x_bins, y_bins)
                    )
                ]
                for flow_data in flow_samples
            ]
        
    else:
        hists = np.array(
            [
                [
                    np.histogram2d(
                        data[:, 0], data[:, 1], bins=[x_bins, y_bins], density=density
                    )[0]
                    for data in flow_data
                ]
                for flow_data in flow_samples
            ]
        )
    if len(hists) == 1:
        return hists[0], None 
    hist_avg = np.mean(hists, axis=0)  # shape: [len(truth_masses), num_samples, 2]
    hist_std = np.std(hists, axis=0)  # shape: [len(truth_masses), num_samples, 2]
    return hist_avg, hist_std
