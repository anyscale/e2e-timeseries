# Time Series Forecasting on ETT Dataset Using Ray Distributed

<div align="left">
<a target="_blank" href="https://console.anyscale.com/"><img src="https://img.shields.io/badge/🚀%20Run%20on-Anyscale-9hf"></a>&nbsp;
<a href="https://github.com/anyscale/e2e-timeseries" role="button"><img src="https://img.shields.io/static/v1?label=&message=View%20On%20GitHub&color=586069&logo=github&labelColor=2f363d"></a>
</div>


## Overview

In this tutorial,
In this tutorial, we'll be implementing a time-series forecasting application that leverages batch inference, distributed training and online serving at scale.

TODO: update with links to ipynb
- `e2e_timeseries/train.py`: load the ETT dataset using [Ray Data](https://docs.ray.io/en/latest/data/data.html) to train a DLinear model using [Ray Train](https://docs.ray.io/en/latest/train/train.html).
- `e2e_timeseries/offline_inference.py`: Evaluate the DLinear model checkpoint by performing offline batch inference on test data with Ray Data.
- `e2e_timeseries/serving.py`: Serve the model using [Ray Serve](https://docs.ray.io/en/latest/serve/index.html) to provide live time series predictions.
- Create production batch [**Jobs**](https://docs.anyscale.com/platform/jobs/) for offline workloads (offline inference, model training, etc.) and production online [**Services**](https://docs.anyscale.com/platform/services/) that can scale.


### Development

We're developing our application on [Anyscale Workspaces](https://docs.anyscale.com/platform/workspaces/), which enables us to develop without thinking about infrastructure, just like we would on a laptop. Workspaces come with:
- **Development tools**: Spin up a remote session from your local IDE (cursor, vscode, etc.) and start coding, using the same tools you love but with the power of Anyscale's compute.
- **Dependencies**: Continue to install dependencies using familiar tools like pip. Anyscale will ensure dependencies are being propagated to your cluster.

```bash
pip install -q "torch==2.7.0" "numpy" "scikit-learn" "pandas" "aiohttp" "ipywidgets" "pyyaml"
```

- **Compute**: Leverage any reserved instance capacity, spot instance from any compute provider of your choice by deploying Anyscale into your account. Alternatively, you can use the Anyscale cloud for a full serverless experience.
  - Under the hood, a cluster will be spun up and smartly managed by Anyscale.
- **Debugging**: Leverage a [distributed debugger](https://docs.anyscale.com/platform/workspaces/workspaces-debugging/#distributed-debugger) to get the same VSCode-like debugging experience.

Learn more about Anyscale Workspaces through the [official documentation](https://docs.anyscale.com/platform/workspaces/).

<div align="center">
  <img src="https://raw.githubusercontent.com/anyscale/foundational-ray-app/refs/heads/main/images/compute.png" width=600>
</div>

**Note**: If you're on [Anyscale](https://console.anyscale.com/), you can run this entire tutorial for free (all dependencies are setup and the necessary compute will autoscale). Otherwise be sure to install the dependencies from the [`containerfile`](https://github.com/anyscale/foundational-ray-app/tree/main/containerfile) and provision the appropriate GPU resources.

### Production
Seamlessly integrate with your existing CI/CD pipelines by leveraging the Anyscale [CLI](https://docs.anyscale.com/reference/quickstart-cli) or [SDK](https://docs.anyscale.com/reference/quickstart-sdk) to deploy [highly available services](https://docs.anyscale.com/platform/services) and run [reliable batch jobs](https://docs.anyscale.com/platform/jobs). Given we've been developing in an environment that's almost identical to production (multinode cluster), this should drastically speed up our dev → prod velocity. We'll also learn about proprietary RayTurbo features to optimize our workloads for performance, fault tolerance, scale and observability.


### No infrastructure headaches
Abstract away infrastructure from your ML/AI developers so they can focus on their core ML development. You can additionally better manage compute resources and costs with our [enterprise governance and observability](https://www.anyscale.com/blog/enterprise-governance-observability) and [admin capabilities](https://docs.anyscale.com/administration/overview) so you can set [resource quotas](https://docs.anyscale.com/reference/resource-quotas/), set [priorities for different workloads](https://docs.anyscale.com/administration/cloud-deployment/global-resource-scheduler) and gain [observability of your utilization across your entire compute fleet](https://docs.anyscale.com/administration/resource-management/telescope-dashboard).
If you're already on a kubernetes cloud (EKS, GKE, etc.), then you can still leverage the proprietary optimizations from RayTubo you'll see in action in these tutorials through our [Anyscale K8s Operator](https://docs.anyscale.com/administration/cloud-deployment/kubernetes/).

## Acknowledgements

This repository is based on the official DLinear implementations:
- [DLinear](https://github.com/vivva/DLinear)
- [LTSF-Linear](https://github.com/cure-lab/LTSF-Linear)

And the original publication:
- ["Are Transformers Effective for Time Series Forecasting?"](https://arxiv.org/abs/2205.13504)
