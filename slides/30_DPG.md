# Deterministic Policy Gradient


## Policies in Continuous Action Space

Problem:

- The action value is in $\mathbb{R}$
- At every step this requires to evaluate the action-value function $Q$ globally over (at least a subset of) $\mathbb{R}$  
- This is infeasible

Solution: 

- Do not maximize over $Q$
- But move policy in direction of $Q$
- The policy is now deterministic, giving a real valued number

## Policies in Continuous Action Space

Specifically: 

$$ \theta^{k+1} = \theta + \alpha \mathbb{E}_{s \sim  \rho^{\mu^{k}}} \left[ \nabla_{\theta} Q^{\mu^{k}}(s, \mu_{\theta}(s)) \right] $$

where $\mu_{\theta}(s)$ is the deterministic policy

Now applying the chain rule:
$$ \theta^{k+1} = \theta + \alpha \mathbb{E}_{s \sim  \rho^{\mu^{k}}} \left[ \nabla_{\theta} \mu_{\theta}(s) \nabla_{a} Q^{\mu^{k}}(s,a)|_{a=\mu_{\theta}(s)} \right] $$

## Determinisitc Policy Gradient Theorem

The deterministic policy gradient now is: 

$$ \nabla_{\theta}J(\mu_{\theta}) = \mathbb{E}_{s \sim  \rho^{\mu}} \left[ \nabla_{\theta} \mu_{\theta}(s) \nabla_{a} Q^{\mu}(s,a)|_{a=\mu_{\theta}(s)} \right] $$


@Silver2014

