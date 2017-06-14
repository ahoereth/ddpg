# Deterministic Policy Gradient


## Policies in continuos action space

Problem:

-  The action value is in $\mathbb{R}$
- At every step this requires to evaluate the action-value function $Q$ globally over (at least a subset of) $\mathbb{R}$  
- This is infeasible

Solution: 

- Do not maximize over $Q$
- but move policy in direction of $Q$
- The policy is now deterministic, giving a real valued number

## Policies in continuos action space

Specifically: 

$$ \theta^{k+1} = \theta + \alpha \mathbb{E}_{s ~ \rho^{\mu^{k}}} \left[ \nabla_{\theta} Q^{\mu^{k}}(s, \mu_{\theta}(s)) \right] $$

where $\mu_{\theta}(s)$ is the deterministic policy

Now applying the chain rule:
$$ \theta^{k+1} = \theta + \alpha \mathbb{E}_{s ~ \rho^{\mu^{k}}} \left[ \nabla_{\theta} \mu_{\theta}(s) \nabla_{a} Q^{\mu^{k}}(s,a)|_{a=\mu_{\theta}(s)} \right] $$

## Determinisitc policy gradient theorem

The deterministic policy gradient now is: 

$$ \nabla_{\theta}J(\mu_{\theta}) = \mathbb{E}_{s ~ \rho^{\mu}} \left[ \nabla_{\theta} \mu_{\theta}(s) \nabla_{a} Q^{\mu}(s,a)|_{a=\mu_{\theta}(s)} \right] $$


@Silver2014

