# Implementation

## Implementation

  - Walk through the notebook

## Target Networks

    - tf.assign / tf.Variable.assign for target networks

## Tensorflow and custom gradients

    - tf.gradients for action/policy gradient
    - Optimizer.minimize / Optimizer.apply_gradients specifities

```python
action_gradient, = tf.gradients(critic.y, actions)
policy_gradients = tf.gradients(actor.y, actor.vars, -action_gradient)
```

## Moar Tensorflow

    - variable scopes and `get_variables` helpers
    - Batch normalization (`training`?)
    - `tf.control_dependencies`
    - Threading? (not implemented as of now)

