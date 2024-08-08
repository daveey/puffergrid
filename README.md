# PufferGrid

PufferGrid is a fast GridWorld engine for Reinforcement Learning implemented in Cython.

## Features

- High-performance grid-based environments
- Customizable actions, events, and observations
- Easy integration with popular RL frameworks

## Installation

You can install PufferGrid using pip or from source.

### Using pip

The easiest way to install PufferGrid is using pip:

```
pip install puffergrid
```

### From Source

To install PufferGrid from source, follow these steps:

1. Clone the repository:
   ```
   git clone https://github.com/daveey/puffergrid.git
   cd puffergrid
   ```

2. Build and install the package:
   ```
   python setup.py build_ext --inplace
   pip install -e .
   ```

## Getting Started

The best way to understand how to create a PufferGrid environment is to look at a complete example. Check out the [`forage.pyx`](https://github.com/daveey/puffergrid/blob/main/examples/forage.pyx) file in the `examples` directory for a full implementation of a foraging environment.

Below is a step-by-step walkthrough of creating a similar environment, explaining each component along the way.

### Step 1: Define Game Objects

First, we'll define our game objects: Agent, Wall, and Tree.

```python
from puffergrid.grid_object cimport GridObject

cdef struct AgentProps:
    unsigned int energy
    unsigned int orientation
ctypedef GridObject[AgentProps] Agent

cdef struct WallProps:
    unsigned int hp
ctypedef GridObject[WallProps] Wall

cdef struct TreeProps:
    char has_fruit
ctypedef GridObject[TreeProps] Tree

cdef enum ObjectType:
    AgentT = 0
    WallT = 1
    TreeT = 2
```

### Step 2: Define Actions

Next, we'll define the actions our agents can take: Move, Rotate, and Eat.

```python
from puffergrid.action cimport ActionHandler, ActionArg

cdef class Move(ActionHandler):
    cdef char handle_action(self, unsigned int actor_id, GridObjectId actor_object_id, ActionArg arg):
        # Implementation details...

cdef class Rotate(ActionHandler):
    cdef char handle_action(self, unsigned int actor_id, GridObjectId actor_object_id, ActionArg arg):
        # Implementation details...

cdef class Eat(ActionHandler):
    cdef char handle_action(self, unsigned int actor_id, GridObjectId actor_object_id, ActionArg arg):
        # Implementation details...
```

### Step 3: Define Event Handlers

We'll create an event handler to reset trees after they've been eaten from.

```python
from puffergrid.event cimport EventHandler, EventArg

cdef class ResetTreeHandler(EventHandler):
    cdef void handle_event(self, GridObjectId obj_id, EventArg arg):
        # Implementation details...

cdef enum Events:
    ResetTree = 0
```

### Step 4: Define Observation Encoder

Create an observation encoder to define what agents can observe in the environment.

```python
from puffergrid.observation_encoder cimport ObservationEncoder

cdef class ObsEncoder(ObservationEncoder):
    cdef encode(self, GridObjectBase *obj, int[:] obs):
        # Implementation details...

    cdef vector[string] feature_names(self):
        return [
            "agent", "agent:energy", "agent:orientation",
            "wall", "tree", "tree:has_fruit"]
```

### Step 5: Define The Environment

Finally, we'll put it all together in our Forage environment class.

```python
from puffergrid.grid_env cimport GridEnv

cdef class Forage(GridEnv):
    def __init__(self, int map_width=100, int map_height=100,
                 int num_agents=20, int num_walls=10, int num_trees=10):
        GridEnv.__init__(
            self,
            map_width,
            map_height,
            0,  # max_timestep
            [ObjectType.AgentT, ObjectType.WallT, ObjectType.TreeT],
            11, 11,  # observation shape
            ObsEncoder(),
            [Move(), Rotate(), Eat()],
            [ResetTreeHandler()]
        )

        # Initialize agents, walls, and trees
        # Implementation details...
```

### Step 6: Using the Environment

Now that we've defined our environment, we can use it in a reinforcement learning loop:

```python
from puffergrid.wrappers.grid_env_wrapper import PufferGridEnv

# Create the Forage environment
c_env = Forage(map_width=100, map_height=100, num_agents=20, num_walls=10, num_trees=10)

# Wrap the environment with PufferGridEnv
env = PufferGridEnv(c_env, num_agents=20, max_timesteps=1000)

# Reset the environment
obs, _ = env.reset()

# Run a simple loop
for _ in range(1000):
    actions = env.action_space.sample()  # Random actions
    obs, rewards, terminals, truncations, infos = env.step(actions)

    if terminals.any() or truncations.any():
        break

# Print final stats
print(env.unwrapped.stats())
```

This example demonstrates the core components of creating a PufferGrid environment: defining objects, actions, events, observations, and putting them together in an environment class.

## Performance Testing

To run performance tests on your PufferGrid environment, use the `test_perf.py` script:

```
python test_perf.py --env examples.forage.Forage --num_agents 20 --duration 20
```

You can also run the script with profiling enabled:

```
python test_perf.py --env examples.forage.Forage --num_agents 20 --duration 20 --profile
```

## Contributing

Contributions to PufferGrid are welcome! Please feel free to submit pull requests, create issues, or suggest improvements.
