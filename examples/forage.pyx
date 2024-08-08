import numpy as np

from libcpp.vector cimport vector
from libcpp.string cimport string

from puffergrid.grid_object cimport GridObjectBase, GridObject, GridLocation, GridObjectId, Orientation
from puffergrid.grid_env cimport GridEnv
from puffergrid.action cimport ActionHandler, ActionArg
from puffergrid.observation_encoder cimport ObservationEncoder
from puffergrid.event cimport EventHandler, EventArg


################################################
# Define Game Objects
################################################

cdef cppclass AgentProps:
    unsigned int energy
    unsigned int orientation

    inline void obs(int[:] obs):
        obs[0] = 1
        obs[1] = energy
        obs[2] = orientation

    @staticmethod
    inline vector[string] feature_names():
        return ["wall", "wall:hp"]

ctypedef GridObject[AgentProps] Agent

cdef cppclass WallProps:
    unsigned int hp

    inline void obs(int[:] obs):
        obs[0] = 1
        obs[1] = hp

    @staticmethod
    inline vector[string] feature_names():
        return ["wall", "wall:hp"]

ctypedef GridObject[WallProps] Wall

cdef cppclass TreeProps:
    char has_fruit

    inline void obs(int[:] obs):
        obs[0] = 1
        obs[1] = has_fruit

    @staticmethod
    inline vector[string] feature_names():
        return ["tree", "tree:has_fruit"]

ctypedef GridObject[TreeProps] Tree

cdef enum ObjectType:
    AgentT = 0
    WallT = 1
    TreeT = 2

################################################
# Define Actions
################################################

cdef class Move(ActionHandler):
    cdef char handle_action(
        self,
        unsigned int actor_id,
        GridObjectId actor_object_id,
        ActionArg arg):

        cdef unsigned short direction = arg
        if direction >= 2:
            return False

        cdef Agent* agent = self.env._grid.object[Agent](actor_object_id)
        cdef Orientation orientation = <Orientation>((agent.props.orientation + 2*(direction)) % 4)
        cdef GridLocation old_loc = agent.location
        cdef GridLocation new_loc = self.env._grid.relative_location(old_loc, orientation)
        if not self.env._grid.is_empty(new_loc.r, new_loc.c):
            return False
        cdef char s = self.env._grid.move_object(actor_object_id, new_loc)
        if s:
            self.env._stats.agent_incr(actor_id, "action.move")

        return s

cdef class Rotate(ActionHandler):
    cdef char handle_action(
        self,
        unsigned int actor_id,
        GridObjectId actor_object_id,
        ActionArg arg):

        cdef unsigned short orientation = arg
        if orientation >= 4:
            return False

        cdef Agent* agent = self.env._grid.object[Agent](actor_object_id)
        agent.props.orientation = orientation
        self.env._stats.agent_incr(actor_id, "action.rotate")
        return True

cdef class Eat(ActionHandler):
    cdef char handle_action(
        self,
        unsigned int actor_id,
        GridObjectId actor_object_id,
        ActionArg arg):
        cdef Tree *tree = NULL
        cdef Agent* agent = self.env._grid.object[Agent](actor_object_id)
        cdef GridLocation target_loc = self.env._grid.relative_location(
            agent.location,
            <Orientation>agent.props.orientation
        )
        tree = self.env._grid.object_at[Tree](
            self.env._grid.type_location(target_loc.r, target_loc.c, ObjectType.TreeT))
        if tree == NULL or tree.props.has_fruit == 0:
            return False

        tree.props.has_fruit = 0
        agent.props.energy += 10
        self.env._rewards[actor_id] += 10
        self.env._stats.agent_incr(actor_id, "action.eat")
        self.env._event_manager.schedule_event(Events.ResetTree, 100, tree.id, 0)
        return True

################################################
# Define Event Handlers
################################################

cdef class ResetTreeHandler(EventHandler):
    cdef void handle_event(self, GridObjectId obj_id, EventArg arg):
        self.env._grid.object[Tree](obj_id).props.has_fruit = True
        self.env._stats.game_incr("fruit_spawned")

cdef enum Events:
    ResetTree = 0

################################################
# Define Observation Encoder
################################################
cdef class ObsEncoder(ObservationEncoder):
    cdef vector[string] _feature_names

    def __init__(self):
        ObservationEncoder.__init__(self)
        f = []
        f.extend(AgentProps.feature_names())
        f.extend(WallProps.feature_names())
        f.extend(TreeProps.feature_names())
        self._feature_names = f

    cdef encode(self, GridObjectBase *obj, int[:] obs):
        cdef Agent *agent
        cdef Wall *wall
        cdef Tree *tree

        if obj._type_id == ObjectType.AgentT:
            (<Agent*>obj).props.obs(obs[0:])
        elif obj._type_id == ObjectType.WallT:
            (<Wall*>obj).props.obs(obs[3:])
        elif obj._type_id == ObjectType.TreeT:
            (<Tree*>obj).props.obs(obs[5:])

    cdef vector[string] feature_names(self):
        return self._feature_names


################################################
# Define The Environment
################################################

cdef class Forage(GridEnv):
    def __init__(
        self, int map_width=100, int map_height=100,
        int num_agents = 20, int num_walls = 10, int num_trees = 10):

        GridEnv.__init__(
            self,
            num_agents,
            map_width,
            map_height,
            0, # max_timestep
            [
                ObjectType.AgentT,
                ObjectType.WallT,
                ObjectType.TreeT
            ],
            11, 11, # observation shape
            ObsEncoder(),
            [
                Move(),
                Rotate(),
                Eat()
            ],
            [
                ResetTreeHandler()
            ]
        )

        # Randomly place agents, walls, and trees
        coords = [(r, c) for r in range(map_height) for c in range(map_width)]
        np.random.shuffle(coords)

        cdef Agent *agent
        cdef Wall *wall
        cdef Tree *tree

        for (row, col) in coords[:num_agents]:
            agent = self._grid.create_object[Agent](ObjectType.AgentT, row, col)
            agent.props.energy = 100
            agent.props.orientation = 0
            self.add_agent(agent)
        coords = coords[num_agents:]

        for (row, col) in coords[:num_walls]:
            wall = self._grid.create_object[Wall](ObjectType.WallT, row, col)
            wall.props.hp = 100
        coords = coords[num_walls:]

        for (row, col) in coords[:num_trees]:
            tree = self._grid.create_object[Tree](ObjectType.TreeT, row, col)
            tree.props.has_fruit = 1

        print(f"Forage environment created {map_width}x{map_height} with {num_agents} agents, {num_walls} walls, and {num_trees} trees")

