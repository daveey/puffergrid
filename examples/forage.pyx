import numpy as np
from libc.stdio cimport printf

from libcpp.vector cimport vector
from libcpp.string cimport string

from puffergrid.grid_object cimport GridObject, GridCoord, GridLocation, GridObjectId, Orientation, TypeId
from puffergrid.grid_env cimport GridEnv
from puffergrid.action cimport ActionHandler, ActionArg
from puffergrid.observation_encoder cimport ObservationEncoder
from puffergrid.event cimport EventHandler, EventArg
import sys

################################################
# Define Game Objects
################################################
cdef enum ObjectType:
    AgentT = 0
    WallT = 1
    TreeT = 2

cdef cppclass Agent(GridObject):
    unsigned int energy
    unsigned int orientation

    Agent(GridCoord r, GridCoord c):
        init(ObjectType.AgentT, GridLocation(r, c))
        energy = 100
        orientation = 0

    inline void obs(int[:] obs):
        obs[0] = 1
        obs[1] = energy
        obs[2] = orientation

    @staticmethod
    inline vector[string] feature_names():
        return ["agent", "agent:energy", "agent:orientation"]

cdef cppclass Wall(GridObject):
    unsigned int hp

    Wall(GridCoord r, GridCoord c):
        init(ObjectType.WallT, GridLocation(r, c))
        hp = 100

    inline void obs(int[:] obs):
        obs[0] = 1
        obs[1] = hp

    @staticmethod
    inline vector[string] feature_names():
        return ["wall", "wall:hp"]


cdef cppclass Tree(GridObject):
    char has_fruit

    Tree(GridCoord r, GridCoord c):
        init(ObjectType.TreeT, GridLocation(r, c))
        this.has_fruit = 1

    inline void obs(int[:] obs):
        obs[0] = 1
        obs[1] = has_fruit

    @staticmethod
    inline vector[string] feature_names():
        return ["tree", "tree:has_fruit"]


################################################
# Define Observation Encoder
################################################
cdef class ObsEncoder(ObservationEncoder):
    cdef vector[string] _feature_names

    def __init__(self):
        ObservationEncoder.__init__(self)
        f = []
        f.extend(Agent.feature_names())
        f.extend(Wall.feature_names())
        f.extend(Tree.feature_names())
        self._feature_names = f

    cdef encode(self, GridObject *obj, int[:] obs):
        cdef Agent *agent
        cdef Wall *wall
        cdef Tree *tree

        if obj._type_id == ObjectType.AgentT:
            (<Agent*>obj).obs(obs[0:3])
        elif obj._type_id == ObjectType.WallT:
            (<Wall*>obj).obs(obs[3:5])
        elif obj._type_id == ObjectType.TreeT:
            (<Tree*>obj).obs(obs[5:7])

    cdef vector[string] feature_names(self):
        return self._feature_names


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

        cdef Agent* agent = <Agent*>self.env._grid.object(actor_object_id)
        cdef Orientation orientation = <Orientation>((agent.orientation + 2*(direction)) % 4)
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

        cdef Agent* agent = <Agent*>self.env._grid.object(actor_object_id)
        agent.orientation = orientation
        self.env._stats.agent_incr(actor_id, "action.rotate")
        return True

cdef class Eat(ActionHandler):
    cdef char handle_action(
        self,
        unsigned int actor_id,
        GridObjectId actor_object_id,
        ActionArg arg):
        cdef Tree *tree = NULL
        cdef Agent* agent = <Agent*>self.env._grid.object(actor_object_id)
        cdef GridLocation target_loc = self.env._grid.relative_location(
            agent.location,
            <Orientation>agent.orientation,
            ObjectType.TreeT
        )

        tree = <Tree*>self.env._grid.object_at(target_loc, ObjectType.TreeT)

        if tree == NULL or tree.has_fruit == 0:
            return False

        tree.has_fruit = 0
        agent.energy += 10
        self.env._rewards[actor_id] += 10
        self.env._stats.agent_incr(actor_id, "action.eat")
        self.env._event_manager.schedule_event(Events.ResetTree, 100, tree.id, 0)
        return True

################################################
# Define Event Handlers
################################################

cdef class ResetTreeHandler(EventHandler):
    cdef void handle_event(self, GridObjectId obj_id, EventArg arg):
        (<Tree*>self.env._grid.object(obj_id)).has_fruit = True
        self.env._stats.game_incr("fruit_spawned")

cdef enum Events:
    ResetTree = 0


ObjectLayers = {
    ObjectType.AgentT: 0,
    ObjectType.WallT: 0,
    ObjectType.TreeT: 0
}

################################################
# Define The Environment
################################################

cdef class Forage(GridEnv):
    def __init__(
        self, int map_width=50, int map_height=50,
        int num_agents = 2, int num_walls = 20, int num_trees = 20):

        GridEnv.__init__(
            self,
            num_agents,
            map_width,
            map_height,
            0, # max_timestep
            ObjectLayers.values(),
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
            agent = new Agent(row, col)
            self._grid.add_object(agent)
            self.add_agent(agent)
        coords = coords[num_agents:]

        for (row, col) in coords[:num_walls]:
            wall = new Wall(row, col)
            self._grid.add_object(wall)
        coords = coords[num_walls:]

        for (row, col) in coords[:num_trees]:
            tree = new Tree(row, col)
            r = self._grid.add_object(tree)

        print(f"Forage environment created {map_width}x{map_height} with {num_agents} agents, {num_walls} walls, and {num_trees} trees")


    def render(this):
        grid = this.render_ascii(["A", "#", "&"])
        sys.stdout.write("\033[H\033[J")
        for row in grid:
            sys.stdout.write(" ".join(map(str, row)) + "\n")
            sys.stdout.flush()

