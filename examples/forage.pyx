
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
        tree = self.env._grid.object_at[Tree](target_loc.r, target_loc.c, ObjectType.TreeT)
        if tree == NULL or tree.props.has_fruit == 0:
            return False

        tree.props.has_fruit = 0
        agent.props.energy += 10
        self.env._rewards[actor_id] += 10
        self.env._stats.agent_incr(actor_id, "fruit_eaten")
        self.env._event_manager.schedule_event(Events.ResetTree, 100, tree.id, 0)
        return True

################################################
# Define Event Handlers
################################################

cdef class ResetTreeHandler(EventHandler):
    cdef void handle_event(self, GridObjectId obj_id, EventArg arg):
        self.env._grid.object[Tree](obj_id).props.has_fruit = True

cdef enum Events:
    ResetTree = 0

################################################
# Define Observation Encoder
################################################
cdef class ObsEncoder(ObservationEncoder):
    cdef encode(self, GridObjectBase *obj, int[:] obs):
        cdef Agent *agent
        cdef Wall *wall
        cdef Tree *tree

        if obj._type_id == ObjectType.AgentT:
            agent = <Agent *>obj
            obs[0] = 1
            obs[1] = agent.props.energy
            obs[2] = agent.props.orientation
        elif obj._type_id == ObjectType.WallT:
            obs[3] = 1
        elif obj._type_id == ObjectType.TreeT:
            tree = <Tree *>obj
            obs[4] = 1
            obs[5] = tree.props.has_fruit

    cdef vector[string] feature_names(self):
        return [
            "agent", "agent:energy", "agent:orientation",
            "wall", "tree", "tree:has_fruit"]


################################################
# Define The Environment
################################################

cdef class Forage(GridEnv):
    def __init__(self, int map_width, int map_height):

        GridEnv.__init__(
            self,
            map_width,
            map_height,
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
