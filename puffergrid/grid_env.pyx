from libc.stdio cimport printf

cimport numpy as cnp
import numpy as np
from puffergrid.action cimport ActionArg, ActionHandler
from puffergrid.grid_object cimport Layer, GridLocation
from puffergrid.observation_encoder cimport ObservationEncoder, ObsType
from puffergrid.grid_object cimport GridObject, GridObjectId
from puffergrid.event cimport EventManager, EventHandler
from puffergrid.grid cimport Grid
from libcpp.vector cimport vector
from puffergrid.stats_tracker cimport StatsTracker
import gymnasium as gym

obs_np_type = np.uint8
cdef class GridEnv:
    def __init__(
            self,
            unsigned int max_agents,
            unsigned int map_width,
            unsigned int map_height,
            unsigned int max_timestep,
            vector[Layer] layer_for_type_id,
            unsigned short obs_width,
            unsigned short obs_height,
            ObservationEncoder observation_encoder,
            list[ActionHandler] action_handlers,
            list[EventHandler] event_handlers
        ):
        self._obs_width = obs_width
        self._obs_height = obs_height
        self._max_timestep = max_timestep
        self._current_timestep = 0

        self._grid = new Grid(map_width, map_height, layer_for_type_id)
        self._obs_encoder = observation_encoder

        self._action_handlers = action_handlers
        for handler in action_handlers:
            (<ActionHandler>handler).init(self)

        self._event_manager = EventManager(self, event_handlers)
        self._stats = StatsTracker(max_agents)

        self.set_buffers(
            np.zeros(
                (max_agents, len(self._obs_encoder.feature_names()),
                self._obs_height, self._obs_width),
                dtype=obs_np_type),
            np.zeros(max_agents, dtype=np.int8),
            np.zeros(max_agents, dtype=np.int8),
            np.zeros(max_agents, dtype=np.float32)
        )

    cdef void add_agent(self, GridObject* agent):
        self._agents.push_back(agent)

    cdef void _compute_observation(
        self,
        unsigned observer_r, unsigned int observer_c,
        unsigned short obs_width, unsigned short obs_height,
        ObsType[:,:,:] observation):

        cdef:
            int r, c, layer
            GridLocation object_loc
            GridObject *obj
            unsigned short obs_width_r = obs_width >> 1
            unsigned short obs_height_r = obs_height >> 1
            cdef unsigned int obs_r, obs_c
            cdef ObsType[:] agent_ob

        cdef unsigned int r_start = max(observer_r, obs_height_r) - obs_height_r
        cdef unsigned int c_start = max(observer_c, obs_width_r) - obs_width_r
        for r in range(r_start, observer_r + obs_height_r + 1):
            if r < 0 or r >= self._grid.height:
                continue
            for c in range(c_start, observer_c + obs_width_r + 1):
                if c < 0 or c >= self._grid.width:
                    continue
                for layer in range(self._grid.num_layers):
                    object_loc = GridLocation(r, c, layer)
                    obj = self._grid.object_at(object_loc)
                    if obj == NULL:
                        continue

                    obs_r = object_loc.r + obs_height_r - observer_r
                    obs_c = object_loc.c + obs_width_r - observer_c
                    agent_ob = observation[:, obs_r, obs_c]
                    self._obs_encoder.encode(obj, agent_ob)

    cdef void _compute_observations(self):
        cdef GridObject *agent
        for idx in range(self._agents.size()):
            agent = self._agents[idx]
            self._compute_observation(
                agent.location.r, agent.location.c,
                self._obs_width, self._obs_height, self._observations[idx])

    cdef void _step(self, int[:,:] actions):
        cdef:
            unsigned int idx
            short action
            ActionArg arg
            GridObject *agent
            ActionHandler handler

        self._rewards[:] = 0
        self._observations[:, :, :, :] = 0

        self._current_timestep += 1
        self._event_manager.process_events(self._current_timestep)

        for idx in range(self._agents.size()):
            action = actions[idx][0]
            arg = actions[idx][1]
            agent = self._agents[idx]
            handler = <ActionHandler>self._action_handlers[action]
            handler.handle_action(idx, agent.id, arg)
        self._compute_observations()

        if self._max_timestep > 0 and self._current_timestep >= self._max_timestep:
            self._truncations[:] = 1

    ###############################
    # Python API
    ###############################
    cpdef tuple[cnp.ndarray, dict] reset(self):
        if self._current_timestep > 0:
            raise NotImplemented("Cannot reset after stepping")

        self._terminals[:] = 0
        self._truncations[:] = 0
        self._compute_observations()
        return (self._observations_np, {})

    cpdef tuple[cnp.ndarray, cnp.ndarray, cnp.ndarray, cnp.ndarray, dict] step(self, int[:,:] actions):
        self._step(actions)
        return (self._observations_np, self._rewards_np, self._terminals_np, self._truncations_np, {})

    cpdef void set_buffers(
        self,
        cnp.ndarray[ObsType, ndim=4] observations,
        cnp.ndarray[char, ndim=1] terminals,
        cnp.ndarray[char, ndim=1] truncations,
        cnp.ndarray[float, ndim=1] rewards):

        self._observations_np = observations
        self._observations = observations
        self._terminals_np = terminals
        self._terminals = terminals
        self._truncations_np = truncations
        self._truncations = truncations
        self._rewards_np = rewards
        self._rewards = rewards

    cpdef grid(self):
        return []

    cpdef unsigned int num_actions(self):
        return len(self._action_handlers)

    cpdef unsigned int current_timestep(self):
        return self._current_timestep

    cpdef unsigned int map_width(self):
        return self._grid.width

    cpdef unsigned int map_height(self):
        return self._grid.height

    cpdef list[str] grid_features(self):
        return self._obs_encoder.feature_names()

    cpdef unsigned int num_agents(self):
        return self._agents.size()

    cpdef tuple observation_shape(self):
        return (len(self._obs_encoder.feature_names()), self.obs_height, self.obs_width)

    cpdef observe(
        self,
        GridObjectId observer_id,
        unsigned short obs_width,
        unsigned short obs_height,
        ObsType[:,:,:] observation):

        cdef GridObject* observer = self._grid.object(observer_id)
        self._compute_observation(
            observer.location.r, observer.location.c, obs_width, obs_height, observation)

    cpdef observe_at(
        self,
        unsigned short row,
        unsigned short col,
        unsigned short obs_width,
        unsigned short obs_height,
        ObsType[:,:,:] observation):

        self._compute_observation(
            row, col, obs_width, obs_height, observation)

    cpdef get_episode_stats(self):
        return self._stats.to_pydict()

    cpdef tuple get_buffers(self):
        return (self._observations_np, self._terminals_np, self._truncations_np, self._rewards_np)

    cpdef cnp.ndarray render_ascii(self, list[char] type_to_char):
        cdef GridObject *obj
        grid = np.full((self._grid.height, self._grid.width), " ", dtype=np.str_)
        for obj_id in range(1, self._grid.objects.size()):
            obj = self._grid.object(obj_id)
            grid[obj.location.r, obj.location.c] = type_to_char[obj._type_id]
        return grid

    cpdef cnp.ndarray grid_objects(self):
        cdef GridObject *obj
        grid = np.zeros((self._grid.height, self._grid.width), dtype=np.uint8)
        for obj_id in range(1, self._grid.objects.size()):
            obj = self._grid.object(obj_id)
            grid[obj.location.r, obj.location.c] = obj._type_id + 1
        return grid

    @property
    def action_space(self):
        return gym.spaces.MultiDiscrete((self.num_actions(), 255), dtype=np.uint32)

    @property
    def observation_space(self):
        type_info = np.iinfo(obs_np_type)
        return gym.spaces.Box(
            low=type_info.min, high=type_info.max,
            shape=(
                len(self._obs_encoder.feature_names()),
                self._obs_height, self._obs_width),
            dtype=obs_np_type
        )


