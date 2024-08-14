#ifndef GRID_HPP
#define GRID_HPP

#include "grid_object.hpp"
#include <vector>
#include <algorithm>

using namespace std;
typedef vector<vector<vector<GridObjectId> > > GridType;

class Grid {
    public:
        unsigned int width;
        unsigned int height;
        vector<Layer> layer_for_type_id;
        Layer num_layers;

        GridType grid;
        vector<GridObject*> objects;

        inline Grid(unsigned int width, unsigned int height, vector<Layer> layer_for_type_id)
            : width(width), height(height), layer_for_type_id(layer_for_type_id) {

                num_layers = *max_element(layer_for_type_id.begin(), layer_for_type_id.end()) + 1;
                grid.resize(height, vector<vector<GridObjectId> >(
                    width, vector<GridObjectId>(this->num_layers, 0)));

                // 0 is reserved for empty space
                objects.push_back(nullptr);
        }

        inline char add_object(GridObject * obj) {
            if (obj->location.r >= height or obj->location.c >= width or obj->location.layer >= num_layers) {
                return false;
            }
            if (this->grid[obj->location.r][obj->location.c][obj->location.layer] != 0) {
                return false;
            }

            obj->id = this->objects.size();
            this->objects.push_back(obj);
            this->grid[obj->location.r][obj->location.c][obj->location.layer] = obj->id;
            return true;
        }


        inline char move_object(GridObjectId id, const GridLocation &loc) {
            if (loc.r >= height or loc.c >= width or loc.layer >= num_layers) {
                return false;
            }

            if (grid[loc.r][loc.c][loc.layer] != 0) {
                return false;
            }

            GridObject* obj = objects[id];
            grid[loc.r][loc.c][loc.layer] = id;
            grid[obj->location.r][obj->location.c][obj->location.layer] = 0;
            obj->location = loc;
            return true;
        }

        inline GridObject* object(GridObjectId obj_id) {
            return objects[obj_id];
        }

        inline GridObject* object_at(const GridLocation &loc) {
            if (loc.r >= height or loc.c >= width or loc.layer >= num_layers) {
                return nullptr;
            }
            if (grid[loc.r][loc.c][loc.layer] == 0) {
                return nullptr;
            }
            return objects[grid[loc.r][loc.c][loc.layer]];
        }

        inline GridObject* object_at(const GridLocation &loc, TypeId type_id) {
            GridObject *obj = object_at(loc);
            if (obj != NULL and obj->_type_id == type_id) {
                return obj;
            }
            return nullptr;
        }

        inline GridObject* object_at(GridCoord r, GridCoord c, TypeId type_id) {
            GridObject *obj = object_at(GridLocation(r, c), this->layer_for_type_id[type_id]);
            if (obj->_type_id != type_id) {
                return nullptr;
            }

            return obj;
        }

        inline const GridLocation location(GridObjectId id) {
            return objects[id]->location;
        }

        inline const GridLocation relative_location(const GridLocation &loc, Orientation orientation) {
            GridLocation new_loc = loc;
            switch (orientation) {
                case Up:
                    if (loc.r > 0) new_loc.r = loc.r - 1;
                    break;
                case Down:
                    new_loc.r = loc.r + 1;
                    break;
                case Left:
                    if (loc.c > 0) new_loc.c = loc.c - 1;
                    break;
                case Right:
                    new_loc.c = loc.c + 1;
                    break;
            }
            return new_loc;
        }

        inline const GridLocation relative_location(const GridLocation &loc, Orientation orientation, TypeId type_id) {
            GridLocation rloc = this->relative_location(loc, orientation);
            rloc.layer = this->layer_for_type_id[type_id];
            return rloc;
        }

        inline char is_empty(unsigned int row, unsigned int col) {
            GridLocation loc;
            loc.r = row; loc.c = col;
            for (int layer = 0; layer < num_layers; ++layer) {
                loc.layer = layer;
                if (object_at(loc) != nullptr) {
                    return 0;
                }
            }
            return 1;
        }
};

#endif // GRID_HPP
