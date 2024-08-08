import test_grid_object
import test_action_handler
import test_event_handler
import test_observation_encoder
import test_grid_env

if __name__ == "__main__":
    test_grid_object.run_tests()
    test_action_handler.run_tests()
    test_event_handler.run_tests()
    test_observation_encoder.run_tests()
    test_grid_env.run_tests()
    print("All tests passed!")
