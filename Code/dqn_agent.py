import torch
import torch.nn as nn
import os

class DQNNetwork(nn.Module):
    """
    Simple MLP for Q-value prediction.
    Adjust architecture based on the actual .pth weights.
    """
    def __init__(self, input_dim, output_dim):
        super(DQNNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class HDQNAgent:
    """
    Hierarchical Deep Q-Network (HDQN) Inference Agent.
    Meta-Controller selects the goal, Controller selects the action.
    """
    def __init__(self, meta_weights_path=None, controller_weights_path=None):
        self.meta_weights_path = meta_weights_path
        self.controller_weights_path = controller_weights_path

        # state_dim = 7 (6 from BFE behavioral features + 1 from Learning Objectives Vector g_t)
        # meta_output_dim = 9 (Meta-strategies)
        # controller_output_dim = 3 (Sub-strategies per meta-strategy)

        self.meta_controller = self._load_model(meta_weights_path, 7, 9)
        self.controller = self._load_model(controller_weights_path, 7 + 1, 3) # state + goal

        # Target Networks
        self.meta_target = DQNNetwork(7, 9)
        self.controller_target = DQNNetwork(7 + 1, 3)
        self.meta_target.load_state_dict(self.meta_controller.state_dict())
        self.controller_target.load_state_dict(self.controller.state_dict())
        self.meta_target.eval()
        self.controller_target.eval()

    def _load_model(self, path, input_dim, output_dim):
        model = DQNNetwork(input_dim, output_dim)
        if path and os.path.exists(path):
            try:
                model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
            except Exception as e:
                print(f"Warning: Could not load weights from {path}: {e}. Using random initialization.")
        else:
            print(f"Weights not found at {path}. Using random initialization.")

        model.eval()
        return model

    def get_optimal_action(self, state_vector):
        """
        Performs the hierarchical forward pass.
        """
        with torch.no_grad():
            # 1. Meta-Controller: Predict the optimal goal
            goal_q_values = self.meta_controller(state_vector)
            goal = torch.argmax(goal_q_values).item()

            # 2. Controller: Predict the optimal action given state and goal
            # We concatenate the goal ID as an additional input feature
            goal_tensor = torch.FloatTensor([goal])
            augmented_state = torch.cat([state_vector, goal_tensor])

            action_q_values = self.controller(augmented_state)
            action = torch.argmax(action_q_values).item()

            return action
