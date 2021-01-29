import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    """
    Main modell class. First 4 layers are convolutional layers, after that the model is split into the
    advantage and value stream. See the documentation. The convolutional layers are initialized with Kaiming He initialization.
    """
    def __init__(self, n_actions, hidden=128):
        """
        Args:
            n_actions: Integer, amount of possible actions of the specific environment
            hidden: Integer, amount of hidden layers (To Do, hidden can change but split_size won't fit anymore)
        """
        super(DQN, self).__init__()
        
        self.n_actions = n_actions
        self.hidden = hidden
        # Output of the 4th conv layer is 20480, if hidden is 128

        # Conv layers
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4, bias=False)
        nn.init.kaiming_uniform_(self.conv1.weight, nonlinearity='relu')
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, bias=False)
        nn.init.kaiming_uniform_(self.conv2.weight, nonlinearity='relu')
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, bias=False)
        nn.init.kaiming_uniform_(self.conv3.weight, nonlinearity='relu')
        self.conv4 = nn.Conv2d(64, self.hidden, kernel_size=7, stride=1, bias=False)
        nn.init.kaiming_uniform_(self.conv4.weight, nonlinearity='relu')


        out = self.conv4(self.conv3(self.conv2(self.conv1(torch.zeros(1,4,210,160)))))
        out = out.view(out.size(0), -1)
        self.split_size = int(out.size(1)/2)


        #Advantage and Value layer output
        self.advantage_l = torch.nn.Linear(self.split_size, self.n_actions)
        nn.init.kaiming_uniform_(self.advantage_l.weight, nonlinearity='relu')
        self.advantage_l.bias.data.zero_()

        self.value_l = torch.nn.Linear(self.split_size, 1)
        nn.init.kaiming_uniform_(self.value_l.weight, nonlinearity='relu')
        self.value_l.bias.data.zero_()



    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)

        self.valuestream, self.advantagestream = torch.split(x, self.split_size, dim=1)
        
        self.advantage = self.advantage_l(self.advantagestream)
        self.value = self.value_l(self.valuestream)

        return self.advantage, self.value

    def predict_q(self, x):
        advantage, V_of_s = self.forward(x)

        self.q_values = V_of_s + (advantage - advantage.mean(dim=1, keepdim=True))
        return self.q_values


    def predict_action(self, x):
        q_values = self.predict_q(x)
        self.best_action = torch.argmax(q_values, 1)
        return self.best_action
        
class ReplayMemory(object):
    """Replay Memory that stores the last size=1,000,000 transitions"""
    def __init__(self, size=1000000, frame_height=210, frame_width=160, 
                 agent_history_length=4, batch_size=32):
        """
        Args:
            size: Integer, Number of stored transitions
            frame_height: Integer, Height of a frame of an Atari game
            frame_width: Integer, Width of a frame of an Atari game
            agent_history_length: Integer, Number of frames stacked together to create a state
            batch_size: Integer, Number if transitions returned in a minibatch
        """
        self.size = size
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.agent_history_length = agent_history_length
        self.batch_size = batch_size
        self.count = 0
        self.current = 0
        
        # Pre-allocate memory
        self.actions = np.empty(self.size, dtype=np.int32)
        self.rewards = np.empty(self.size, dtype=np.float32)
        self.frames = np.empty((self.size, self.frame_height, self.frame_width), dtype=np.uint8)
        self.terminal_flags = np.empty(self.size, dtype=np.bool)
        
        # Pre-allocate memory for the states and new_states in a minibatch
        self.states = np.empty((self.batch_size, self.agent_history_length, 
                                self.frame_height, self.frame_width), dtype=np.uint8)
        self.new_states = np.empty((self.batch_size, self.agent_history_length, 
                                    self.frame_height, self.frame_width), dtype=np.uint8)
        self.indices = np.empty(self.batch_size, dtype=np.int32)
        
    def add_experience(self, action, frame, reward, terminal):
        """
        Args:
            action: An integer between 0 and env.action_space.n - 1 
                determining the action the agent perfomed
            frame: A (84, 84, 1) frame of an Atari game in grayscale
            reward: A float determining the reward the agend received for performing an action
            terminal: A bool stating whether the episode terminated
        """
        if frame.shape != (self.frame_height, self.frame_width):
            raise ValueError('Dimension of frame is wrong!')
        self.actions[self.current] = action
        self.frames[self.current, ...] = frame
        self.rewards[self.current] = reward
        self.terminal_flags[self.current] = terminal
        self.count = max(self.count, self.current+1)
        self.current = (self.current + 1) % self.size
             
    def _get_state(self, index):
        if self.count == 0:
            raise ValueError("The replay memory is empty!")
        if index < self.agent_history_length - 1:
            raise ValueError("Index must be min 3")
        return self.frames[index-self.agent_history_length+1:index+1, ...]
        
    def _get_valid_indices(self):
        for i in range(self.batch_size):
            while True:
                index = random.randint(self.agent_history_length, self.count - 1)
                if index < self.agent_history_length:
                    continue
                if index >= self.current and index - self.agent_history_length <= self.current:
                    continue
                if self.terminal_flags[index - self.agent_history_length:index].any():
                    continue
                break
            self.indices[i] = index
            
    def get_minibatch(self):
        """
        Returns a minibatch of self.batch_size = 32 transitions
        """
        if self.count < self.agent_history_length:
            raise ValueError('Not enough memories to get a minibatch')
        
        self._get_valid_indices()
            
        for i, idx in enumerate(self.indices):
            self.states[i] = self._get_state(idx - 1)
            self.new_states[i] = self._get_state(idx)
        
        return np.transpose(self.states, axes=(0, 1, 2, 3)), self.actions[self.indices], \
        self.rewards[self.indices], np.transpose(self.new_states, axes=(0, 1, 2, 3)), \
        self.terminal_flags[self.indices]