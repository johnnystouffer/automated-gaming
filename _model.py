# ALL IMPORTS
import torch
import random
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class DeepQLearning(nn.Module):
    # Our neural network 
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__() # Inheriting from nn.Module
        
        # Defining the layers and ReLU activation function
        self.fc1 = nn.Linear(input_size,hidden_size) 
        self.fc2 = nn.Linear(hidden_size,output_size)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        
        # Passing layer one through our activation function
        x = self.relu(self.linear1(x))
        # Passing layer two and returning the output
        x = self.linear2(x)
        return x
    
    def save(self, file_name='snake.pth'):
        
        # Creating a new folder to save our model
        model_folder_path = './model'
        
        # Unless I already have a folder
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
            
        file_name = os.path.join(model_folder_path,file_name)
        torch.save(self.state_dict(),file_name)
        
class QTrainer:
    def __init__(self, model, lr, gamma):
        
        # Learning Rate
        self.lr = lr
        # Discount Rate
        self.gamma = gamma 
        # Model / Network
        self.model = model
        # Optimizer
        self.optimizer = optim.Adam(model.parameters(),lr=self.lr)
        # Loss function
        self.criterion = nn.MSELoss()
        
    def train_step(self, state, action, reward, next_state, done):
        
        # Convert to tensors
        state = torch.tensor(state,dtype=torch.float)
        action = torch.tensor(action,dtype=torch.long)
        reward = torch.tensor(reward,dtype=torch.float)
        next_state = torch.tensor(next_state,dtype=torch.float)
        
        # Checking the length
        if len(state.shape) == 1:
            
            # Unsqueeze to make it a 2D tensor (1, x)
            state = torch.unsqueeze(state,0)
            next_state = torch.unsqueeze(next_state,0)
            reward = torch.unsqueeze(reward,0)
            action = torch.unsqueeze(action,0)
            # Making Done a Tuple
            done = (done, )
            
        # predicted Q values with the state we have
        prediction = self.model(state)
        
        target = prediction.clone()
        for idx in range(len(action)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
                
            target[idx][torch.argmax(action[idx]).item()] = Q_new
        
        # Zero the gradients
        self.optimizer.zero_grad()
        loss = self.criterion(target,prediction)
        loss.backward()
        
        