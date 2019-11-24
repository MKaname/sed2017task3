import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import processing

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data, label, transform = None):
        self.transform = transform
        self.data = []
        self.label = []
        for i in range(len(data)):
            self.data.append(data[i])
            self.label.append(label[i])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        out_data = self.data[idx]
        out_label = self.label[idx]
        if self.transform:
            out_data= self.transform(out_data)
        return out_data, out_label

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1,128,3,padding = 1)
        self.bn1 = nn.BatchNorm2d(128)
        self.pool1= nn.MaxPool2d((1,5))
        self.dropout = nn.Dropout()
        self.conv2 = nn.Conv2d(128,128,3,padding = 1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d((1,2))
        self.conv3 = nn.Conv2d(128,128,3,padding = 1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d((1,2))
        self.gru = nn.GRU(256, 32, num_layers = 2, dropout = 0.5, batch_first = True, bidirectional = True)
        self.linear1 = nn.Linear(64, 16)
        self.linear2 = nn.Linear(16, 6)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = self.pool1(x)
        x = self.dropout(x)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = self.pool2(x)
        x = self.dropout(x)
        x = F.relu(self.conv3(x))
        x = self.bn3(x)
        x = self.pool3(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)#(N,T,128,2)
        x = x.reshape(x.shape[0],x.shape[1],-1)#(N,T,256)
        x , _= self.gru(x)
        x = self.linear1(x)
        x = self.dropout(x)
        x = torch.sigmoid(self.linear2(x))
        return x

if __name__ == "__main__":
    mbe_dir = "mbe/street"
    label_dir = "label/street"
    subdivs = 256
    epochs = 100
    device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
    print(device)
    transform = transforms.Compose([transforms.ToTensor()])
    batch_size = 4
    X, Y = processing.make_validation_data(mbe_dir, label_dir, 40, 6)
    X = processing.split_in_seqs(X, subdivs)
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    Y = processing.split_in_seqs(Y, subdivs)
    dataset = MyDataset(X, Y, transform = transform)

    n_samples = len(dataset)
    train_size = int(len(dataset) * 0.8)
    val_size = n_samples - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle=True)
    valloader = torch.utils.data.DataLoader(val_dataset, batch_size = batch_size, shuffle = False)

    net = Net()
    net = net.float()
    net.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(net.parameters())

    train_loss_list, val_loss_list = [], []
    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()

            outputs = net(inputs.float())
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            """
            if i % 50 == 49:
                print("[%d, %4d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 50))
                running_loss = 0.0
            """
        net.eval()
        eval_loss = 0.0
        with torch.no_grad():
            for data in valloader:
                inputs, labels = data[0].to(device), data[1].to(device)

                outputs = net(inputs.float())
                loss = criterion(outputs, labels.float())
                eval_loss += loss.item()
        print("Epoch [%d/%d], train_loss: %.4f, val_loss: %.4f" % (epoch + 1, epochs, running_loss/train_size, eval_loss/val_size))
        train_loss_list.append(running_loss/train_size)
        val_loss_list.append(eval_loss/val_size)

    print ("Finished Training")

    PATH = "./dcase2017_net.pth"
    torch.save(net.state_dict(), PATH)

    plt.figure()
    plt.plot(range(epochs), train_loss_list, color='blue', linestyle='-', label='train_loss')
    plt.plot(range(epochs), val_loss_list, color='green', linestyle='--', label='val_loss')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Training and validation loss')
    plt.grid()
    plt.savefig("result.pdf", bbox_inches ="tight")
