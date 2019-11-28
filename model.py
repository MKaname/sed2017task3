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
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.dropout(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)
        x = self.dropout(x)
        x = x.permute(0, 2, 1, 3)#(N,T,128,2)
        # print(x.size())
        x = x.reshape(x.shape[0],x.shape[1],-1)#(N,T,256)
        # print(x.size())
        x , _= self.gru(x)
        x = self.linear1(x)
        x = self.dropout(x)
        x = torch.sigmoid(self.linear2(x))
        return x

if __name__ == "__main__":
    mbe_dir = "mbe/street"
    label_dir = "label/street"
    validation_dir = "evaluation_setup"
    subdivs = 256
    epochs = 200
    for fold in np.array([1, 2, 3, 4]):
        print('\n\n----------------------------------------------')
        print('FOLD: {}'.format(fold))
        print('----------------------------------------------\n')

        device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
        print(device)

        transform = transforms.Compose([transforms.ToTensor()])
        batch_size = 128
        X_train, Y_train, X_val, Y_val = processing.make_validation_data_1(mbe_dir, label_dir, validation_dir, fold)

        X_train = processing.split_in_seqs(X_train, subdivs)
        X_val = processing.split_in_seqs(X_val, subdivs)
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
        X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], X_val.shape[2], 1)
        Y_train = processing.split_in_seqs(Y_train, subdivs)
        Y_val = processing.split_in_seqs(Y_val, subdivs)
        train_dataset = MyDataset(X_train, Y_train, transform = transform)
        val_dataset = MyDataset(X_val, Y_val, transform = transform)

        train_size = len(train_dataset)
        val_size = len(val_dataset)
        print(train_size)
        print(val_size)

        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = False)
        valloader = torch.utils.data.DataLoader(val_dataset, batch_size = batch_size, shuffle = False)

        net = Net().float()
        net.to(device)
        criterion = nn.BCELoss(reduction="sum")
        optimizer = optim.Adam(net.parameters())

        best_epoch, pat_cnt, best_er, f1_for_best_er, best_conf_mat = 0, 0, 99999, None, None
        tr_loss, val_loss, f1_overall_1sec_list, er_overall_1sec_list = [0] * epochs, [0] * epochs, [0] * epochs, [0] * epochs
        train_loss_list, val_loss_list = [], []
        posterior_thresh = 0.5

        for epoch in range(epochs):
            net.train()
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data[0].to(device), data[1].to(device)

                optimizer.zero_grad()

                outputs = net(inputs.float())
                # import sys
                # sys.exit(1)
                loss = criterion(outputs, labels.float())
                loss.backward()
                optimizer.step()

                running_loss += loss.item()/train_size

            net.eval()
            eval_loss = 0.0
            with torch.no_grad():
                for data in valloader:
                    inputs, labels = data[0].to(device), data[1].to(device)
                    outputs = net(inputs.float())
                    # outputs_thresh = pred > posterior_thresh

                    loss = criterion(outputs, labels.float())
                    eval_loss += loss.item()/val_size

            print("Epoch [%d/%d], train_loss: %.4f, val_loss: %.4f" % (epoch + 1, epochs, running_loss/train_size, eval_loss/val_size))
            train_loss_list.append(running_loss/train_size)
            val_loss_list.append(eval_loss/val_size)

        print ("Finished Training")

        PATH = "./dcase2017_net_fold{}.pth".format(fold)
        torch.save(net.state_dict(), PATH)

        plt.figure()
        plt.plot(range(epochs), train_loss_list, color='blue', linestyle='-', label='train_loss')
        plt.plot(range(epochs), val_loss_list, color='green', linestyle='--', label='val_loss')
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title('Training and validation loss')
        plt.grid()
        plt.savefig("result_fold{}.pdf".format(fold), bbox_inches ="tight")
