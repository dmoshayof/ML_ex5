import torch
from model import Net
from functions import load_data,train_model,test_model,print_prediction

lr=0.001



if __name__=='__main__':
    train,validation=load_data()
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    model=Net().to(device)
    optimizer=torch.optim.Adam(model.parameters(), lr=lr)

    # train the model
    train_model(train,optimizer,model,device)
    # test the model
    test_model(validation,model,device)

    print_prediction(model,device)

