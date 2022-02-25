import os.path

import torch
from torch_geometric.data import DataLoader
from SimiTUDatasets import Metalistic
import configparser
from Semimodel import Model, LabelPropagation
from ProtoNetLoss import PrototypicalLoss as loss_fn
from torch.optim import SGD, Adam
from tqdm import tqdm
from SimiTUDatasets import EposideSampler

def init_sampler(dataset, classes_per_it, NumLabeled, NumQuert, iterations = 100):
    Indices = dataset._indices
    sampler = EposideSampler(labels=dataset.data.y[Indices], classes_per_it=classes_per_it,
                             num_samples=NumLabeled + NumQuert, iterations=100)
    return sampler

if __name__ == '__main__':
    config = configparser.ConfigParser()
    ConfigFilename = r'./config.ini'
    config.read(ConfigFilename)
    data_type = 'NCI1'
    NumLabeled = int(config.get(data_type, 'NShots'))
    NumunLabeled = int(config.get(data_type, 'Nunlabeled'))
    NumQuert = int(config.get(data_type, 'NQuery'))
    EPOCH = int(config.get(data_type, 'TotalEpoch'))
    savefiles = config.get(data_type, 'savefiles')
    ckps_path = os.path.join(savefiles, 'ckps')

    traindata = Metalistic(config, split='train')
    valdata = Metalistic(config, split='val')
    NumFeatures, NumClasses = traindata.NumFeatures, traindata.NumClasses

    traindataset = traindata.DatasetPartition()
    valdataset = valdata.DatasetPartition()

    trainLblIndices, trainUnlblIndices = traindata.LblAndUnlblSplit()
    valLblIndices, valUnlblIndices = valdata.LblAndUnlblSplit()

    traindatasetLbl = traindataset[trainLblIndices]
    valdatasetLbl = valdataset[valLblIndices]
    traindatasetUnlbl = traindataset[trainUnlblIndices]
    valdatasetUnlbl = valdataset[valUnlblIndices]

    samplerTrainLbl = init_sampler(traindatasetLbl, 2, NumLabeled, NumQuert)
    samplerValLbl = init_sampler(valdatasetLbl, 2, NumLabeled, NumQuert)

    samplerTrainUnlbl = init_sampler(traindatasetUnlbl, 2, NumunLabeled, NumQuert = 0)
    samplerValUnlbl = init_sampler(valdatasetUnlbl, 2, NumunLabeled, NumQuert = 0)

    train_loader = DataLoader(dataset=traindatasetLbl, batch_sampler=samplerTrainLbl)
    val_loader = DataLoader(dataset=valdatasetLbl, batch_sampler=samplerValLbl)
    trainUnlabel_loader = DataLoader(dataset=traindatasetUnlbl, batch_sampler=samplerTrainUnlbl)
    valUnlbl_loader = DataLoader(dataset=valdatasetUnlbl, batch_sampler=samplerValUnlbl)
    # train_loader = DataLoader(dataset=traindatasetLbl, batch_size=NumClasses * (NumLabeled + NumQuert), shuffle=False, drop_last = True)
    # val_loader = DataLoader(dataset=valdatasetLbl, batch_size=NumClasses * (NumLabeled + NumQuert), shuffle=False, drop_last = True)
    # trainUnlabel_loader = DataLoader(dataset=traindatasetUnlbl, batch_size=NumClasses * NumunLabeled, shuffle=True, drop_last = True)
    # valUnlbl_loader = DataLoader(dataset=valdatasetUnlbl, batch_size=NumClasses * NumunLabeled, shuffle=True, drop_last=True)

    loss_fn = loss_fn(NumLabeled)
    model = Model(NumFeatures, NumClasses)
    # model = LabelPropagation(NumFeatures, NumClasses, config)
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    BestValAcc = 0
    it = tqdm(range(EPOCH))
    for i in it:
        loss_mean = 0
        acc_mean = 0
        for index, LblData in enumerate(train_loader):
            embeddings = model(LblData)
            UnlblData = next(iter(trainUnlabel_loader))
            UnlblEmbeddings = model(UnlblData)
            loss, acc = loss_fn(embeddings, LblData.y, UnlblEmbeddings)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_mean += loss.item()
            acc_mean += acc.item()
        print('\n')
        print('----------Train Loss and Train Acc', loss_mean / len(train_loader), acc_mean / len(train_loader))
        if i % 1 == 0:
            model.eval()
            valloss_mean = 0
            valacc_mean = 0
            for index, TestData in enumerate(val_loader):
                embeddings = model(TestData)
                UnlblData = next(iter(valUnlbl_loader))
                UnlblEmbeddings = model(UnlblData)
                loss, acc = loss_fn(embeddings, TestData.y, UnlblEmbeddings)

                valloss_mean += loss.item()
                valacc_mean += acc.item()
            print('\n')
            print('----------Val Loss and Val Acc', valloss_mean / len(val_loader), valacc_mean / len(val_loader))
            print('=========================================================')

            if valacc_mean / len(val_loader) > BestValAcc:
                BestValAcc = valacc_mean / len(val_loader)
                torch.save(model.state_dict(), os.path.join(ckps_path, 'BestModel.pth'))

    testdata = Metalistic(config, split='test')
    testdataset = testdata.DatasetPartition()
    testLblIndices, testUnlblIndices = testdata.LblAndUnlblSplit()
    testdatasetLbl = testdataset[testLblIndices]
    testdatasetUnlbl = testdataset[testUnlblIndices]

    test_loader = DataLoader(dataset=testdatasetLbl, batch_size=NumClasses * (NumLabeled + NumQuert), shuffle=False)
    testUnlbl_loader = DataLoader(dataset=testdatasetUnlbl, batch_size=NumClasses * NumunLabeled, shuffle=False)
    model.eval()
    testloss_mean = 0
    testacc_mean = 0
    for index, TestData in enumerate(test_loader):
        embeddings = model(TestData)
        UnlblData = next(iter(testUnlbl_loader))
        UnlblEmbeddings = model(UnlblData)
        loss, acc = loss_fn(embeddings, TestData.y, UnlblEmbeddings)

        testloss_mean += loss.item()
        testacc_mean += acc.item()

    print('----------', testloss_mean / len(test_loader), testacc_mean / len(test_loader))
    print('=========================================================')
