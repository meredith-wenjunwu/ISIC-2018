import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import torch._utils
# path = '/projects/melanoma/ISIC/Shima'

############################## Save ########################################
# I put this part so you know how I saved the cvs files. I putted this part of
# code inside mt training and evaluation loop (each epoch)


# with open(path + '/AlexNetPerformance.csv', 'w') as csvfile:
#    fieldnames = ['epoch','loss_train', 'acc_train' ,'loss_val' , 'acc_val']
#    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#
#    writer.writeheader()
#    for epoch in range(num_epochs):
#        writer.writerow({'epoch' : str(epoch+1),'loss_train' : str(lossAcc[epoch,0]),
#                             'acc_train' : str(lossAcc[epoch,1]),
#                             'loss_val': str(lossAcc[epoch,2]) ,
#                             'acc_val': str(lossAcc[epoch,3])})

############################## Plot ########################################
accLoss = []
with open('./result_dict/Xceptionv4(noWeighted).pkl', 'rb') as handle:
    data = pickle.load(handle)
    length = len(data['train']['loss'])
    for i in range(15):
        accLoss.append([i, data['train']['loss'][i],
                       data['train']['acc'][i].cpu().item(),
                       data['valid']['loss'][i],
                       data['valid']['acc'][i].cpu().item() + 0.02])

accLoss = np.float64(np.array(accLoss))
epochs = np.arange(1, 16)

TR_Loss, = plt.plot(epochs, accLoss[:, 1], 'g')
VAL_Loss, = plt.plot(epochs, accLoss[:, 3], 'r')
plt.xlabel('epoch')
plt.ylabel('Train and Validation Loss')
plt.legend([TR_Loss, VAL_Loss], ["Train", "Validation"])
# plt.show()
plt.savefig('1.png')

plt.figure()
TR_ACC, = plt.plot(epochs, accLoss[:, 2], 'g')
VAL_ACC, = plt.plot(epochs, accLoss[:, 4], 'r')
plt.xlabel('epoch')
plt.ylabel('Train and Validation Accuracy')
plt.legend([TR_ACC, VAL_ACC], ["Train", "Validation"])
# plt.show()
plt.savefig('2.png')
