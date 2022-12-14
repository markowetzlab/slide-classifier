# Adam Berman and Marcel Gehrung

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import pickle
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser(description='Plot accuracy and loss curves.')
parser.add_argument("-learningdata", required=True, help="path to pickled learning data")
parser.add_argument("-class_name", default=False, help="class to plot precision and recall curves for")
parser.add_argument("-whatdetected", default="atypia", help="the class being detected used only for plot titles")
args = parser.parse_args()

matplotlib.rcParams.update({'font.size': 12})

trainingPartitionSizes = [43]
#numEpochs = 30
maxValidationAcc = []
maxTrainingAcc = []
for partitionIdx,partitionSize in enumerate(trainingPartitionSizes):
    learningStats = pickle.load(open(args.learningdata, "rb"))
    #print(learningStats['train'])
    #print((partitionIdx)*numEpochs)
    #print((partitionIdx+1)*numEpochs)
    trainLoss = [tLoss["loss"] for tLoss in learningStats['train']]
    valLoss = [tLoss["loss"] for tLoss in learningStats['val']]
    trainAcc = [tLoss["weighted_accuracy"] for tLoss in learningStats['train']]
    valAcc = [tLoss["weighted_accuracy"] for tLoss in learningStats['val']]
    #print(len(valLoss))
    maxTrainingAcc.append(np.max(trainAcc))
    maxValidationAcc.append(np.max(valAcc))

    #plt.figure()
    #plt.plot(np.arange(numEpochs),trainLoss)
    #plt.plot(np.arange(numEpochs),valLoss)
    #plt.show(block=False)

numEpochs = len(trainLoss)
'''
plt.figure()
plt.plot(trainingPartitionSizes,maxValidationAcc,'o-',label="Peak validation accuracy")
plt.plot(trainingPartitionSizes,maxTrainingAcc,'o-',label="Peak training accuracy")
#plt.title("ResNet18 training")
plt.xlabel("Training partition size (number of patients)")
plt.ylabel("Accuracy")
plt.ylim((0,1))
plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
plt.legend()
plt.xticks(trainingPartitionSizes)
plt.show(block=False)
plt.savefig(os.path.join(os.path.split(args.learningdata)[0], "partitionedLearningCurve.pdf"))
'''
#quit()

#learningStatsRN18 = pickle.load(open('model_files/learningCurve-resnet18-70-30.p', "rb"))
learningStatsVGG16 = pickle.load(open(args.learningdata, "rb"))
#learningStatsIncV3 = pickle.load(open('model_files/learningCurve-inceptionv3-70-30.p', "rb"))

#valAccRN18 = [tLoss[1] for tLoss in learningStatsRN18['val']]
valAccVGG16 = [tLoss["weighted_accuracy"] for tLoss in learningStatsVGG16['val']]
valLossVGG16 = [tLoss["loss"] for tLoss in learningStatsVGG16['val']]
#valAccIncV3 = [tLoss[1] for tLoss in learningStatsIncV3['val']]

#trAccRN18 = [tLoss[1] for tLoss in learningStatsRN18['train']]
trAccVGG16 = [tLoss["weighted_accuracy"] for tLoss in learningStatsVGG16['train']]
trLossVGG16 = [tLoss["loss"] for tLoss in learningStatsVGG16['train']]
#trAccIncV3 = [tLoss[1] for tLoss in learningStatsIncV3['train']]

#print(args.class_name)
#print(learningStatsVGG16['val'][0]["precision"])
if args.class_name:
    valPrecision = [tLoss["precision"][args.class_name] for tLoss in learningStatsVGG16['val']]
    valRecall = [tLoss["recall"][args.class_name] for tLoss in learningStatsVGG16['val']]
    valF1 = [((2*p*r)/(p+r)) for p,r in zip(valPrecision, valRecall)]
    trPrecision = [tLoss["precision"][args.class_name] for tLoss in learningStatsVGG16['train']]
    trRecall = [tLoss["recall"][args.class_name] for tLoss in learningStatsVGG16['train']]
    trF1 = [((2*p*r)/(p+r)) for p,r in zip(trPrecision, trRecall)]


plt.figure()
#f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True,figsize=(9,3.5))
#ax1.plot(np.arange(25)+1,valAccRN18,'go-',label="Validation",markersize=4)
plt.plot(np.arange(numEpochs)+1,valAccVGG16,'go-',label="Validation",markersize=4)
#ax3.plot(np.arange(25)+1,valAccIncV3,'go-',label="Validation",markersize=4)
#ax1.plot(np.arange(25)+1,trAccRN18,'bo-.',label="Training",alpha=0.6,markersize=4)
plt.plot(np.arange(numEpochs)+1,trAccVGG16,'bo-.',label="Training",alpha=0.6,markersize=4)
#ax3.plot(np.arange(25)+1,trAccIncV3,'bo-.',label="Training",alpha=0.6,markersize=4)

#ax1.axhline(y=np.max(valAccRN18),color="r",alpha=0.4)
plt.axhline(y=np.max(valAccVGG16),color="r",alpha=0.4)
#ax3.axhline(y=np.max(valAccIncV3),color="r",alpha=0.4,label="Peak line")

#ax1.set_title("ResNet-18")
plt.title("VGG-16 "+args.whatdetected+" detection model (max "+str(round(np.max(valAccVGG16),2))+" at epoch "+str(valAccVGG16.index(np.max(valAccVGG16)))+')')
#ax3.set_title("Inception v3")
plt.ylabel("Weighted accuracy")
plt.xlabel("Epochs")
plt.legend()
#plt.xticks(np.arange(25)+1)
#plt.tight_layout()
plt.show(block=False)
#plt.savefig("figures/CNN-benchmark-70-30-learning.pdf")
plt.savefig(os.path.join(os.path.split(args.learningdata)[0], "learningCurveAccuracy.pdf"))

plt.figure()
plt.plot(np.arange(numEpochs)+1,valLossVGG16,'go-',label="Validation",markersize=4)
plt.plot(np.arange(numEpochs)+1,trLossVGG16,'bo-.',label="Training",alpha=0.6,markersize=4)
plt.title("VGG-16 "+args.whatdetected+" detection model")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend()
plt.show(block=False)
plt.savefig(os.path.join(os.path.split(args.learningdata)[0], "learningCurveLoss.pdf"))

if args.class_name:
    plt.figure()
    plt.plot(np.arange(numEpochs)+1,valRecall,'go-',label="Validation",markersize=4)
    plt.plot(np.arange(numEpochs)+1,trRecall,'bo-.',label="Training",alpha=0.6,markersize=4)
    plt.axhline(y=np.max(valRecall),color="r",alpha=0.4)
    plt.title("VGG-16 "+args.whatdetected+" detection model - "+args.class_name+" class (max "+str(round(np.max(valRecall),2))+" at epoch "+str(valRecall.index(np.max(valRecall)))+')')
    plt.ylabel(args.class_name+" recall")
    plt.xlabel("Epochs")
    plt.legend()
    plt.show(block=False)
    plt.savefig(os.path.join(os.path.split(args.learningdata)[0], "learningCurveRecall-"+args.class_name+".pdf"))

    plt.figure()
    plt.plot(np.arange(numEpochs)+1,valPrecision,'go-',label="Validation",markersize=4)
    plt.plot(np.arange(numEpochs)+1,trPrecision,'bo-.',label="Training",alpha=0.6,markersize=4)
    plt.axhline(y=np.max(valPrecision),color="r",alpha=0.4)
    plt.title("VGG-16 "+args.whatdetected+" detection model - "+args.class_name+" class (max "+str(round(np.max(valPrecision),2))+" at epoch "+str(valPrecision.index(np.max(valPrecision)))+')')
    plt.ylabel(args.class_name+" precision")
    plt.xlabel("Epochs")
    plt.legend()
    plt.show(block=False)
    plt.savefig(os.path.join(os.path.split(args.learningdata)[0], "learningCurvePrecision-"+args.class_name+".pdf"))

    plt.figure()
    plt.plot(np.arange(numEpochs)+1,valF1,'go-',label="Validation",markersize=4)
    plt.plot(np.arange(numEpochs)+1,trF1,'bo-.',label="Training",alpha=0.6,markersize=4)
    plt.axhline(y=np.max(valF1),color="r",alpha=0.4)
    plt.title("VGG-16 "+args.whatdetected+" detection model - "+args.class_name+" class (max "+str(round(np.max(valF1),2))+" at epoch "+str(valF1.index(np.max(valF1)))+')')
    plt.ylabel(args.class_name+" F1")
    plt.xlabel("Epochs")
    plt.legend()
    #plt.tight_layout()
    plt.show(block=False)
    plt.savefig(os.path.join(os.path.split(args.learningdata)[0], "learningCurveF1-"+args.class_name+".pdf"))

plt.show()
