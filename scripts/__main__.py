import numpy as np
from scripts import neural_net as NN
from scripts import io
from Bio import Seq
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold

"""First I will use the neural net to solve the 8x3x8 autoendcoder problem"""
# running the neural net on the identity matrix
X = np.identity(8)
# labels are equal to the input since we want it to return the same input
y = X
nn = NN.NeuralNetwork(X, y, 3, .05)
print('training neural net on 8 x 8 identity matrix...')
nn.train(X, y, iterations = 100000)
# round the output and print it
print(np.round(nn.output))

""" I want to make sure it runs on matrices of different dimensions to make sure
the function is selecting the correct dimensions """
# an odd square matrix
X = np.identity(5)
y = X
nn = NN.NeuralNetwork(X, y, 3, .05)
print('training neural net on 5 x 5 identity matrix...')
nn.train(X, y, iterations = 100000)
print(np.round(nn.output))

# a non-square matrix
X = X[0:4,]
y = X
print('training neural net on 4 x 5 matrix...')
nn.train(X, y, iterations = 100000)
print(np.round(nn.output))

"""Now importing the binding site data and converting to a format I can train on"""
print('Reading in sequences...')
# read in the positive binding sites
positiveF = io.read_txt('data/rap1-lieb-positives.txt')
# reverse complement them to get more data
positivesRC = [str(Seq.Seq(x).reverse_complement()) for x in positiveF]
# combine positives and reverse complements
positiveSeqs = positiveF + positivesRC
# read in the negative sequences
negativeSeqs = io.read_fasta('data/yeast-upstream-1k-negative.fa')
# downsample the negative sequences
negativeSeqs_sub = io.downsample(negativeSeqs, len(positiveSeqs), 17)
# remove any negative sequences that are in the positive list by chance
negativeSeqs_sub = [n for n in negativeSeqs_sub if n not in positiveSeqs]
# read in the test sequences
testSeqs = io.read_txt('data/rap1-lieb-test.txt')
# convert all three lists from characters (bases) to binary
positiveBin = io.translate_list(positiveSeqs)
testBin = io.translate_list(testSeqs)
negativeBin = io.translate_list(negativeSeqs_sub)

# combine the positives and negatives in a single array
x = np.concatenate([positiveBin,negativeBin])
# create a label array, 1 for positive, 0 for negative
y = np.concatenate([np.repeat(1,len(positiveBin)), np.repeat(0,len(negativeBin))])
# rehshape so labels has two dimensions
y = y.reshape(len(y),1)


"""Now I need to figure out the best set of parameters"""
"""first I will look at how my model error and validation error change with
increasing number of iterations"""
xtrain, holdoutx, ytrain, holdouty = train_test_split(x, y, test_size=0.1)
print('Testing number of iterations')
iterations = [1, 10, 100, 1000, 10000]
m_errors = []
v_errors = []
kf = KFold(n_splits=10, shuffle = True)
for i in iterations:
        m = []
        v = []
        for train_index, test_index in kf.split(xtrain):
            xr, xv = xtrain[train_index], xtrain[test_index]
            yr, yv = ytrain[train_index], ytrain[test_index]
            nn = NN.NeuralNetwork(np.array(xr), np.array(yr), 10, .01)
            nn.train(np.array(xr), np.array(yr), iterations = i)
            m.append(nn.error)
            v.append(nn.predict(np.array(xv), y = np.array(yv))[1])
        v_errors.append(np.mean(v))
        m_errors.append(np.mean(m))
#plotting errors vs. iterations
fig1 = plt.figure(dpi = 100)
plt.xscale("log")
plt.xlabel('iterations')
plt.ylabel('error')
plt.title("Varying_Iterations")
plt.plot(iterations, m_errors, label = 'model')
plt.plot(iterations, v_errors, label = 'validation_set')
plt.legend()
fig1.savefig('testing_iterations.png')
plt.cla()

print('Lowest validation error is %s' %min(v_errors))
print('Corresponding model error is %s'%m_errors[v_errors.index(min(v_errors))])

# confirm that the error is not due to overfitting by testing the holdout data
nn = NN.NeuralNetwork(np.array(xtrain), np.array(ytrain), 10, .001)
nn.train(np.array(xtrain), np.array(ytrain), iterations = 100)
prediction, error = nn.predict(np.array(holdoutx), np.array(holdouty))

print('testing learning parameters...')
#hold out data
learning_rate = [.0001, .00025, .0005, .001, .005, .01, .05, .1, .15]
nodes = [2, 4, 6, 8, 10, 12, 18, 30, 60, 68]
paramErrors = pd.DataFrame(index = nodes, columns = learning_rate)
# make empty matrix
paramErrors = pd.DataFrame(index = nodes, columns = learning_rate)
kf = KFold(n_splits=10, shuffle = True)
for n in nodes:
    for l in learning_rate:
        validation_error = []
        for train_index, test_index in kf.split(xtrain):
            xr, xv = xtrain[train_index], xtrain[test_index]
            yr, yv = ytrain[train_index], ytrain[test_index]
            nn = NN.NeuralNetwork(np.array(xr), np.array(yr), n, l)
            nn.train(np.array(xr), np.array(yr), iterations = 1000)
            error = nn.predict(np.array(xv), y = np.array(yv))[1]
            validation_error.append(error)
        paramErrors.loc[n, l] = np.mean(validation_error)

print('plotting parameters tested...')
paramErrors = paramErrors[paramErrors.columns].astype(float)
plt.xscale("linear")
cmap = sns.cm.rocket_r
hm = sns.heatmap(paramErrors, annot=True,vmin = .03, vmax = .3, cmap = cmap)
fig = hm.get_figure()
fig.savefig('params_heatmap.png')

print('testing held out data with best parameters...')
# train all but held out w/ best parameters
nn = NN.NeuralNetwork(np.array(xtrain), np.array(ytrain), 12, .01)
nn.train(np.array(xtrain), np.array(ytrain))
prediction, error = nn.predict(np.array(holdoutx), np.array(holdouty))

print('Error for holdout data is %s' %error)
from sklearn import metrics
fpr, tpr, thresholds = metrics.roc_curve(holdouty, prediction)
auc = metrics.auc(fpr, tpr)
print('AUC for hold data is %s' %auc)

nn=NN.NeuralNetwork(x, y, 23, .01)
nn.train(x, y)
predicted_sites = nn.predict(testBin)
predictions = pd.DataFrame(list(zip(testSeqs, predicted_sites)))
predictions.to_csv('cmartyn_predictions.txt', sep = '\t', header= False, index = False)
