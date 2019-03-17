import numpy as np
from scripts import neural_net as NN
from scripts import io
from Bio import Seq
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from sklearn import metrics

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
print('Testing number of iterations...')
iterations = [1, 10, 100, 1000, 10000]
m_errors = []
v_errors = []
kf = KFold(n_splits=5, shuffle = True)
for i in iterations: # loop through iterations
        m = [] # store the error of the model
        v = [] # store the error of the validation set
        for train_index, test_index in kf.split(xtrain):
            # use indices generate training and testing sets for each fold
            xr, xv = xtrain[train_index], xtrain[test_index]
            yr, yv = ytrain[train_index], ytrain[test_index]
            # generate neural network with best guess parameters
            nn = NN.NeuralNetwork(np.array(xr), np.array(yr), 10, .01)
            # train with different number of iterations and store the error
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
fig1.savefig('./outputs/testing_iterations.png')
plt.cla()


print('Lowest validation error is %s' %min(v_errors))
print('Corresponding model error is %s'%m_errors[v_errors.index(min(v_errors))])
# confirm that the error is not due to overfitting by testing the holdout data
nn = NN.NeuralNetwork(np.array(xtrain), np.array(ytrain), 10, .001)
nn.train(np.array(xtrain), np.array(ytrain), iterations = 100)
prediction, error = nn.predict(np.array(holdoutx), np.array(holdouty))
print('Holdout error is %s' %error)

"""Now I want to loop through several different learning rates and hidden
layer size to figure out the best combination of parameters"""

print('testing learning parameters...')

learning_rate = [.0001, .00025, .0005, .001, .005, .01, .05, .1, .15]
nodes = [2, 4, 6, 8, 10, 12, 18, 30, 60]
# make empty dataframe to store results
paramErrors = pd.DataFrame(index = nodes, columns = learning_rate)
# hold out 10% of the data, trian and test on 90%
xtrain, holdoutx, ytrain, holdouty = train_test_split(x, y, test_size=0.1)
# generate indices for the folds, I will do 5-fold since we only have about 500 observations
kf = KFold(n_splits=5, shuffle = True)
for n in nodes: # loop through hidden layer size
    for l in learning_rate: # loop through learning rates
        validation_error = [] # keep track of the average error for each combination
        for train_index, test_index in kf.split(xtrain):
            # use indices generate training and testing sets for each fold
            xr, xv = xtrain[train_index], xtrain[test_index]
            yr, yv = ytrain[train_index], ytrain[test_index]
            # generate NN and train on training set
            nn = NN.NeuralNetwork(np.array(xr), np.array(yr), n, l)
            nn.train(np.array(xr), np.array(yr), iterations = 1000)
            # predict values for test set (fold)
            error = nn.predict(np.array(xv), y = np.array(yv))[1]
            validation_error.append(error) # average errors across folds
        paramErrors.loc[n, l] = np.mean(validation_error) # store error in dataframe

print('plotting parameters tested...')
# fix dataframe so it will plot
paramErrors = paramErrors[paramErrors.columns].astype(float)
cmap = sns.cm.rocket_r
# plot dataframe as heatmap
hm = sns.heatmap(paramErrors, annot=True,vmin = .03, vmax = .3, cmap = cmap)
fig = hm.get_figure()
fig.savefig('./outputs/params_heatmap.png')

print('testing held out data with best parameters...')
# train all folds together  w/ best parameters from heatmap
nn = NN.NeuralNetwork(np.array(xtrain), np.array(ytrain), 12, .001)
nn.train(np.array(xtrain), np.array(ytrain))
# predict on the 10% of data that was held out
prediction, error = nn.predict(np.array(holdoutx), np.array(holdouty))
print('Error for holdout data is %s' %error)

# evaluate AUC for holdout data
fpr, tpr, thresholds = metrics.roc_curve(holdouty, prediction)
auc = metrics.auc(fpr, tpr)
print('AUC for holdout data is %s' %auc)

# now train a neural net on all of the available data
nn=NN.NeuralNetwork(x, y, 12, .001)
nn.train(x, y)
# make predictions for the test sequences
predicted_sites = nn.predict(testBin).flatten()
# combine predictions with sequences and write to txt
predictions = pd.DataFrame(list(zip(testSeqs, predicted_sites)))
predictions.to_csv('./outputs/cmartyn_predictions.txt', sep = '\t', header= False, index = False)
