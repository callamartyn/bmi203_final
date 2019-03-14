import numpy as np
from scripts import neural_net as NN
from scripts import io
from Bio import Seq
from matplotlib import pyplot as plt

"""First I will use the neural net to solve the 8x3x8 autoendcoder problem"""
# running the neural net on the identity matrix
X = np.identity(8)
# labels are equal to the input since we want it to return the same input
y = X
nn = NN.NeuralNetwork(X, y, 3, .05)
print('training neural net on 8 x 8 identity matrix')
nn.train(X, y, iterations = 100000)
# round the output and print it
print(np.round(nn.output))

""" I want to make sure it runs on matrices of different dimensions to make sure
the function is selecting the correct dimensions """
# an odd square matrix
X = np.identity(5)
y = X
nn = NN.NeuralNetwork(X, y, 3, .05)
print('training neural net on 5 x 5 identity matrix')
nn.train(X, y, iterations = 100000)
print(np.round(nn.output))

# a non-square matrix
X = X = X[0:4,]
y = X
print('training neural net on 4 x 5 matrix')
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

print('testing number of iterations...')
iterations = [1, 10, 100, 1000, 10000, 100000]
m_errors = []
v_errors = []
for i in iterations:
    m, v = NN.mean_cross_validate(x, y, 6, .01, iterations = i)
    m_errors.append(m)
    v_errors.append(v)

print('Lowest validation error is %s' %min(v_errors))
print('Corresponding model error is %s'%m_errors[v_errors.index(min(v_errors))])

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

"""Now I will see how different values of the learning rate changes the
validation error"""
print('Testing learning rates...')
learning_rate = [.0001, .00025, .0005, .001, .005, .01, .05, .1, .15]
l_errors = []
for l in learning_rate:
    val_error = NN.mean_cross_validate(x, y, 3, l)[1]
    l_errors.append(val_error)
print('Best learning rate is %s' %learning_rate[l_errors.index(min((l_errors)))])

# plotting validation error vs. learning rate
fig2 = plt.figure(dpi = 100)
plt.xscale("log")
filename = 'testing_lr.png'
plt.xlabel('Learning_rates')
plt.ylabel('error')
plt.title("Testing the Learning Rate")
plt.plot(learning_rate, l_errors)
fig2.savefig('learning_rate.png')


"""Now I will test some different numbers of nodes for the hidden layer"""
print('Testing hidden layer node size...')
nodes = [2, 4, 6, 8, 10, 12, 14, 16]
n_errors = []
for n in nodes:
    val_error = NN.mean_cross_validate(x, y, n, .001)[1]
    n_errors.append(val_error)
print('Best number of nodes is %s' %nodes[n_errors.index(min(n_errors))])

# plotting
fig3 = plt.figure(dpi = 100)
plt.xlabel('Number of nodes')
plt.ylabel('error')
plt.title("Testing the Number of Hidden Layer Nodes")
plt.plot(nodes, n_errors)
fig3.savefig('testing_nodes.png')
