from scripts import neural_net, io
import numpy as np

# testing that training the neural net on the identity matrix will retrun
#the identity matrix
def test_NN():
    X = np.identity(8)
    y = X
    nn = neural_net.NeuralNetwork(X, y, 3, .05)
    nn.train(X, y, iterations = 100000)
    out = np.round(nn.output)
    assert np.array_equiv(out, X)

# testing that the sequence to binary function is correctly converting the sequences
def test_translation():
    testseq = 'TGGCAT'
    x = io.seq_to_binary(testseq)
    assert np.array_equiv(x, [0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 1., 0., 1.,
       0., 0., 0., 0., 1., 0., 0.])

dir = 'data/'
d = dir + 'rap1-lieb-positives.txt'
io.read_txt(d)
