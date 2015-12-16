from subprocess import call
import os
import tempfile

class FM:
    """ Class that wraps `libFM` parameters. For more information read
    [libFM manual](http://www.libfm.org/libfm-1.42.manual.pdf)

    Parameters
    ----------
    task : string, MANDATORY
        regression: for regression
        classification: for binary classification

    num_iter: int, optional
        Number of iterations
        Defaults to 100
    init_stdev : double, optional
        Standard deviation for initialization of 2-way factors
        Defaults to 0.1
    k0 : bool, optional
        Use bias.
        Defaults to True
    k1 : bool, optional
        Use 1-way interactions.
        Defaults to True
    k2 : int, optional
        Dimensionality of 2-way interactions.
        Defaults to 8
    learning_method: string, optional
        sgd: parameter learning with SGD
        sgda: parameter learning with adpative SGD
        als: parameter learning with ALS
        mcmc: parameter learning with MCMC
        Defaults to 'mcmc'
    learn_rate: double, optional
        Learning rate for SGD
        Defaults to 0.1
    r0_regularization: int, optional
        bias regularization for SGD and ALS
        Defaults to 0
    r1_regularization: int, optional
        1-way regularization for SGD and ALS
        Defaults to 0
    r2_regularization: int, optional
        2-way regularization for SGD and ALS
        Defaults to 0
    verbose: bool, optional
        How much infos to print
        Defaults to False.
    """

    """
    ### unsused libFM flags
    meta: filename for meta information about data set
        FUTURE WORK
    validation: filename for validation data (only for SGDA)
        FUTURE WORK
    cache_size: cache size for data storage (only applicable if data is in binary format), default=infty
        datafile is text so we don't need this parameter
    relation: BS - filenames for the relations, default=''
        not dealing with BS extensions since they are only used for binary files
    """

    def __init__(self,
                 task,
                 num_iter = 100,
                 init_stdev = 0.1,
                 k0 = True,
                 k1 = True,
                 k2 = 8,
                 learning_method = 'mcmc',
                 learn_rate = 0.1,
                 r0_regularization = 0,
                 r1_regularization = 0,
                 r2_regularization = 0,
                 verbose = False):

        # gets first letter of either regression or classification
        self.__task = task[0]
        self.__num_iter = num_iter
        self.__init_stdev = init_stdev
        self.__dim = "%d,%d,%d" % (int(k0), int(k1), k2)
        self.__learning_method = learning_method
        self.__learn_rate = learn_rate
        self.__regularization = "%d,%d,%d" % (r0_regularization, r1_regularization, r2_regularization)
        self.__verbose = int(verbose)

        # gets real path of package
        self.__libfm_path=os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                       "libfm/bin/libFM")

    def run(self, x_train, y_train, x_test, y_test):
        """Run factorization machine model against train and test data

        Parameters
        ----------
        x_train : {array-like, matrix}, shape = [n_train, n_features]
            Training data
        y_train : numpy array of shape [n_train]
            Target values
        x_test: {array-like, matrix}, shape = [n_test, n_features]
            Testing data
        y_test : numpy array of shape [n_test]
            Testing target values

        Return
        -------
        Returns `namedtuple` with the following properties:

        predictions: array [n_samples of x_test]
           Predicted target values per element in x_test.
        global_bias: float
            If k0 is True, returns the model's global bias w0
        weights: array [n_features]
            If k1 is True, returns the model's weights for each features Wj
        pairwise_interactions: numpy matrix [n_features x k2]
            Matrix with pairwise interactions Vj,f
        rlog: pandas dataframe [nrow = num_iter]
            `pandas` DataFrame with measurements about each iteration
        """

        from sklearn.datasets import dump_svmlight_file

        _,train_path = tempfile.mkstemp()
        _,test_path = tempfile.mkstemp()
        _,out_path = tempfile.mkstemp()
        _,model_path = tempfile.mkstemp()
        _,rlog_path = tempfile.mkstemp()

        # converts train and test data to libSVM format
        dump_svmlight_file(x_train, y_train, train_path)
        dump_svmlight_file(x_test, y_test, test_path)

        # builds arguments array
        args = [self.__libfm_path,
                "-task %s" % self.__task,
                "-train %s" % train_path,
                "-test %s" % test_path,
                "-dim '%s'" % self.__dim,
                "-init_stdev %g" % self.__init_stdev,
                "-iter %d" % self.__num_iter,
                "-method %s" % self.__learning_method,
                "-out %s" % out_path,
                "-verbosity %d" % self.__verbose,
                "-save_model %s" % model_path,
                "-rlog %s" % rlog_path]

        # appends arguments that only work for certain learning methods
        if self.__learning_method in ['sgd', 'sgda']:
            args.append("-learn_rate %d" % self.__learn_rate)

        if self.__learning_method in ['sgd', 'sgda', 'als']:
            args.append("-regular '%s'" % self.__regularization)

        # call libfm with parsed arguments
        # unkown bug with "-dim" option on array -- forced to concatenate string
        args = ' '.join(args)
        call(args, shell=True)

        # reads output file
        preds = []
        with open(out_path, 'r') as out_file:
            out_read = out_file.read()
            preds = [float(p) for p in out_read.split('\n') if p]

        # "hidden" feature that allows users to save the model
        # We use this to get the feature weights
        # https://github.com/srendle/libfm/commit/19db0d1e36490290dadb530a56a5ae314b68da5d
        num_features = x_train.shape[1]
        import numpy as np
        model_file = open(model_path).read()
        model_enum = enumerate(model_file.split('\n'))
        global_bias = weights = pairwise_interactions = None
        for num, line in model_enum:
            if "#global bias W0" in line:
                # gets next item value
                global_bias = float(next(model_enum)[1])
            elif "#unary interactions Wj" in line:
                weights = [float(next(model_enum)[1]) for w in range(num_features-1)]
            elif "#pairwise interactions Vj,f" in line:
                pairwise_interactions = np.matrix([float(x) for x in next(model_enum)[1].split(' ') for w in range(num_features-1)])

        # parses rlog into
        import pandas as pd
        rlog = pd.read_csv(rlog_path, sep='\t')

        # removes temporary output file after using
        os.remove(train_path)
        os.remove(test_path)
        os.remove(out_path)
        os.remove(model_path)
        os.remove(rlog_path)

        # return as named collection for multiple output
        import collections
        fm = collections.namedtuple('model', ['predictions',
                                              'global_bias',
                                              'weights',
                                              'pairwise_interactions',
                                              'rlog'])
        return fm(preds, global_bias, weights, pairwise_interactions, rlog)
