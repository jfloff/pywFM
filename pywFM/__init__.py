from subprocess import call
import os
import tempfile

class FM:
    """ libFM wrapper. For more information regarding the parameters read
    libFM manual at http://www.libfm.org/libfm-1.42.manual.pdf

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
    rlog: write measurements within iterations to a file; default=''
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
        self.task = task[0]
        self.num_iter = num_iter
        self.init_stdev = init_stdev
        self.dim = "%d,%d,%d" % (int(k0), int(k1), k2)
        self.learning_method = learning_method
        self.learn_rate = learn_rate
        self.regularization = "%d,%d,%d" % (r0_regularization, r1_regularization, r2_regularization)
        self.verbose = int(verbose)

        # gets real path of package
        self.libfm_path=os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                     "libfm/bin/libFM")

    def predict(self, x_train, y_train, x_test, y_test):
        """Predict using the factorization machine

        Parameters
        ----------
        x_train : {array-like, matrix}, shape = [n_samples, n_features]
            Training data
        y_train : numpy array of shape [n_samples]
            Target values
        x_test: {array-like, matrix}, shape = [n_samples, n_features]
            Testing data

        Returns
        -------
        array, shape = [n_samples of x_test]
           Predicted target values per element in x_test.
        """

        from sklearn.datasets import dump_svmlight_file

        _,train_path = tempfile.mkstemp()
        _,test_path = tempfile.mkstemp()
        _,out_path = tempfile.mkstemp()
        _,model_path = tempfile.mkstemp()
        # out_path = "~/home/yelp"
        # model_path = "~/home/yelp"

        # converts train and test data to libSVM format
        dump_svmlight_file(x_train, y_train, train_path)
        dump_svmlight_file(x_test, y_test, test_path)

        # builds arguments array
        args = [self.libfm_path,
                "-task %s" % self.task,
                "-train %s" % train_path,
                "-test %s" % test_path,
                "-dim '%s'" % self.dim,
                "-init_stdev %g" % self.init_stdev,
                "-iter %d" % self.num_iter,
                "-method %s" % self.learning_method,
                "-out %s" % out_path,
                "-verbosity %d" % self.verbose,
                "-save_model %s" % model_path]

        # appends arguments that only work for certain learning methods
        if self.learning_method in ['sgd', 'sgda']:
            args.append("-learn_rate %d" % self.learn_rate)

        if self.learning_method in ['sgd', 'sgda', 'als']:
            args.append("-regular '%s'" % self.regularization)

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
        import itertools
        with open(model_path, 'r') as model_file:
            self.weights = [float(w) for w in itertools.islice(model_file, 3, num_features-1)]

        # removes temporary output file after using
        os.remove(train_path)
        os.remove(test_path)
        os.remove(out_path)
        os.remove(model_path)

        return preds
