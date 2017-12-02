import subprocess
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
    rlog: bool, optional
        Enable/disable rlog output
        Defaults to True.
    verbose: bool, optional
        How much infos to print
        Defaults to False.
    seed: int, optional
        seed used to reproduce the results
        Defaults to None.
    silent: bool, optional
        Completly silences all libFM output
        Defaults to False.
    temp_path: string, optional
        Sets path for libFM temporary files. Usefull when dealing with large data.
        Defaults to None (default NamedTemporaryFile behaviour)
    """

    """
    ### unsused libFM flags
    cache_size: cache size for data storage (only applicable if data is in binary format), default=infty
        datafile is text so we don't need this parameter
    relation: BS - filenames for the relations, default=''
        not dealing with BS extensions since they are only used for binary files
    """

    def __init__(self,
                 task,
                 num_iter=100,
                 init_stdev=0.1,
                 k0=True,
                 k1=True,
                 k2=8,
                 learning_method='mcmc',
                 learn_rate=0.1,
                 r0_regularization=0,
                 r1_regularization=0,
                 r2_regularization=0,
                 rlog=True,
                 verbose=False,
                 seed=None,
                 silent=False,
                 temp_path=None):

        # gets first letter of either regression or classification
        self.__task = task[0]
        self.__num_iter = num_iter
        self.__init_stdev = init_stdev
        self.__dim = "%d,%d,%d" % (int(k0), int(k1), k2)
        self.__learning_method = learning_method
        self.__learn_rate = learn_rate
        self.__regularization = "%.5f,%.5f,%.5f" % (r0_regularization, r1_regularization, r2_regularization)
        self.__rlog = rlog
        self.__verbose = int(verbose)
        self.__seed = int(seed) if seed else None
        self.__silent = silent
        self.__temp_path = temp_path

        # gets libfm path
        self.__libfm_path = os.path.join(os.environ.get('LIBFM_PATH'), "")
        if self.__libfm_path is None:
            raise OSError("`LIBFM_PATH` is not set. Please install libFM and set the path variable "
                          "(https://github.com/jfloff/pywFM#installing).")

        # #ShameShame
        # Once upon a time, there was a bug in libFM that allowed any type of
        # learning_method to save the model. I @jfloff built this package at
        # that time, and did not find anything that showed me that MCMC couldn't
        # use save_model flag. Nowadays only SGD and ALS can use this parameter.
        # Hence, we need to reset the repo to this specific commit pre-fix, so
        # we can use MCMC with save_model flag.
        # Can we contribute to main libFM repo so this is possible again??
        GITHASH = '91f8504a15120ef6815d6e10cc7dee42eebaab0f'
        c_githash = subprocess.check_output(['git', '--git-dir', os.path.join(self.__libfm_path, "..", ".git"), 'rev-parse', 'HEAD']).strip()
        if c_githash.decode("utf-8") != GITHASH:
            raise OSError("libFM is not checked out to the correct commit."
                          "(https://github.com/jfloff/pywFM#installing).")

    def run(self, x_train, y_train, x_test, y_test, x_validation_set=None, y_validation_set=None, meta=None):
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
        x_validation_set: optional, {array-like, matrix}, shape = [n_train, n_features]
            Validation data (only for SGDA)
        y_validation_set: optional, numpy array of shape [n_train]
            Validation target data (only for SGDA)
        meta: optional, numpy array of shape [n_features]
            Grouping input variables

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

        TMP_SUFFIX = '.pywfm'
        train_fd = tempfile.NamedTemporaryFile(suffix=TMP_SUFFIX, dir=self.__temp_path)
        test_fd = tempfile.NamedTemporaryFile(suffix=TMP_SUFFIX, dir=self.__temp_path)
        out_fd = tempfile.NamedTemporaryFile(suffix=TMP_SUFFIX, dir=self.__temp_path)
        model_fd = tempfile.NamedTemporaryFile(suffix=TMP_SUFFIX, dir=self.__temp_path)

        # converts train and test data to libSVM format
        dump_svmlight_file(x_train, y_train, train_fd)
        train_fd.seek(0)
        dump_svmlight_file(x_test, y_test, test_fd)
        test_fd.seek(0)

        # builds arguments array
        args = [os.path.join(self.__libfm_path, "libFM"),
                '-task', "%s" % self.__task,
                '-train', "%s" % train_fd.name,
                '-test', "%s" % test_fd.name,
                '-dim', "'%s'" % self.__dim,
                '-init_stdev', "%g" % self.__init_stdev,
                '-iter', "%d" % self.__num_iter,
                '-method', "%s" % self.__learning_method,
                '-out', "%s" % out_fd.name,
                '-verbosity', "%d" % self.__verbose,
                '-save_model', "%s" % model_fd.name]

        # appends rlog if true
        rlog_fd = None
        if self.__rlog:
            rlog_fd = tempfile.NamedTemporaryFile(suffix=TMP_SUFFIX, dir=self.__temp_path)
            args.extend(['-rlog', "%s" % rlog_fd.name])

        # appends seed if given
        if self.__seed:
            args.extend(['-seed', "%d" % self.__seed])

        # appends arguments that only work for certain learning methods
        if self.__learning_method in ['sgd', 'sgda']:
            args.extend(['-learn_rate', "%.5f" % self.__learn_rate])

        if self.__learning_method in ['sgd', 'sgda', 'als']:
            args.extend(['-regular', "'%s'" % self.__regularization])

        # adds validation if sgda
        # if validation_set is none, libFM will throw error hence, I'm not doing any validation
        validation_fd = None
        if self.__learning_method == 'sgda' and (x_validation_set is not None and y_validation_set is not None):
            validation_fd = tempfile.NamedTemporaryFile(suffix=TMP_SUFFIX, dir=self.__temp_path)
            dump_svmlight_file(x_validation_set, y_validation_set, validation_fd.name)
            args.extend(['-validation', "%s" % validation_fd.name])

        # if meta data is given
        meta_fd = None
        if meta is not None:
            meta_fd = tempfile.NamedTemporaryFile(suffix=TMP_SUFFIX, dir=self.__temp_path, text=True)
            # write group ids
            for group_id in meta:
                meta_fd.write("%s\n" % group_id)
            args.extend(['-meta', "%s" % meta_fd.name])
            meta_fd.seek(0)

        # if silent redirects all output
        stdout = None
        if self.__silent:
            stdout = open(os.devnull, 'wb')

        # call libfm with parsed arguments
        # had unkown bug with "-dim" option on array. At the time was forced to
        # concatenate string `args = ' '.join(args)` but looks like its working
        # needs further tests
        subprocess.call(args, shell=False, stdout=stdout)

        # reads output file
        preds = [float(p) for p in out_fd.read().decode("utf-8").split('\n') if p]

        # "hidden" feature that allows users to save the model
        # We use this to get the feature weights
        # https://github.com/srendle/libfm/commit/19db0d1e36490290dadb530a56a5ae314b68da5d
        import numpy as np
        global_bias = None
        weights = []
        pairwise_interactions = []
        # if 0 its global bias; if 1, weights; if 2, pairwise interactions
        out_iter = 0
        for line in model_fd.read().decode("utf-8").splitlines():
            # checks which line is starting with #
            if line.startswith('#'):
                if "#global bias W0" in line:
                    out_iter = 0
                elif "#unary interactions Wj" in line:
                    out_iter = 1
                elif "#pairwise interactions Vj,f" in line:
                    out_iter = 2
            else:
                # check context get in previous step and adds accordingly
                if out_iter == 0:
                    global_bias = float(line)
                elif out_iter == 1:
                    weights.append(float(line))
                elif out_iter == 2:
                    try:
                        pairwise_interactions.append([float(x) for x in line.split(' ')])
                    except ValueError as e:
                        pairwise_interactions.append(0.0) #Case: no pairwise interactions used

        pairwise_interactions = np.matrix(pairwise_interactions)

        # parses rlog into dataframe
        if self.__rlog:
            # parses rlog into
            import pandas as pd
            rlog_fd.seek(0)
            print(os.stat(rlog_fd.name).st_size)
            rlog = pd.read_csv(rlog_fd.name, sep='\t')
            rlog_fd.close()
        else:
            rlog = None

        if self.__learning_method == 'sgda' and (x_validation_set is not None and y_validation_set is not None):
            validation_fd.close()
        if meta is not None:
            meta_fd.close()

        # removes temporary output file after using
        train_fd.close()
        test_fd.close()
        model_fd.close()
        out_fd.close()

        # return as named collection for multiple output
        import collections
        fm = collections.namedtuple('model', ['predictions',
                                              'global_bias',
                                              'weights',
                                              'pairwise_interactions',
                                              'rlog'])
        return fm(preds, global_bias, weights, pairwise_interactions, rlog)
