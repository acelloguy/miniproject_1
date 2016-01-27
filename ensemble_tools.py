'''
Helper functions for building an emsemble of models and ensemble selection
Kevin Yang 2016

Current call to CV: X_train, X_cv, Y_train, Y_cv = \
        cross_validation.train_test_split(X, Y, test_size=0.2,
                                          random_state=1)
'''

def train_model (X, Y, X_cv, Y_cv, X_test, model, args={}):
    '''
    Instantiates a model, trains it, and then
    writes 5 files:
    1. Appends the arguments to the end of 'models.txt'
    2. Writes 0/1 results on the cv set to n_cv_predictions.txt
    3. Writes probabilistic results on the cv set to
       n_cv_probabilities.txt
    4. Writes 0/1 results on the test set to
       n_test_predictions.txt
    5. Writes probabilistic results on the test set to
       n_test_probabilities.txt

    Returns:
        n (int): id number for the model
    '''
    clf = model(**args)
    clf.fit(X, Y)
    cv_predictions = clf.predict(X_cv)
    cv_probabilities = clf.predict_proba(X_cv)[:,0]
    test_predictions = clf.predict(X_test)
    test_probabilities = clf.predict_proba(X_test)[:,0]
    score = clf.score(X_cv, Y_cv)

    with open('models/models.txt', 'r') as f:
        # get n
        n = len(list(f))
    with open('models/models.txt', 'a') as f:
        # write model
        f.write(str(n))
        f.write(',')
        f.write(str(model))
        f.write(',')
        f.write(str(args))
        f.write(',')
        f.write(str(score))
        f.write('\n')
    write_results(n, 'cv', 'Prediction', cv_predictions)
    write_results(n, 'test', 'Prediction', test_predictions)
    write_results(n, 'cv', 'Probability', cv_probabilities)
    write_results(n, 'test', 'Probability', test_probabilities)

    return n

def write_results(n, name, pred_type, results):
    '''
    Writes the results to a file

    Parameters:
        n (int): id number for model
        name (string): 'cv' or 'test'
        pred_type (string): 'Prediction' or 'Probability'
        results (np.ndarray): list of predictions
    '''
    if pred_type not in set(['Prediction','Probability']):
        raise ValueError("Name must be 'Prediction' or 'Probability'")
    if name not in set(['cv','test']):
        raise ValueError("Name must be 'Prediction' or 'Probability'")
    filename = 'models/'+'_'.join([str(n), name.lower(), pred_type]) + '.txt'
    with open(filename, 'w') as f:
        f.write('Id,' + pred_type + '\n')
        for i, p in enumerate(results):
            f.write(str(i) + ',' + str(p)+'\n')

