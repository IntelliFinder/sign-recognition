import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    #go over each model for each word and compare this model to other words then you can check the probability that a particular word is the  one the person is signing
    
    #for each word in the dataset
    for word_id in range(len(test_set.wordlist)):
        logL = {}
        #get its X and lengths
        X, lengths = test_set.get_item_Xlengths( word_id )
        #for every trained model in the dataset
        for word, model in models.items():
            #train the model on this word
            try:
                X, lengths = test_set.get_item_Xlengths( word_id )
                #get log-likelihood
                logL[word] = model.score( X, lengths )
            except:
                logL[word] = float("-inf")
        probabilities.append( logL )
        guesses.append( max( [ key for key, val in logL.items() ]) )

    return probabilities, guesses
