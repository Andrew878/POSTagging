from nltk.corpus import brown
from nltk import FreqDist, WittenBellProbDist
from nltk import ngrams, bigrams
from nltk.parse import viterbi
from nltk.metrics import ConfusionMatrix, accuracy

import Viterbi as vit

""""Contains methods to assist with analysing results"""

def measure_performance_single_sentence(test_sent_tags_actual, test_sent_tags_est):

    """Measures performance on a sentence by sentence basis"""

    # first check if length is the same
    if (len(test_sent_tags_actual) != len(test_sent_tags_est)):
        print("ERROR: SIZE IS NOT THE SAME")
        print("test actual:", test_sent_tags_actual)
        print("test est:", test_sent_tags_est)
        return 10 ** 10

    # then check and count differences between estimated and actual
    sum_error = 0
    for i in range(0, len(test_sent_tags_est)):
        if (test_sent_tags_est[i] != test_sent_tags_actual[i]):
            sum_error += 1

    return sum_error


def measure_performance_all_sentence(test_sent_tags_actual, test_sent_tags_est, unique_tag_dict):

    """Measure how an entire test set performs (i.e. many sentences). The large error threshold tells us when a sentence is
    particularly poorly tagged"""

    overall_error = 0
    overall_N = 0
    overall_large_error_count = 0
    overall_large_error_index = []
    LARGE_ERROR_THRESHOLD = 0.70
    total_num_est_sent = len(test_sent_tags_est)
    total_num_act_sent = len(test_sent_tags_actual)

    # first check if length of test and actual data sets equal
    if (total_num_act_sent != total_num_est_sent):
        print("ERROR: DIFFERENT NUMBER OF SENTENCES IN TEST AND ACTUAL")
        print("test actual:", test_sent_tags_actual)
        print("test est:", test_sent_tags_est)
        return 0
    else:

        # cycle through sentences and measure performance
        for i in range(total_num_act_sent):
            length_sent_actual = len(test_sent_tags_actual[i][1:-1])
            error_for_sent = measure_performance_single_sentence(test_sent_tags_actual[i][1:-1], test_sent_tags_est[i])

            # check for very poor accuracy, and make a record
            if (LARGE_ERROR_THRESHOLD > 1.0 - (error_for_sent * 1.0 / length_sent_actual)):
                overall_large_error_count += 1
                overall_large_error_index.append(i)

            overall_error += error_for_sent
            overall_N += length_sent_actual

        print("Overall error")
        print(1.0 - overall_error * 1.0 / overall_N)
        print("Overall large error freq")
        print(overall_large_error_count * 1.0 / overall_N)

    return overall_error, overall_N, overall_large_error_index;



def get_confusion_matrix_for_sent_size(test_sent_tags_actual, test_sent_tags_est, min_sent_length=0):

    """Gets the confusion matrix for a test set. Can also only perform analytics for a minimum sentence length"""

    single_line_act_tags = []
    single_line_est_tags = []

    # cycle through sentences
    for sent_index in range(len(test_sent_tags_actual)):

        # min sentence size condition
        if (len(test_sent_tags_actual[sent_index]) >= min_sent_length):
            single_line_act_tags = single_line_act_tags + test_sent_tags_actual[sent_index][1:-1]
            single_line_est_tags = single_line_est_tags + test_sent_tags_est[sent_index]

    # collate and print results
    cm = ConfusionMatrix(single_line_act_tags, single_line_est_tags)
    print("accuracy is ", accuracy(single_line_act_tags, single_line_est_tags))
    print(cm.pretty_format(sort_by_count=True, show_percents=True, truncate=12))


def compare_two_approaches(test_sent_tags_actual, test_sent_tags_est_1, est_1_name, test_sent_tags_est_2, est_2_name):

    """Compares two different tagging methods. Looks for differences in how one correctly tagged a token,
    and how another incorrectly tagged a token. Used to try and isolate the instances where one method is
    performing poorly"""

    # cycle through sentences
    for sent_index in range(len(test_sent_tags_actual)):

        # assume no differences at start
        different_flag = False

        # check tag by tag
        for tag_index in range(len(test_sent_tags_est_1[sent_index])):

            # if they are different and tag set 2 is correct
            if (test_sent_tags_est_1[sent_index][tag_index] != test_sent_tags_est_2[sent_index][tag_index]) and (
                    test_sent_tags_actual[sent_index][tag_index + 1] == test_sent_tags_est_2[sent_index][tag_index]):
                different_flag = True

        # print sentence
        if (different_flag):
            print ("Sent index", sent_index)
            print (est_1_name)
            print (test_sent_tags_est_1[sent_index])
            print (est_2_name)
            print (test_sent_tags_est_2[sent_index])
            print ("actual")
            print (test_sent_tags_actual[sent_index])


def compare_accuracy_for_sentence_size(test_sent_tags_actual, test_sent_tags_est, min_sent_length, min_tag_diversity, name):

    """Computes accuracy for a est tagged test set. Does it for a particular sentence size """

    single_line_act_tags = []
    single_line_est_tags = []
    bucket_size = 10

    # to prevent NaN results
    non_zero_demominator = False

    # cycle through sentences
    for sent_index in range(len(test_sent_tags_actual)):

        # check number of unique tags in sentence
        tag_diversity = len(set(test_sent_tags_actual[sent_index]))

        # check desired sentence attributes (i.e. length, diversity)
        sentence_length = len(test_sent_tags_actual[sent_index])
        if (sentence_length >= min_sent_length and sentence_length < (min_sent_length+ bucket_size)): #
            single_line_act_tags = single_line_act_tags + test_sent_tags_actual[sent_index][1:-1]
            single_line_est_tags = single_line_est_tags + test_sent_tags_est[sent_index]
            non_zero_demominator = True

    # calculate accuracy for sentences that meet conditions
    if (non_zero_demominator):
        print(name, accuracy(single_line_act_tags, single_line_est_tags))

def compare_accuracy_for_tag_diversity(test_sent_tags_actual, test_sent_tags_est, min_sent_length, min_tag_diversity, name):

    """Computes accuracy for a est tagged test set. Does it for a particular number of unique POS states
    (labelled sentence diversity)"""

    single_line_act_tags = []
    single_line_est_tags = []

    # to prevent NaN results
    non_zero_demominator = False

    # cycle through sentences
    for sent_index in range(len(test_sent_tags_actual)):

        # check number of unique tags in sentence
        tag_diversity = len(set(test_sent_tags_actual[sent_index]))

        # check desired sentence attributes (i.e. length, diversity)
        if ((tag_diversity == min_tag_diversity)):
            single_line_act_tags = single_line_act_tags + test_sent_tags_actual[sent_index][1:-1]
            single_line_est_tags = single_line_est_tags + test_sent_tags_est[sent_index]
            non_zero_demominator = True

    # calculate accuracy for sentences that meet conditions
    if (non_zero_demominator):
        print(name, accuracy(single_line_act_tags, single_line_est_tags))


def compare_accuracy(test_sent_tags_actual, test_sent_tags_est, min_sent_length, min_tag_diversity, name):

    """Computes accuracy for a est tagged test set. """

    single_line_act_tags = []
    single_line_est_tags = []

    # to prevent NaN results
    non_zero_demominator = False

    # cycle through sentences
    for sent_index in range(len(test_sent_tags_actual)):

        single_line_act_tags = single_line_act_tags + test_sent_tags_actual[sent_index][1:-1]
        single_line_est_tags = single_line_est_tags + test_sent_tags_est[sent_index]
        non_zero_demominator = True

    # calculate accuracy for sentences that meet conditions
    if (non_zero_demominator):
        print(name, accuracy(single_line_act_tags, single_line_est_tags))