from nltk.corpus import brown
from nltk import FreqDist, WittenBellProbDist
from nltk import ngrams, bigrams
from nltk.parse import viterbi
from nltk.metrics import ConfusionMatrix, accuracy

import Viterbi as vit


def measure_performance_single_sentence(test_sent_tags_actual, test_sent_tags_est):
    if (len(test_sent_tags_actual) != len(test_sent_tags_est)):
        print("ERROR: SIZE IS NOT THE SAME")
        print("test actual:", test_sent_tags_actual)
        print("test est:", test_sent_tags_est)
        return 10 ** 10

    sum_error = 0
    for i in range(0, len(test_sent_tags_est)):
        if (test_sent_tags_est[i] != test_sent_tags_actual[i]):
            sum_error += 1

    return sum_error


def measure_performance_all_sentence(test_sent_tags_actual, test_sent_tags_est, unique_tag_dict):
    overall_error = 0
    overall_N = 0
    overall_large_error_count = 0
    overall_large_error_index = []
    LARGE_ERROR_THRESHOLD = 0.70

    for unique_tag in unique_tag_dict.keys():
        # we wish to understand three features [tag chosen, actual tag, size sentence]
        unique_tag_dict[unique_tag] = dict.fromkeys(unique_tag_dict.keys())

    total_num_est_sent = len(test_sent_tags_est)
    total_num_act_sent = len(test_sent_tags_actual)

    if (total_num_act_sent != total_num_est_sent):
        print("ERROR: DIFFERENT NUMBER OF SENTENCES IN TEST AND ACTUAL")
        print("test actual:", test_sent_tags_actual)
        print("test est:", test_sent_tags_est)
        return 0
    else:
        for i in range(total_num_act_sent):
            length_sent_actual = len(test_sent_tags_actual[i][1:-1])
            error_for_sent = measure_performance_single_sentence(test_sent_tags_actual[i][1:-1], test_sent_tags_est[i])
            # print("sentence ", i)

            if (LARGE_ERROR_THRESHOLD > 1.0 - (error_for_sent * 1.0 / length_sent_actual)):
                # print("---------------")
                # print("error is ", error_for_sent)
                # print("out of ", length_sent_actual)
                # print("Score for sentence i", i, (1.0 - (error_for_sent * 1.0 / length_sent_actual)))
                # print("actual")
                # print(test_sent_tags_actual[i][1:-1])
                # print("estimated")
                # print(test_sent_tags_est[i])
                # print("LARGE ERROR")
                overall_large_error_count += 1
                overall_large_error_index.append(i)

            overall_error += error_for_sent
            overall_N += length_sent_actual

        print("Overall error")
        print(1.0 - overall_error * 1.0 / overall_N)
        print("Overall large error freq")
        print(overall_large_error_count * 1.0 / overall_N)

    return overall_error, overall_N, overall_large_error_index;


def large_error_analysis(overall_large_error_index, all_test_tags_actual, all_test_sent_tags_est):
    # for i in range(len(overall_large_error_index)):
    print()


def get_confusion_matrix_for_sent_size(test_sent_tags_actual, test_sent_tags_est, min_sent_length=100):
    single_line_act_tags = []
    single_line_est_tags = []

    for sent_index in range(len(test_sent_tags_actual)):
        # print("len(test_sent_tags_actual[sent_index])", len(test_sent_tags_actual[sent_index]))
        if (len(test_sent_tags_actual[sent_index]) >= min_sent_length):
            single_line_act_tags = single_line_act_tags + test_sent_tags_actual[sent_index][1:-1]
            single_line_est_tags = single_line_est_tags + test_sent_tags_est[sent_index]

    # print("CM single_line_est_tags ", single_line_est_tags)
    # print("CM single_line_act_tags", single_line_act_tags)

    cm = ConfusionMatrix(single_line_act_tags, single_line_est_tags)

    print("accuracy is ", accuracy(single_line_act_tags, single_line_est_tags))
    print(cm.pretty_format(sort_by_count=True, show_percents=True, truncate=12))



def compare_two_approaches(test_sent_tags_actual, test_sent_tags_est_1, est_1_name, test_sent_tags_est_2, est_2_name):
    for sent_index in range(len(test_sent_tags_actual)):

        different_flag = False
        for tag_index in range(len(test_sent_tags_est_1[sent_index])):
            if (test_sent_tags_est_1[sent_index][tag_index] != test_sent_tags_est_2[sent_index][tag_index]) and (
                    test_sent_tags_actual[sent_index][tag_index + 1] == test_sent_tags_est_2[sent_index][tag_index]):
                different_flag = True

        if (different_flag):
            print ("Sent index", sent_index)
            print (est_1_name)
            print (test_sent_tags_est_1[sent_index])
            print (est_2_name)
            print (test_sent_tags_est_2[sent_index])
            print ("actual")
            print (test_sent_tags_actual[sent_index])


def compare_accuracy_for_sentence_size(test_sent_tags_actual, test_sent_tags_est, min_sent_length, min_tag_diversity, name):
    single_line_act_tags = []
    single_line_est_tags = []

    non_zero_demominator = False

    for sent_index in range(len(test_sent_tags_actual)):
        tag_diversity = set(test_sent_tags_actual[sent_index])
        if (len(test_sent_tags_actual[sent_index]) >= min_sent_length) and tag_diversity >= min_tag_diversity:
            single_line_act_tags = single_line_act_tags + test_sent_tags_actual[sent_index][1:-1]
            single_line_est_tags = single_line_est_tags + test_sent_tags_est[sent_index]
            non_zero_demominator = True

    if (non_zero_demominator):
        print(name, accuracy(single_line_act_tags, single_line_est_tags))
