from nltk.corpus import brown
from nltk import FreqDist, WittenBellProbDist
from nltk import ngrams, bigrams
from nltk.parse import viterbi

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


def measure_performance_all_sentence(test_sent_tags_actual, test_sent_tags_est):
    overall_error = 0
    overall_N = 0
    overall_large_error_count = 0
    overall_large_error_index = []
    LARGE_ERROR_THRESHOLD = 0.70


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
            print("sentence ",i)

            if(LARGE_ERROR_THRESHOLD > 1.0 - (error_for_sent * 1.0 / length_sent_actual)):
                print("---------------")
                print("error is ", error_for_sent)
                print("out of ", length_sent_actual)
                print("Score for sentence i", i, (1.0 - (error_for_sent * 1.0 / length_sent_actual)))
                print("actual")
                print(test_sent_tags_actual[i][1:-1])
                print("estimated")
                print(test_sent_tags_est[i])
                print("LARGE ERROR")
                overall_large_error_count += 1
                overall_large_error_index.append(i)

            overall_error += error_for_sent
            overall_N += length_sent_actual

        print("Overall error")
        print(1.0 - overall_error * 1.0 / overall_N)
        print("Overall large error freq")
        print(overall_large_error_count * 1.0 / overall_N)

    return overall_error, overall_N, overall_large_error_index;

def large_error_analysis():