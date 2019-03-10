from nltk.corpus import brown
from nltk import FreqDist, WittenBellProbDist
from nltk import ngrams, bigrams
from nltk import accuracy
from math import log, exp
import time

import Viterbi as vit
import ErrorAnalysis as error
import BeamSearch
import BeamClass
import ForwardProbabilities
import BackwardProbabilities



def tag_all_sentences_beam(test_sents, freq_dist_tag_single, smoothed_tag_tag, smoothed_word_tag, beam_width):

    """Tag all sentences using Beam search. Return results"""

    tag_estimated_by_beam = []

    # Create appropriate beam search objects
    beam_search_vit = BeamSearch.BeamSearch(freq_dist_tag_single, smoothed_tag_tag, smoothed_word_tag, beam_width)

    # tag each sentence
    for test_sent in test_sents:
        tag_sequence = beam_search_vit.viterbi_path(test_sent)
        tag_estimated_by_beam.append(tag_sequence)

    return tag_estimated_by_beam


def tag_all_sentences_forward_Back(test_sents, freq_dist_tag_single, smoothed_tag_tag, smoothed_word_tag):

    """Tag all sentences using Beam search. Return results"""

    tag_estimated_by_fwd_bwd = []

    # Create appropriate forward, backward objects
    forward_prob_calculator = ForwardProbabilities.ForwardProbabilities(freq_dist_tag_single, smoothed_tag_tag,
                                                                        smoothed_word_tag)
    backward_prob_calculator = BackwardProbabilities.BackwardProbabilities(freq_dist_tag_single, smoothed_tag_tag,
                                                                           smoothed_word_tag)


    non_beg_or_end_states = list(freq_dist_tag_single.keys())
    non_beg_or_end_states.remove('<s>')
    non_beg_or_end_states.remove('</s>')

    # tag each sentence
    for test_sent in test_sents:
        tag_sequence = forward_and_backward_combined(test_sent, forward_prob_calculator, backward_prob_calculator,
                                                     non_beg_or_end_states)
        tag_estimated_by_fwd_bwd.append(tag_sequence)

    return tag_estimated_by_fwd_bwd


def forward_and_backward_combined(sent, forward_prob_calculator, backward_prob_calculator, non_beg_or_end_states):
    """Takes Forward Prob and Backward objects and returns an estimated list of tags for a single sentence. Note,
    all probabilities are in log form """

    # obtain alpha, beta matrices for the sentence
    f_prob, alpha = forward_prob_calculator.return_forward_prob(sent)
    b_prob, beta = backward_prob_calculator.return_backward_prob(sent)

    # need to reverse the sentence (because it was previously reversed in the backward prob object)
    sent.reverse()

    tag_estimated_by_forward_backward = []

    # cycle through each word of the sentence (ignoring start/end tokens)
    sentence_length = len(sent)
    for word_index in range(1, sentence_length - 1):

        # calculates the appropriate indices for aplha and beta matrices (recall that beta matrix is backwards)
        word_index_alpha = word_index
        word_index_beta = (sentence_length - 1) - word_index_alpha

        max_prob = -10 ** 10
        max_prob_state = None

        # for each token, cycle through each state and calculate the probability of it occurring using forward and backward probs
        for tested_state_index in range(0, len(non_beg_or_end_states)):

            # forward log prob to that token, that that particular state
            forward_prob_to_state = alpha[word_index_alpha][tested_state_index]

            # backward log prob to that token, that that particular state
            back_prob_to_state = beta[word_index_beta][tested_state_index]

            # calculate prob estimate
            # note, entries in alpha and beta matrices both include an emission prob. To avoid double counting this prob,
            # and adhere to the forward-backward formula, we need to remove it once.
            prob_est = forward_prob_to_state + back_prob_to_state - smoothed_word_tag[
                non_beg_or_end_states[tested_state_index]].prob(sent[word_index_alpha])

            # find the state that maximises prob
            if (max_prob < prob_est):
                max_prob = prob_est
                max_prob_state = non_beg_or_end_states[tested_state_index]

        # construct estimated tag sequence with highest prob tags
        tag_estimated_by_forward_backward.append(max_prob_state)

    # return after going through entire sentence
    return tag_estimated_by_forward_backward



print("-------------- Intialise variables, preprocess data set-----------------")

# start and end symbols
start = '<s>'
end = '</s>'
max_sentence_length = 101
sents = brown.tagged_sents(tagset='universal')
train_size = 10000
test_size = 500

# add start and end states and tokens to each sentence. Making sure sentences are not too long (to avoid underflow)
sents = [[(start, start)] + sent + [(end, end)] for sent in sents[:train_size + test_size] if len(sent) < max_sentence_length]

# split into test and training sets
train_sents = sents[:train_size]
test_sents = sents[train_size:train_size + test_size]

# Initialise variables
test_words_individual = []
test_tags_individual = []
freq_dist_tag_single = FreqDist()
bigrams_tags = []

print("-------------- Processing training set ---------------------------------")
# (1): we need a list of unique states/tags
for sent in train_sents:

    # split tag lists
    tag_list = [t for (_, t) in sent]
    freq_dist_tag_single += FreqDist(tag_list)
    bigrams_tags.append(bigrams(tag_list))


# (2): create a unique list of states
unique_tag_list = list(freq_dist_tag_single.keys())
unique_tag_list.remove(start)
unique_tag_list.remove(end)
un_smoothed = {}
smoothed_word_tag = {}
tags = set(unique_tag_list)

# (3): For each sentence count the word frequency for each tag
for sent in train_sents:
    for tag in tags:
        words = [w for (w, t) in sent if t == tag]

        if (not un_smoothed.has_key(tag)):
            un_smoothed[tag] = FreqDist(words)
        else:
            un_smoothed[tag].update(words)

# (4): For each tag, find the smoothed probabilities using WittenBell
for tag in tags:
    smoothed_word_tag[tag] = WittenBellProbDist(un_smoothed[tag], bins=1e5)


# (5): repeat step (3) but for pairs of consecutive tags
un_smoothed = {}
smoothed_tag_tag = {}
tags.add(start)

# (6): for each tag bigram, count the frequency
for bigram_list in bigrams_tags:

    bigram_sent = list(bigram_list)
    for tag in tags:

        following_tag = [t2 for (t1, t2) in bigram_sent if t1 == tag]

        if (not un_smoothed.has_key(tag)):
            un_smoothed[tag] = FreqDist(following_tag)
        else:
            un_smoothed[tag].update(following_tag)


# (7): For each tag bigram, find the smoothed probabilities using WittenBell
for tag in tags:
    smoothed_tag_tag[tag] = WittenBellProbDist(un_smoothed[tag], bins=1e5)


print("-------------- Processing test set -------------------------------------")
# split test into tokens and states
for sent in test_sents:
    word_list = [w for (w, _) in sent]
    tag_list = [t for (_, t) in sent]

    test_words_individual.append(word_list)
    test_tags_individual.append(tag_list)


test_words_to_tag = test_words_individual
test_sent_tags_act = test_tags_individual


print("-------------- Performing tagging on test set using various methods ----")


time_start = time.time()
all_test_sent_tags_est_beam_all = tag_all_sentences_beam(test_words_to_tag, freq_dist_tag_single,
                                                         smoothed_tag_tag, smoothed_word_tag,
                                                         len(list(freq_dist_tag_single.keys())) - 2)
time_end = time.time()
beam_all_time = time_end - time_start

time_start = time.time()
all_test_sent_tags_est_beam_one = tag_all_sentences_beam(test_words_to_tag, freq_dist_tag_single,
                                                         smoothed_tag_tag, smoothed_word_tag, 1)
time_end = time.time()
beam_one_time = time_end - time_start

time_start = time.time()
all_test_sent_tags_est_beam_three = tag_all_sentences_beam(test_words_to_tag, freq_dist_tag_single,
                                                           smoothed_tag_tag, smoothed_word_tag, 3)
time_end = time.time()
beam_three_time = time_end - time_start

time_start = time.time()
all_test_sent_tags_est_beam_five = tag_all_sentences_beam(test_words_to_tag, freq_dist_tag_single,
                                                          smoothed_tag_tag, smoothed_word_tag, 5)
time_end = time.time()
beam_five_time = time_end - time_start

time_start = time.time()
all_test_sent_tags_est_beam_seven = tag_all_sentences_beam(test_words_to_tag, freq_dist_tag_single,
                                                           smoothed_tag_tag, smoothed_word_tag, 7)
time_end = time.time()
beam_seven_time = time_end - time_start

time_start = time.time()
all_test_sent_tags_est_beam_nine = tag_all_sentences_beam(test_words_to_tag, freq_dist_tag_single,
                                                          smoothed_tag_tag, smoothed_word_tag, 9)
time_end = time.time()
beam_nine_time = time_end - time_start

time_start = time.time()
all_test_sent_tags_est_beam_eleven = tag_all_sentences_beam(test_words_to_tag, freq_dist_tag_single,
                                                            smoothed_tag_tag, smoothed_word_tag, 11)
time_end = time.time()
beam_eleven_eleven = time_end - time_start

time_start = time.time()
all_test_sent_tags_est_forward_back = tag_all_sentences_forward_Back(test_words_to_tag, freq_dist_tag_single,
                                                                     smoothed_tag_tag, smoothed_word_tag)
time_end = time.time()
ford_back_time = time_end - time_start


print("-------------- Perform analysis on results-----------------------------")

print("")
print("----- Testing Time to complete-------")
print("")
min_sentence_size = 0
min_tag_diversity = 0
print("train_size ", train_size)
print("test_size ", test_size)
error.compare_accuracy(test_sent_tags_act, all_test_sent_tags_est_beam_all, min_sentence_size,min_tag_diversity,
                                         "all_test_sent_tags_est_beam_all" + "," + str(min_sentence_size)+ "," + str(min_tag_diversity) + "," + str(
                                             beam_all_time))
error.compare_accuracy(test_sent_tags_act, all_test_sent_tags_est_beam_one, min_sentence_size,min_tag_diversity,
                                         "all_test_sent_tags_est_beam_one" + "," + str(min_sentence_size)+ "," + str(min_tag_diversity) + "," + str(
                                             beam_one_time))
error.compare_accuracy(test_sent_tags_act, all_test_sent_tags_est_beam_three, min_sentence_size,min_tag_diversity,
                                         "all_test_sent_tags_est_beam_three" + "," + str(min_sentence_size)+ "," + str(min_tag_diversity) + "," + str(
                                             beam_three_time))
error.compare_accuracy(test_sent_tags_act, all_test_sent_tags_est_beam_five, min_sentence_size,min_tag_diversity,
                                         "all_test_sent_tags_est_beam_five" + "," + str(min_sentence_size)+ "," + str(min_tag_diversity) + "," + str(
                                             beam_five_time))
error.compare_accuracy(test_sent_tags_act, all_test_sent_tags_est_beam_seven, min_sentence_size,min_tag_diversity,
                                         "all_test_sent_tags_est_beam_seven" + "," + str(min_sentence_size)+ "," + str(min_tag_diversity) + "," + str(
                                             beam_seven_time))
error.compare_accuracy(test_sent_tags_act, all_test_sent_tags_est_beam_nine, min_sentence_size,min_tag_diversity,
                                         "all_test_sent_tags_est_beam_nine" + "," + str(min_sentence_size)+ "," + str(min_tag_diversity) + "," + str(
                                             beam_nine_time))
error.compare_accuracy(test_sent_tags_act, all_test_sent_tags_est_beam_eleven, min_sentence_size,min_tag_diversity,
                                         "all_test_sent_tags_est_beam_eleven" + "," + str(
                                             min_sentence_size)+ "," + str(min_tag_diversity) + "," + str(beam_eleven_eleven))
error.compare_accuracy(test_sent_tags_act, all_test_sent_tags_est_forward_back, min_sentence_size,min_tag_diversity,
                                         "all_test_sent_tags_est_forward_back" + "," + str(
                                             min_sentence_size)+ "," + str(min_tag_diversity) + "," + str(ford_back_time))

print("")
print("----- Testing accuracy versus sentence size -----------------------")
print("")
print("train_size ", train_size)
print("test_size ", test_size)
min_tag_diversity = 0

for min_sentence_size in range(0, 100, 10):
    error.compare_accuracy_for_sentence_size(test_sent_tags_act, all_test_sent_tags_est_beam_one, min_sentence_size,min_tag_diversity,
                                             "all_test_sent_tags_est_beam_one" + "," + str(min_sentence_size)+ "," + str(min_tag_diversity))
    error.compare_accuracy_for_sentence_size(test_sent_tags_act, all_test_sent_tags_est_beam_three, min_sentence_size,min_tag_diversity,
                                             "all_test_sent_tags_est_beam_three" + "," + str(min_sentence_size)+ "," + str(min_tag_diversity))
    error.compare_accuracy_for_sentence_size(test_sent_tags_act, all_test_sent_tags_est_beam_five, min_sentence_size,min_tag_diversity,
                                             "all_test_sent_tags_est_beam_five" + "," + str(min_sentence_size)+ "," + str(min_tag_diversity))
    error.compare_accuracy_for_sentence_size(test_sent_tags_act, all_test_sent_tags_est_beam_seven, min_sentence_size,min_tag_diversity,
                                             "all_test_sent_tags_est_beam_seven" + "," + str(min_sentence_size)+ "," + str(min_tag_diversity))
    error.compare_accuracy_for_sentence_size(test_sent_tags_act, all_test_sent_tags_est_beam_nine, min_sentence_size,min_tag_diversity,
                                             "all_test_sent_tags_est_beam_nine" + "," + str(min_sentence_size)+ "," + str(min_tag_diversity))
    error.compare_accuracy_for_sentence_size(test_sent_tags_act, all_test_sent_tags_est_beam_eleven, min_sentence_size,min_tag_diversity,
                                             "all_test_sent_tags_est_beam_eleven" + "," + str(min_sentence_size)+ "," + str(min_tag_diversity))
    error.compare_accuracy_for_sentence_size(test_sent_tags_act, all_test_sent_tags_est_beam_all, min_sentence_size,min_tag_diversity,
                                             "all_test_sent_tags_est_beam_all" + "," + str(min_sentence_size)+ "," + str(min_tag_diversity))
print("")
print("----- Testing accuracy versus tag diversity -----------------------")
print("")
print("\ntrain_size ", train_size)
print("test_size ", test_size)
min_sentence_size = 0

for min_tag_diversity in range(1, 13):
    error.compare_accuracy_for_tag_diversity(test_sent_tags_act, all_test_sent_tags_est_beam_one, min_sentence_size,min_tag_diversity,
                                             "all_test_sent_tags_est_beam_one" + "," + str(min_sentence_size)+ "," + str(min_tag_diversity))
    error.compare_accuracy_for_tag_diversity(test_sent_tags_act, all_test_sent_tags_est_beam_three, min_sentence_size,min_tag_diversity,
                                             "all_test_sent_tags_est_beam_three" + "," + str(min_sentence_size)+ "," + str(min_tag_diversity))
    error.compare_accuracy_for_tag_diversity(test_sent_tags_act, all_test_sent_tags_est_beam_five, min_sentence_size,min_tag_diversity,
                                             "all_test_sent_tags_est_beam_five" + "," + str(min_sentence_size)+ "," + str(min_tag_diversity))
    error.compare_accuracy_for_tag_diversity(test_sent_tags_act, all_test_sent_tags_est_beam_seven, min_sentence_size,min_tag_diversity,
                                             "all_test_sent_tags_est_beam_seven" + "," + str(min_sentence_size)+ "," + str(min_tag_diversity))
    error.compare_accuracy_for_tag_diversity(test_sent_tags_act, all_test_sent_tags_est_beam_nine, min_sentence_size,min_tag_diversity,
                                             "all_test_sent_tags_est_beam_nine" + "," + str(min_sentence_size)+ "," + str(min_tag_diversity))
    error.compare_accuracy_for_tag_diversity(test_sent_tags_act, all_test_sent_tags_est_beam_eleven, min_sentence_size,min_tag_diversity,
                                             "all_test_sent_tags_est_beam_eleven" + "," + str(min_sentence_size)+ "," + str(min_tag_diversity))
    error.compare_accuracy_for_tag_diversity(test_sent_tags_act, all_test_sent_tags_est_beam_all, min_sentence_size,min_tag_diversity,
                                             "all_test_sent_tags_est_beam_all" + "," + str(min_sentence_size)+ "," + str(min_tag_diversity))

print("")
print("----- Overall Confusion Matrices and Accuracy-----------------------")
print("")
print("Beam one:")
error.get_confusion_matrix_for_sent_size(test_sent_tags_act, all_test_sent_tags_est_beam_one, 0)
print("Beam three:")
error.get_confusion_matrix_for_sent_size(test_sent_tags_act, all_test_sent_tags_est_beam_three, 0)
print("Beam five:")
error.get_confusion_matrix_for_sent_size(test_sent_tags_act, all_test_sent_tags_est_beam_five, 0)
print("Beam seven:")
error.get_confusion_matrix_for_sent_size(test_sent_tags_act, all_test_sent_tags_est_beam_seven, 0)
print("Beam nine:")
error.get_confusion_matrix_for_sent_size(test_sent_tags_act, all_test_sent_tags_est_beam_nine, 0)
print("Beam eleven:")
error.get_confusion_matrix_for_sent_size(test_sent_tags_act, all_test_sent_tags_est_beam_eleven, 0)
print("Beam all:")
error.get_confusion_matrix_for_sent_size(test_sent_tags_act, all_test_sent_tags_est_beam_all, 0)
print("forward, backward:")
error.get_confusion_matrix_for_sent_size(test_sent_tags_act, all_test_sent_tags_est_forward_back, 0)
