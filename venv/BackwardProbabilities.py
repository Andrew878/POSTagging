from math import log, exp
from nltk import FreqDist, WittenBellProbDist

from math import log, exp
from nltk import FreqDist, WittenBellProbDist


class BackwardProbabilities:

    def __init__(self, freq_dist_tag_single, smoothed_tag_tag, smoothed_word_tag):
        self.freq_dist_tag_single = freq_dist_tag_single
        self.smoothed_tag_tag = smoothed_tag_tag
        self.smoothed_word_tag = smoothed_word_tag
        self.start = '<s>'
        self.end = '</s>'

        self.non_beg_or_end_states = list(freq_dist_tag_single.keys())
        self.non_beg_or_end_states.remove(self.end)
        self.tag_to_index_dict = dict.fromkeys(self.non_beg_or_end_states)
        self.non_beg_or_end_states.remove(self.start)

    def return_prob_sum(self, exisiting_alpha, current_state, current_word,
                        previous_word_index, ):
        current_tag_word_pair = (current_word, current_state)
        prob_est = 0

        prob_current_word_given_state = log(self.smoothed_word_tag[current_state].prob(current_word))

        for previous_state_index in range(len(self.non_beg_or_end_states)):
            tag_pair_to_test = (self.non_beg_or_end_states[previous_state_index], current_state)

            # print("current_tag_word_pair", current_tag_word_pair)
            # print("tag_pair_to_test", tag_pair_to_test)
            # print("previous vit at location = ", previous_word_index, previous_state_index,
            #       exisiting_alpha[previous_word_index][previous_state_index])
            # print("tag transition prob = ",
            #       log(self.smoothed_tag_tag[current_state].prob(self.non_beg_or_end_states[previous_state_index])))

            prob_est += exp(exisiting_alpha[previous_word_index][previous_state_index] + log(
                self.smoothed_tag_tag[current_state].prob(
                    self.non_beg_or_end_states[previous_state_index])) + prob_current_word_given_state)

        # print("summed prob", log(prob_est))

        return log(prob_est)

    def final_return_prob_sum(self, exisiting_alpha, previous_word_index):

        prob_est = 0

        for previous_state_index in range(len(self.non_beg_or_end_states)):
            tag_pair_to_test = (self.non_beg_or_end_states[previous_state_index], self.start)

            # print(tag_pair_to_test)
            # print("previous vit at location = ", previous_word_index, previous_state_index,
            #       exisiting_viterbi[previous_word_index][previous_state_index])

            prob_est += exp(exisiting_alpha[previous_word_index][previous_state_index] + log(
                self.smoothed_tag_tag[self.start].prob(self.non_beg_or_end_states[previous_state_index])))

            # print("tag_pair_to_test, final stage", tag_pair_to_test)
            # print("previous exisiting_alpha at location, final stage = ", previous_word_index, previous_state_index,
            #       exisiting_alpha[previous_word_index][previous_state_index])
            # print("tag transition prob, final stage = ", log(
            #     self.smoothed_tag_tag[self.start].prob(self.non_beg_or_end_states[previous_state_index])))
            #
            # print("summed prob", log(prob_est))

            # print("max prob is", maximum_prob)

        return log(prob_est);

    def return_backward_prob(self, word_list):

        # word_list = [w for (w, _) in sent]
        # create a list of states without start or end states
        # non_beg_or_end_states = list(freq_dist_tag_single.keys())
        # non_beg_or_end_states.remove(start)
        # tag_to_index_dict = dict.fromkeys(non_beg_or_end_states)
        # non_beg_or_end_states.remove(end)

        alpha = [['' for x in range(len(self.non_beg_or_end_states))] for y in range(len(word_list))]
        print(word_list)

        word_list.reverse()

        print(alpha)

        state_num = 0
        print(word_list)

        self.intialise_matrix_values(alpha, state_num, word_list)

        word_num = 2
        for word in word_list[2:-1]:

            state_num = 0
            for state in self.non_beg_or_end_states:
                # print("word investigated", word)
                # print(word_list[2:-1])
                alpha[word_num][state_num] = self.return_prob_sum(
                    alpha,
                    state,
                    word,
                    word_num - 1)
                state_num += 1
                # print("viterbi")
                # print(viterbi)
                #
                # print("backpoint")
                # print(back_pointer)
            word_num += 1

        total_path_prob = self.final_return_prob_sum(alpha, word_num - 1)

        return total_path_prob, alpha;

    def intialise_matrix_values(self, alpha, state_num, word_list):
        for state in self.non_beg_or_end_states:
            self.tag_to_index_dict[state] = state_num
            a = log(self.smoothed_tag_tag[state].prob(self.end))
            b = log(self.smoothed_word_tag[state].prob(word_list[1]))

            # if (state == start):
            #     viterbi[0][state_num] = log(1)

            alpha[1][state_num] = a + b
            state_num += 1
