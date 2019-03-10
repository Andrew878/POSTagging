from math import log, exp
from nltk import FreqDist, WittenBellProbDist

from math import log, exp
from nltk import FreqDist, WittenBellProbDist

"""Returns forward probabilities. Operates very similar to BeamSearch class"""


class ForwardProbabilities:
    """Same as Beam search"""

    def __init__(self, freq_dist_tag_single, smoothed_tag_tag, smoothed_word_tag):
        self.freq_dist_tag_single = freq_dist_tag_single
        self.smoothed_tag_tag = smoothed_tag_tag
        self.smoothed_word_tag = smoothed_word_tag
        self.start = '<s>'
        self.end = '</s>'

        self.non_beg_or_end_states = list(freq_dist_tag_single.keys())
        self.non_beg_or_end_states.remove(self.start)
        self.tag_to_index_dict = dict.fromkeys(self.non_beg_or_end_states)
        self.non_beg_or_end_states.remove(self.end)

    def return_forward_prob(self, word_list):

        """The core method of the class. Returns the forward probability and alpha matrix of path probabilities."""

        # initialise empty matrix
        alpha = [['' for x in range(len(self.non_beg_or_end_states))] for y in range(len(word_list))]

        # fill in first values
        self.intialise_matrix_values(alpha, word_list)

        # cycle through tokens
        word_num = 2
        for word in word_list[2:-1]:

            # cycle through tokens for each state
            state_num = 0
            for state in self.non_beg_or_end_states:

                # update alpha matrix using prior path values
                alpha[word_num][state_num] = self.return_prob_sum(alpha, state, word, word_num - 1)
                state_num += 1

            word_num += 1

        # termination stage
        total_path_prob = self.final_return_prob_sum(alpha, word_num - 1)

        # return values
        return total_path_prob, alpha;

    def return_prob_sum(self, exisiting_alpha, current_state, current_word,
                        previous_word_index, ):

        """Same as Beam search except no beam index ordering and summing all path ways instead of choosing best one."""

        prob_est = 0

        # emission probability for word and current state
        prob_current_word_given_state = log(self.smoothed_word_tag[current_state].prob(current_word))


        for previous_state_index in range(len(self.non_beg_or_end_states)):

            # add all probabilities to this state
            prob_est += exp(exisiting_alpha[previous_word_index][previous_state_index] + log(
                self.smoothed_tag_tag[self.non_beg_or_end_states[previous_state_index]].prob(
                    current_state)))


        return log(prob_est) + prob_current_word_given_state


    def final_return_prob_sum(self, exisiting_alpha, previous_word_index):

        """Same as Beam search except no beam index ordering and summing all path ways instead of choosing best one. """

        prob_est = 0

        # cycle through, and add, each path from final token to end of sentence token
        for previous_state_index in range(len(self.non_beg_or_end_states)):

            prob_est += exp(exisiting_alpha[previous_word_index][previous_state_index] + log(
                self.smoothed_tag_tag[self.non_beg_or_end_states[previous_state_index]].prob(self.end)))

        return log(prob_est);

    def intialise_matrix_values(self, alpha, word_list):

        """Same as Beam search except no back pointer matrix"""

        state_num = 0
        for state in self.non_beg_or_end_states:
            self.tag_to_index_dict[state] = state_num
            a = log(self.smoothed_tag_tag[self.start].prob(state))
            b = log(self.smoothed_word_tag[state].prob(word_list[1]))

            alpha[1][state_num] = a + b
            state_num += 1
