from math import log, exp
from nltk import FreqDist, WittenBellProbDist

from math import log, exp
from nltk import FreqDist, WittenBellProbDist

"""Returns backward probabilities. Operates very similar to BeamSearch and Forward Search class"""


class BackwardProbabilities:
    """Same as Beam search and Forward probabilities"""

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

    def return_backward_prob(self, word_list):

        """Main method of class. Returns the final probability and matrix. Similar to Forward probabilities but the
        order of reviewing tokens is reversed"""

        beta = [['' for x in range(len(self.non_beg_or_end_states))] for y in range(len(word_list))]

        # reverse word ordering so we can work backwards through the sentence
        word_list.reverse()

        self.intialise_matrix_values(beta, word_list)

        # update beta matrix token by token...
        word_num = 2
        for word in word_list[2:-1]:

            # ...and state by state
            state_num = 0
            for state in self.non_beg_or_end_states:

                beta[word_num][state_num] = self.return_prob_sum(beta, state, word, word_num - 1)

                state_num += 1

            word_num += 1

        total_path_prob = self.final_return_prob_sum(beta, word_num - 1, word_list[word_num])

        # return final backward probability and beta matrix
        return total_path_prob, beta;

    def return_prob_sum(self, exisiting_beta, current_state, current_word,
                        previous_word_index, ):

        """Same as Beam search except no beam index ordering and summing all path ways instead of choosing best one"""

        prob_est = 0

        # emission probability
        prob_current_word_given_state = log(self.smoothed_word_tag[current_state].prob(current_word))

        # cycle through each state and add state transition probabilities
        for previous_state_index in range(len(self.non_beg_or_end_states)):

            prob_est += exp(exisiting_beta[previous_word_index][previous_state_index] + log(
                self.smoothed_tag_tag[current_state].prob(
                    self.non_beg_or_end_states[previous_state_index])))

        # return all sum of path probabilities (include emission probability too)
        return log(prob_est) + prob_current_word_given_state

    def final_return_prob_sum(self, exisiting_beta, previous_word_index, first_word):

        """Same as Beam search except no beam index ordering and summing all path ways instead of choosing best one.
        Unlike the final step for Forward probabilities, we now include the emission probability, per page 190,
        section 6.5 of Jurafsky & Martin"""

        prob_est = 0

        # cycle through, and add, each path from first token to start of sentence token
        for previous_state_index in range(len(self.non_beg_or_end_states)):

            prob_current_word_given_state = log(
                self.smoothed_word_tag[self.non_beg_or_end_states[previous_state_index]].prob(first_word))

            prob_est += exp(exisiting_beta[previous_word_index][previous_state_index] + log(
                self.smoothed_tag_tag[self.start].prob(
                    self.non_beg_or_end_states[previous_state_index])) + prob_current_word_given_state)


        return log(prob_est)

    def intialise_matrix_values(self, beta, word_list):

        """Operates in similar fashion to Forward and Beam, except that we do not include the emission probability,
        per page 190, section 6.5 of Jurafsky & Martin"""

        # cycle through states
        state_num = 0
        for state in self.non_beg_or_end_states:
            self.tag_to_index_dict[state] = state_num

            # update beta matrix with transition prob from final state
            a = log(self.smoothed_tag_tag[state].prob(self.end))

            beta[1][state_num] = a
            state_num += 1
