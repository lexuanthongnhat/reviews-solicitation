import numpy as np

from reviews_soli import ReviewsSolicitation, SimulationStats
from data_model import Review, Feature


class EdmundsReviewSolicitation(ReviewsSolicitation):
    """Edmunds reviews have a fixed set of features that make the
    simulation much simpler.
    """

    def ask_greedily_answer_by_gen(self):
        """Greedily ask question, answer using sampling star's distribution
        of this product's reviews.
        Note: Always have answer
        """
        return self.ask_then_answer_by_gen(
            pick_func='pick_highest_cost_feature')

    def ask_greedily_prob_answer_by_gen(self):
        """Ask question with probability proportional to feature's cost,
        answer using sampling star's distribution of this product's reviews.
        Note: Always have answer
        """
        return self.ask_then_answer_by_gen(
            pick_func='pick_feature_with_prob')

    def ask_randomly_answer_by_gen(self):
        """Ask question randomly, answer using sampling star's distribution
        of this product's reviews.
        Note: Always have answer
        """
        return self.ask_then_answer_by_gen(
            pick_func='pick_random_feature')

    def ask_then_answer_by_gen(self, pick_func='pick_highest_cost_feature'):
        """Ask question by 'pick_func', answer using sampling
        star's distribution of this product's reviews
        Note: Always have answer
        """
        star_dist = Review.sample_star_dist(self.reviews)
        stars = np.arange(1, len(star_dist) + 1, 1)

        picked_feature = self.__getattribute__(pick_func)()
        answered_star = np.random.choice(stars, p=star_dist)
        return (picked_feature, answered_star)

    def ask_greedily_answer_mostly(self):
        """Greedily ask questions to reduce the cost
        Args:
            answer_possibility: float [0, 1] representing the possibility
                that customers answer a question
        """
        picked_feature = self.pick_highest_cost_feature()

        for review in self.reviews:
            if picked_feature.name in review.feature_to_star.keys():
                answered_star = review.feature_to_star[picked_feature.name]
                self.reviews.remove(review)
                return (picked_feature, answered_star)

        # No answer for this feature
        self.reviews.extend(self.original_reviews.copy())
        return (picked_feature, None)

    def ask_greedily_answer_in_time_order(self):
        return self.ask_then_answer_in_time_order(
            pick_func='pick_highest_cost_feature')

    def ask_greedily_prob_answer_in_time_order(self):
        return self.ask_then_answer_in_time_order(
            pick_func='pick_feature_with_prob')

    def ask_randomly_answer_in_time_order(self):
        return self.ask_then_answer_in_time_order(
            pick_func='pick_random_feature')

    def ask_then_answer_in_time_order(self, pick_func='pick_random_feature'):
        """Ask questions using pick_func, answer in time order."""
        picked_feature = self.__getattribute__(pick_func)()

        answered_review = self.reviews[0]   # earliest review
        answered_star = None
        if picked_feature.name in answered_review.feature_to_star.keys():
            answered_star = answered_review.feature_to_star[
                picked_feature.name]
        self.num_waiting_answers -= 1
        
        if self.num_waiting_answers <= 0:
            self.reviews.pop(0)
            
        return (picked_feature, answered_star)
