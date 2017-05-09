import numpy as np

from reviews_soli import ReviewsSolicitation


class EdmundsReviewSolicitation(ReviewsSolicitation):
    """Edmunds reviews have a fixed set of features that make the
    simulation much simpler.
    """

    def answer_by_gen(self, picked_feature):
        """Answer using sampling star's distribution of this product's reviews.
        Note: Always have answer
        Args:
            picked_feature: datamodel.Feature, returned by pick_method
        Returns:
            answered_star: int
        """
        star_dist = self.feature_to_star_dist[picked_feature.name]
        stars = np.arange(1, self.star_rank + 1, 1)
        answered_star = np.random.choice(stars, p=star_dist)
        return answered_star

    def answer_in_time_order(self, picked_feature):
        """Answer using real reviews sorted in time order.
        Args:
            picked_feature: datamodel.Feature, returned by pick_method
        Returns:
            answered_star: int
        """
        answered_review = self.reviews[0]   # earliest review
        answered_star = None
        if picked_feature.name in answered_review.feature_to_star.keys():
            answered_star = answered_review.feature_to_star[
                picked_feature.name]
        self.num_waiting_answers -= 1

        if self.num_waiting_answers <= 0:
            self.reviews.pop(0)
        return answered_star
