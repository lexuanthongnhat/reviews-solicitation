import scipy.stats as stats


class Review(object):
    """Abstract class for review
    """ 
    def __init__(self, feature_to_star, star_rank=5):
        self.feature_to_star = feature_to_star 
        self.star_rank = star_rank

        self.features = self.feature_to_star.keys()

    def star_of_feature(self, feature):
        return self.feature_to_star[feature]

    def __repr__(self):
        return repr(self.feature_to_star)

    @classmethod
    def import_csv(cls, file_path, star_rank=5):
        """
        Args:
            file_path (string)
            star_rank (int): e.g. 5 means 1, 2, 3, 4 and 5 stars system
        Returns:
            product_to_reviews (dict): product -> list of Reviews
        """
        pass


#TODO(Nhat): unit test for class Feature 
class Feature(object):
    """
    Represent a product feature/attribute/aspect
    The Feature's comparision is based on feature's cost. E.g.,
        'feature1 > feature2' means cost of feature 1 > cost of feature 2

    Attributes:
        name (string)
        ratings (list): e.g., [3, 0, 6] corresponds to 3, 0, 6 ratings for
            1, 2, 3 stars respectively. Require 0 with no ratings for that
            star.
        criterion: cost need to be optimized
    """ 
    def __init__(self, name, ratings): 
        self.name = name
        self.ratings = ratings
        self.criterion = self.sum_dirichlet_variances
        
    def increase_star(self, star, count=1):
        if star < 1 or star > len(self.ratings):
            raise IndexError
        self.ratings[star - 1] += count
    
    def get_num_ratings(self, star):
        if star > 0 and star <= len(self.ratings):
            return self.ratings[star - 1]
        else:
            raise IndexError

    def __repr__(self):
        return "{}: {}".format(self.name, self.ratings)

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.name == other.name

    def __neq__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self.name)

    def __lt__(self, other):
        return self.criterion() < other.criterion()

    def __le__(self, other):
        return self.criterion() <= other.criterion()

    def __gt__(self, other):
        return self.criterion() > other.criterion()

    def __ge__(self, other):
        return self.criterion() >= other.criterion()

    def sum_dirichlet_variances(self):
        alphas = [star + 1 for star in self.ratings]
        return sum(stats.dirichlet.var(alphas))

    @classmethod
    def product_cost(cls, features):
        return sum([feature.criterion() for feature in features]) 
