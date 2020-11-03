from sklearn.tree import DecisionTreeClassifier

from paje.searchspace.hp import CatHP, IntHP, RealHP, FixedHP
from paje.searchspace.configspace import ConfigSpace
from numpy.random import choice, uniform
from paje.ml.element.modelling.supervised.supervisedmodel import SupervisedModel
from paje.ml.element.preprocessing.supervised.feature.metaheuristic.metabase import MetaBase
##
from feature_selection import HarmonicSearch
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import make_scorer

class HS(MetaBase):
    
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        
        self.model = HarmonicSearch(**self.param())
        # self.model.estimator = eval(self.model.estimator)()
        self.model.cv_metric_function = make_scorer(eval(self.model.cv_metric_function))

    @classmethod
    def cs_impl(cls):
        # Sw
        # cs = ConfigSpace('Switch')
        # st = cs.start()
        # st.add_children([a.start, b.start, c.start])
        # cs.finish([a.end,b.end,c.end])

        hps = {
            # 'estimator': CatHP(choice, a=['DecisionTreeClassifier']),
            'number_gen': IntHP(uniform, low=10, high=500),
            'size_pop': IntHP(uniform, low=20, high=100),
            'sorting_method': CatHP(choice, a=['NSGA2']),
            'HMCR':   FixedHP(value=0.3),
            'repeat': FixedHP(value=1),
            'verbose':  FixedHP(value=0), 
            'make_logbook':  FixedHP(value=0), 
            'cv_metric_function': CatHP(choice, a=['matthews_corrcoef']),
        }

        return ConfigSpace(name='BBH', hps=hps)

if __name__== '__main__':
    cs = HS.cs().sample()

    HS(cs)