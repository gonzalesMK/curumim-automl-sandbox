""" Scaler Module
"""
from abc import ABC

from paje.ml.element.element import Element
from functools import partial

from paje.ml.element.modelling.supervised.supervisedmodel import SupervisedModel
class MetaBase(Element, ABC):
    def apply_impl(self, data, **kwargs):
        # self.model will be set in the child class
        for component in kwargs.get('components'):
            
            try:
                if isinstance(component.components[0], SupervisedModel):
                    self.model.estimator = component.components[0].model
                    print("Set!")
            except Exception as e:
                pass

        if not self.model.estimator:
            raise("The Metaheuristic needs a Classifier in the pipeline")

        self.model.fit(*data.Xy)
        return self.use_impl(data)

    def use_impl(self, data):
        return data.updated(self, X=self.model.transform(data.X))

    def modifies(self, op):
        return ['X']
    
    def apply(self, data=None, **kwargs):
        components = kwargs.get("components")
        if components:
            self.apply_impl = partial( self.apply_impl, components=components)
        
        return super().apply(data)