
    
import json
from typing import Iterable

import numpy

#
# Copyright 2022 by Vmio System JSC
# All rights reserved.
# Implement base class for json serialization purpose
# Ref: https://docs.python.org/3/library/json.html
#

class serializable(json.JSONEncoder):
    
    def serializable_default(self, obj):
        try:
            if hasattr(obj, '_custom_dict'):
                return obj._custom_dict()
            if hasattr(obj, '__dict__'):
                return obj.__dict__()
            
            try:
                iterable = iter(obj)
            except TypeError:
                pass
            else:
                return list(iterable)
            
            # Check if numppy object
            if isinstance(obj, numpy.integer):
                return int(obj)
            if isinstance(obj, numpy.floating):
                return float(obj)
            if isinstance(obj, numpy.ndarray):
                return obj.tolist()
            
            # Check other type here
            
            # Let the base class default method raise the TypeError
            return super(serializable, self).default(obj)
        except Exception as e:
            print(e)

    def __str__(self):
        '''
        __str__ is supposed to be human-readable
        '''
        return json.dumps(self._custom_dict(), ensure_ascii=False, sort_keys=False,default=self.serializable_default)
    
    def __repr__(self):
        '''A special method used to represent a class’s objects as a string.
        __repr__ is called by the repr() built-in function.
        __repr__ for machines (e.g., processed by other programs)
        '''        
        return str(self.__dict__)

    def to_json(self):
        return self.__str__()
    
    @staticmethod
    def convert_to_json(obj):
        if obj is None:
            return None
        
        if isinstance(obj, Iterable):
            return json.dumps(obj=obj, ensure_ascii=False, sort_keys=False, indent=4)#todo, default=self.serializable_default)
        return obj.to_json()
    
    
    def _custom_dict(self):
        """
        Present your class as a dictionay
        Child class should be implement it own method for overriding this
        Returns:
            dict: custom dict
        """    
        return self.__dict__


if __name__ == "__main__":
    # Sample usage
    class A(serializable):
        def __init__(self) -> None:
            self.a1 = 1
            self.a2 = 2
            
        def _custom_dict(self):
            return {
                "data-a1": self.a1,
                "data-a2": self.a2,
            }

    class B(serializable, list):
        
        def __init__(self):
            self.A1 = A()
            self.X1 = None
        
        def _custom_dict(self):
            return {
                "my A1": [self.A1,self.A1, 5],
                "my X1": self.X1,
            }
        

    # Use class method
    B1 = B()
    print(B1.to_json())
    
    # Use static method
    print(B.convert_to_json(B1))
    
    # Use static method for iterable object (like list, tuple...)
    B2 = B()
    B3 = B()
    print(B.convert_to_json([B1, B2, B3]))
    print(B.convert_to_json((B1, B2, B3)))
