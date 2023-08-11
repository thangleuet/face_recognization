import os
from pyclbr import Function
import sys

#
# Copyright 2022 by Vmio System JSC
# All rights reserved.
# Utility functions for file streaming
#

class file_iterator(object):
    first = True
    def __init__(self, filename:str, chunksize:int=-1, progress_callback:Function = None):
        self.filename = filename
        self.totalsize = os.path.getsize(filename)
        self.chunksize = chunksize if chunksize > 0 else self.totalsize 
        self.readsofar = 0
        self.progress_callback = progress_callback

    def __iter__(self):
        with open(self.filename, 'rb') as file:
            while True:
                data = file.read(self.chunksize)
                if not data:
                    break
                self.readsofar += len(data)
                if self.progress_callback != None:
                    self.progress_callback(self.totalsize, self.readsofar)
                yield data

    def __len__(self):
        return self.totalsize
    
    def remain(self):
        return self.totalsize - self.readsofar

class file_chuck_adapter(object):
    def __init__(self, filename:str, chunksize:int=-1, progress_callback:Function = None):
        self.iterable = file_iterator(filename, chunksize, progress_callback)
        self.iterator = iter(self.iterable)
        self.length = len(self.iterable)

    def read(self, size:int=-1): 
        data =  next(self.iterator, b'')
        return data

    def __len__(self):
        return self.iterable.remain()