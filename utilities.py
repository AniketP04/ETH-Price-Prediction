#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json


class Hyperparameters():
    def __init__(self):
        self.update('hyperparameters.json')

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        with open(json_path) as f:
            hyperparameters = json.load(f)
            self.__dict__.update(hyperparameters)

    @property
    def dict(self):
        return self.__dict__


# In[ ]:




