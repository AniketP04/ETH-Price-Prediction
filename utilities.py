
import json


class Hyperparameters():
    def __init__(self):
        self.update('hyperparameters.json')

    def save(self, json_path):

        """
        Save the hyperparameters to a JSON file.

        Parameters:
        - json_path (str): The path to the JSON file where hyperparameters will be saved.
        """

        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):

        """
        Update the hyperparameters by loading them from a JSON file.

        Parameters:
        - json_path (str): The path to the JSON file containing hyperparameters.
        """

        with open(json_path) as f:
            hyperparameters = json.load(f)
            self.__dict__.update(hyperparameters)

    @property
    def dict(self):

        """
        Get the hyperparameters as a dictionary.

        Returns:
        - dict: A dictionary containing the hyperparameters.
        """
        
        return self.__dict__
