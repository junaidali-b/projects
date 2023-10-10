#Loading required libraries
import pickle
import inspect

#Backup and restore functions via pickle

#Backup

#Creating function to make pickle backups of the environment
def backup_env(*values):
    # Create a dictionary to store the values
    values_to_save = {}
    
    # Get the names of the input variables
    frame = inspect.currentframe().f_back
    _, _, _, local_vars = inspect.getargvalues(frame)
    
    # Iterate over the input values and add them to the dictionary
    for i, value in enumerate(values):
        value_name = [name for name, val in local_vars.items() if val is value][0]
        values_to_save[value_name] = value
    
    # Specify the file path for the pickle backup
    backup_file_path = 'D:/Python/Spyder/Honey Bee/Backups/beehive.pkl'
    
    # Open the file in binary mode and save the values
    with open(backup_file_path, 'wb') as backup_file:
        pickle.dump(values_to_save, backup_file)