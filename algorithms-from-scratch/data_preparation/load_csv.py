# Loads a CSV file into a list
def load_csv(filename):
    dataset = []
    # Values in a data point are separated by a delimiter (comma, space, tab, etc.)
    delimiter = ','
    file = open(filename, 'r')
    for line in file:
        if not line:
            continue
        line = line.split(delimiter)
        dataset.append(line)
    return dataset
        
