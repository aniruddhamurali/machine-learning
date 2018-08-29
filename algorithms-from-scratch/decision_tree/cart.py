# Class for performing a Classification and Regression Tree

class cart:
    n_data_sets = 7  # number of sets that the data is split into for model
    max_depth = 5    # maximum height of tree
    min_size = 7     # minimum number of patterns within a node

    ''' Splits a dataset based on an attribute and its value on splitting point.'''
    def splitGroup(index, value, dataset):
        left, right = list(), list()
        for row in dataset:
            if row[index] < value:
                left.append(row)
            else:
                right.append(row)
        return left, right


    ''' Calculates the Gini index for a split dataset.'''
    def calculateGiniIndex(groups, classes):
	# count all samples at split point
        total_samples = float(sum([len(group) for group in groups]))
        # sum weighted Gini index for each group
        gini = 0.0
        for group in groups:
            size = float(len(group))
            # avoid divide by zero
            if size == 0:
                continue
            score = 0.0
            # score the group based on the score for each class
            for class_val in classes:
                p = [row[-1] for row in group].count(class_val) / size
                score += p * p
            # weigh the group score by its relative size
            gini += (1.0 - score) * (size / total_samples)
        return gini


    ''' Selects the best split point for a dataset.'''
    def getSplitBestNode(dataset):
        class_values = list(set(row[-1] for row in dataset))
        best_index, best_value, best_score, best_groups = 999, 999, 999, None
        for index in range(len(dataset[0])-1):
            for row in dataset:
                groups = cart.splitGroup(index, row[index], dataset)
                gini = cart.calculateGiniIndex(groups, class_values)
                if gini < best_score:
                    best_index, best_value, best_score, best_groups = index, row[index], gini, groups
        return {'index':best_index, 'value':best_value, 'groups':best_groups}


    ''' Sets a terminal node value.'''
    def toTerminalNode(group):
        outcomes = [row[-1] for row in group]
        return max(set(outcomes), key=outcomes.count)

			
    ''' Creates child splits for a node or creates a terminal.'''
    def splitNode(node, max_depth, min_size, depth):
        left, right = node['groups']
        del(node['groups'])
        # check for a node split
        if not left or not right:
            node['left'] = node['right'] = cart.toTerminalNode(left + right)
            return
        # check for max depth
        if depth >= max_depth:
            node['left'], node['right'] = cart.toTerminalNode(left), cart.toTerminalNode(right)
            return
        # process left child
        if len(left) <= min_size:
            node['left'] = cart.toTerminalNode(left)
        else:
            node['left'] = cart.getSplitBestNode(left)
            cart.splitNode(node['left'], max_depth, min_size, depth+1)
        # process right child
        if len(right) <= min_size:
            node['right'] = cart.toTerminalNode(right)
        else:
            node['right'] = cart.getSplitBestNode(right)
            cart.splitNode(node['right'], max_depth, min_size, depth+1)

			
    ''' Builds a decision tree.'''
    def buildTree(train, max_depth, min_size):
        root = cart.getSplitBestNode(train)
        cart.splitNode(root, max_depth, min_size, 1)
        return root


    ''' Prints a decision tree.'''
    def printTree(node, depth=0):
        if isinstance(node, dict):
            printTree(node['left'], depth+1)
            printTree(node['right'], depth+1)
        else:
            print('%s[%s]' % ((depth*' ', node)))

		
    ''' Makes a prediction with a decision tree.'''
    def predict(node, row):
        if row[node['index']] < node['value']:
            if isinstance(node['left'], dict):
                return cart.predict(node['left'], row)
            else:
                return node['left']
        else:
            if isinstance(node['right'], dict):
                return cart.predict(node['right'], row)
            else:
                return node['right']			
				
				
    # Split a dataset into k folds
    ''' Splits a dataset based on an attribute and its value for validation.'''
    def splitData(dataset, random_set_splits):
        dataset_split = list()
        dataset_copy = list(dataset)
        fold_size = int(len(dataset) / random_set_splits)
        
        for i in range(random_set_splits):
            fold = list()
            while len(fold) < fold_size:
                index = random.randrange(len(dataset_copy))
                fold.append(dataset_copy.pop(index))
            dataset_split.append(fold)
            
        return dataset_split


    ''' Returns the accuracy of a set of data by comparing actual to predicted values.'''
    def calculateAccuracy(actual, predicted):
        correct = 0
        for i in range(len(actual)):
            if actual[i] == predicted[i]:
                correct += 1
        return correct / float(len(actual)) * 100.0


    ''' Returns predictions based on the decision tree.'''
    def testPredictions(train, test, max_depth, min_size):
        tree = cart.buildTree(train, max_depth, min_size)
        predictions = list()
        for row in test:
            prediction = cart.predict(tree, row)
            predictions.append(prediction)
        return(predictions)


    # Evaluate an algorithm using a cross validation split
    ''' Splits the data into sets and returns the mean accuracy of the model.'''
    def testPredictiveModel(dataset, random_set_splits, max_depth, min_size):
        scores = list()
        sets = cart.splitData(dataset, random_set_splits)
		
        for random_set in sets:
            train_set = list(sets)
            train_set.remove(random_set)
            train_set = sum(train_set, [])
            test_set = list()
            
            for row in random_set:
                row_copy = list(row)
                test_set.append(row_copy)
                row_copy[-1] = None

            predictions = cart.testPredictions(train_set, test_set, max_depth, min_size)

            actual = [row[-1] for row in random_set]
            accuracy = cart.calculateAccuracy(actual, predictions)
            scores.append(accuracy)

        # return average score (accuracy)
        return (sum(scores)/float(len(scores)))
	
