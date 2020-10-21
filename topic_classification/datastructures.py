class TrainingData:
    def __init__(self, cv_train_features, train_label_names,
                 cv_test_features, test_label_names):
        self.cv_train_features = cv_train_features
        self.train_label_names = train_label_names
        self.cv_test_features = cv_test_features
        self.test_label_names = test_label_names
