# Optimizing-machine-learning-pipelines-using-genetic-programming
This is a variation of the Linear Support Vector Classiifier for Hong Kong Economic Data. It automatically chooses the best machine learning algorithm and hyperparameters with genetic algorithm for 100 generations. 
It generates better result than the original Linear Support Vector Classiifier by almost 5%.

# Result:
After running 100 generations, we achieve a 0.94 accuracy, which is based on sklearn accuracy rather than the TPOT default accuracy to avoid confusion. 

Best pipeline: GradientBoostingClassifier(GaussianNB(FeatureAgglomeration(GradientBoostingClassifier(PCA(StandardScaler(CombineDFs(input_matrix, input_matrix)), 8), 19.0, 4.0), 24, 9)), 0.01, 28.0)
(None, 0.94935064935064939) with the same accuracy scale.

# Acknowledgement: 
Thanks to @rhiever and @weixuanfu2016, I successfully implemented TPOT into choosing classifier.
