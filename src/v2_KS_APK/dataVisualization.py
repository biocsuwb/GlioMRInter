from . import *

class DataVisualizer:

    def __init__(self, model_list, s_method, s_features):
        self.model_list = model_list
        self.s_method = s_method
        self.s_features = s_features

        data = []
        for model in self.model_list:
            if model.scores is not None:
                for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc_score', 'mcc']:
                    data.append({
                        'n_splits': model.n_splits,
                        'Features': model.features,
                        'Model Name': f'{model.modelName} ({model.n_splits} splits, {model.features} features)',
                        'Metric': metric,
                        'Score': sum(model.scores[metric])/len(model.scores[metric])
                    })

        boxplot_data = []
        for model in self.model_list:
            if model.scores is not None:
                for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc_score', 'mcc']:
                    for score in model.scores[metric]:  # Iteracja przez każdą wartość score zamiast obliczania średniej
                        data.append({
                            'Model Name': f'{model.modelName} ({model.n_splits} splits, {model.features} features)',
                            'Metric': metric,
                            'Score': score
                        })


        self.df = pd.DataFrame(data)
        self.boxplot_df = pd.DataFrame(boxplot_data)
        print(self.boxplot_df)

    def visualize_models(self):

        plt.figure(figsize=(15, 8))
        sns.barplot(x='Model Name', y='Score', hue='Metric', data=self.df)
        plt.title(f'Model Scores ({self.s_method}; {self.s_features} features)')
        plt.ylabel('Score')
        plt.xlabel('Model')
        plt.show()

    def boxplot(self, metric):
        df_metric = self.df[self.df['Metric'] == metric]
        plt.figure(figsize=(15, 8))
        sns.boxplot(x="Model Name", y="Score", data=model.scores[metric])
        plt.title(f'Boxplot of {metric}')
        plt.ylabel('Score')
        plt.xlabel('Model')
        plt.show()

    def venn_plot(self):
        feature_sets = [set(model.features) for model in self.model_list]
        venn3(feature_sets, ('Model 1', 'Model 2', 'Model 3', 'Model 4'))
        plt.show()

    def feature_dependency_plot(self):
        data = []
        for model in self.model_list:
            data.append({
                'Model Name': model.modelName,
                'Number of Features': len(model.features),
                'Score': model.score
            })

        df = pd.DataFrame(data)

        plt.figure(figsize=(15, 8))
        sns.lineplot(x='Number of Features', y='Score', hue='Model Name', data=df)
        plt.title('Accuracy vs Number of Features')
        plt.ylabel('Score')
        plt.xlabel('Number of Features')
        plt.show()
