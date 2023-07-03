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
            print(f'{model.modelName} scores: {model.scores}')
            if model.scores is not None:
                for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc_score', 'mcc']:
                    for score in model.scores[metric]:  # Iteracja przez każdą wartość score zamiast obliczania średniej
                        boxplot_data.append({
                            'Model Name': f'{model.modelName} ({model.n_splits} splits, {model.features} features)',
                            'Metric': metric,
                            'Score': score
                        })


        self.df = pd.DataFrame(data)
        self.boxplot_df = pd.DataFrame(boxplot_data)
        print(self.boxplot_df)

    def visualize_models(self):

        if(self.model_list[0].skip):
            return

        plt.figure(figsize=(15, 8))
        print(self.df)
        sns.barplot(x='Model Name', y='Score', hue='Metric', data=self.df)
        plt.title(f'Model Scores ({self.s_method}; {self.s_features} features)')
        plt.ylabel('Score')
        plt.xlabel('Model')
        plt.show()

    def boxplot(self, metric):

        if(self.model_list[0].skip):
            return

        df_metric = self.df[self.df['Metric'] == metric]
        plt.figure(figsize=(15, 8))
        sns.boxplot(x='Model Name', y="Score", data=self.boxplot_df)
        plt.title(f'Boxplot of {metric}')
        plt.ylabel('Score')
        plt.xlabel('Model')
        plt.show()


    def venn_plot(self):
        data = {model.modelName: (model.featureNames if model.modelName != "IMG" else None) for model in self.model_list}
        print(data)

        for subset in itertools.combinations(data.keys(), 3):
            if ("IMG" in subset): continue
            subset_data = {key: data[key] for key in subset}
            subset_values = [set(values) for values in subset_data.values() if values is not None]

            if len(subset_values) > 0:
                if None in subset_values:
                    subset_values.remove(None)

                if len(subset_values) > 0:
                    venn3(subset_values, set_labels=list(subset_data.keys()))
                    plt.title('Venn Diagram')
                    plt.show()




    def feature_dependency_plot(self):
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc_score', 'mcc']

        for metric in metrics:
            data = []
            for i, model in enumerate(self.model_list):
                avg_score = sum(model.scores[metric])/len(model.scores[metric])
                data.append({'Liczba cech': features[i], 'Score': avg_score, 'Metric': metric})

            df = pd.DataFrame(data)
        
            plt.figure(figsize=(10, 5))
            sns.lineplot(x='Liczba cech', y='Score', data=df, marker='o')
            plt.title(f'Zmiana wartości metryki {metric} w zależności od liczby cech')
            plt.show()
