# C'est une fonction récursive qui sélectionne la décision si possible.
# Sinon, créez des nœuds enfants.


def recursive_generate_tree(self, sample_data, sample_target, current_depth):
    # S'il n'y a qu'un seul résultat possible, sélectionnez-le comme décision
    if len(sample_target.unique()) == 1:
        self.decision = sample_target.unique()[0]
        # Si l'échantillon a moins de min_samples_split, sélectionnez la classe majoritaire
    elif len(sample_target) < self.min_samples_split:
        self.decision = self.get_maj_class(sample_target)

        # Si la profondeur de la branche courante est égale à max_depth \
        # sélectionnez la classe majoritaire
    elif current_depth == self.max_depth:
        self.decision = self.get_maj_class(sample_target)

        best_attribute, best_threshold, splitter = self.split_attribute(sample_data, sample_target)
        self.children = {}
        self.split_feat_name = best_attribute
        self.threshold = best_threshold
        current_depth += 1

        for v in splitter.unique():
            index = splitter == v  # Sélectionnez les index de chaque classe
            # S'il y a des données dans le nœud, créez un nouveau nœud d'arborescence avec cette partition
            if len(sample_data[index]) > 0:
                self.children[v] = TreeNode(min_samples_split=self.min_samples_split,
                                            max_depth=self.max_depth,
                                            seed=self.seed,
                                            verbose=self.verbose)
                self.children[v].recursive_generate_tree(sample_data[index], sample_target[index],
                                                         current_depth)

            else:
                self.children[v] = TreeNode(min_samples_split=self.min_samples_split,
                                            max_depth=1,
                                            seed=self.seed,
                                            verbose=self.verbose)
                self.children[v].recursive_generate_tree(sample_data, sample_target, current_depth=1)

                # Cette fonction définit quel est le meilleur attribut à diviser (chaîne \
                # ou continu)


def split_attribute(self, sample_data, sample_target):
    info_gain_max = -1 * float("inf")

    # Création d'une série vide pour stocker la variable dans laquelle la division est basée
    splitter = pd.Series(dtype='str')
    best_attribute = None
    best_threshold = None
    for attribute in sample_data.keys():
        if is_string_dtype(sample_data[attribute]):
            # Calculer le gain d'informations en utilisant cet attribut pour diviser la cible
            aig = self.compute_info_gain(sample_data[attribute], sample_target)
            if aig > info_gain_max:
                splitter = sample_data[attribute]
                info_gain_max = aig
                best_attribute = attribute
                best_threshold = None
                # Si l'attribut est un continu
            else:
                # Trier la variable continue dans un ordre croissant. Modifier l'ordre cible \
                # basé sur cela
                sorted_index = sample_data[attribute].sort_values(ascending=True).index
                sorted_sample_data = sample_data[attribute][sorted_index]
                sorted_sample_target = sample_target[sorted_index]
                # Itérer entre chaque échantillon, sauf le dernier
                for j in range(0, len(sorted_sample_data) - 1):
                    classification = pd.Series(dtype='str')

                    # Si deux échantillons consécutifs ne sont pas identiques,
                    # utilisez sa moyenne comme Un seuil
                    if sorted_sample_data.iloc[j] != sorted_sample_data.iloc[j + 1]:
                        threshold = (sorted_sample_data.iloc[j] +
                                     sorted_sample_data.iloc[j + 1]) / 2
                        classification = sample_data[attribute] > threshold
                        classification[classification] = 'greater'
                        classification[classification == False] = 'less'

                        # Calculer le gain d'information en utilisant la variable précédente \
                        # (maintenant catégorique)
                        aig = self.compute_info_gain(classification, sample_target)

                        if aig >= info_gain_max:
                            splitter = classification
                            info_gain_max = aig
                            best_attribute = attribute
                            best_threshold = threshold
            # Si verbeux est vrai, affiche le résultat de la division
            if self.verbose:
                if is_string_dtype(sample_data[best_attribute]):
                    print(f"Split by {best_attribute}, IG: {info_gain_max:.2f}")
                else:
                    print(f"Split by {best_attribute}, at {threshold}, IG: {info_gain_max:.2f}")

            return best_attribute, best_threshold, splitter
