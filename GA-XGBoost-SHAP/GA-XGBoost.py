import numpy as np
import pandas as pd
import random
import json
import pickle
from pathlib import Path
from typing import Dict, Tuple, Any
import warnings
from tqdm import tqdm
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score
import xgboost as xgb

warnings.filterwarnings('ignore')


# =========================== 配置参数 ===========================
class Config:
    """config"""

    # path
    DATA_PATH = 'dataset2.csv'  # CSVdata
    DATA_ENCODING = 'gbk'  #

    # data
    TEST_SIZE = 0.2  # test data
    RANDOM_STATE = 42  # random

    # GA
    POP_SIZE = 20  #
    GENERATIONS = 100  #
    PARENTS_RATIO = 0.4  #
    EARLY_STOP_PATIENCE = 20  #
    CV_FOLDS = 5  #

    # XGBoost hyparams
    PARAM_BOUNDS = {
        'learning_rate': (0.01, 0.3),
        'n_estimators': (100, 1000),
        'max_depth': (3, 10),
        'min_child_weight': (1, 10),
        'gamma': (0, 5),
        'subsample': (0.6, 1.0),
        'colsample_bytree': (0.6, 1.0)
    }

    # output
    OUTPUT_DIR = 'model/GA-XGBoost'  #
    MODEL_FILENAME = 'best_xgboost_model.pkl'  #
    PARAMS_FILENAME = 'best_params.json'  #
    TRAIN_INDICES_FILENAME = 'train_indices.csv'  #
    TEST_INDICES_FILENAME = 'test_indices.csv'  #

    #
    MODEL_FIXED_PARAMS = {
        'objective': 'reg:squarederror',
        'n_jobs': -1,
        'random_state': RANDOM_STATE
    }


# ===============================================================


class GeneticXGBoostOptimizer:
    """GA-XGBoost"""

    def __init__(self, param_bounds: Dict = None):
        """
        Args:
            param_bounds:
        """
        self.param_bounds = param_bounds if param_bounds else Config.PARAM_BOUNDS
        self.param_keys = list(self.param_bounds.keys())

    def initialize_population(self, pop_size: int) -> np.ndarray:
        """"""
        population = []
        for _ in range(pop_size):
            individual = []
            for key in self.param_keys:
                low, high = self.param_bounds[key]
                if key in ['n_estimators', 'max_depth']:  #
                    individual.append(random.randint(low, high))
                else:  #
                    individual.append(round(random.uniform(low, high), 4))
            population.append(individual)
        return np.array(population)

    def evaluate_fitness(self, params: Dict, X: np.ndarray, y: np.ndarray,
                         cv_folds: int = None) -> float:
        """5-fold cross（avgR²）"""
        if cv_folds is None:
            cv_folds = Config.CV_FOLDS

        #
        processed_params = params.copy()
        for key in ['n_estimators', 'max_depth']:
            if key in processed_params:
                processed_params[key] = int(processed_params[key])

        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=Config.RANDOM_STATE)
        scores = []

        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # create model
            model = xgb.XGBRegressor(
                **processed_params,
                **Config.MODEL_FIXED_PARAMS
            )

            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            scores.append(r2_score(y_val, y_pred))

        return np.mean(scores)

    def selection(self, population: np.ndarray, fitness: np.ndarray,
                  num_parents: int) -> np.ndarray:
        """select"""
        indices = np.argsort(fitness)[-num_parents:]  #
        return population[indices]

    def crossover(self, parents: np.ndarray, offspring_size: Tuple) -> np.ndarray:
        """avg cross"""
        offspring = np.empty(offspring_size)

        for i in range(offspring_size[0]):
            parent1_idx = i % parents.shape[0]
            parent2_idx = (i + 1) % parents.shape[0]

            for j in range(offspring_size[1]):
                if random.random() < 0.5:
                    offspring[i, j] = parents[parent1_idx, j]
                else:
                    offspring[i, j] = parents[parent2_idx, j]

        return offspring

    def mutation(self, offspring: np.ndarray) -> np.ndarray:
        """bianyi"""
        for i in range(offspring.shape[0]):
            param_idx = random.randint(0, offspring.shape[1] - 1)
            param_key = self.param_keys[param_idx]

            # random
            mutation_strength = random.uniform(-0.2, 0.2)

            if param_key in ['n_estimators', 'max_depth']:
                mutation = int(offspring[i, param_idx] * mutation_strength)
                offspring[i, param_idx] += mutation
            else:
                mutation = offspring[i, param_idx] * mutation_strength
                offspring[i, param_idx] += mutation

            # border check
            low, high = self.param_bounds[param_key]
            offspring[i, param_idx] = np.clip(offspring[i, param_idx], low, high)

            # int trans
            if param_key in ['n_estimators', 'max_depth']:
                offspring[i, param_idx] = int(offspring[i, param_idx])

        return offspring

    def optimize(self, X_train: np.ndarray, y_train: np.ndarray,
                 pop_size: int = None, generations: int = None,
                 parents_ratio: float = None, early_stop_patience: int = None) -> Dict:
        """
        main

        Returns:

        """
        #
        pop_size = pop_size if pop_size is not None else Config.POP_SIZE
        generations = generations if generations is not None else Config.GENERATIONS
        parents_ratio = parents_ratio if parents_ratio is not None else Config.PARENTS_RATIO
        early_stop_patience = (early_stop_patience if early_stop_patience is not None
                               else Config.EARLY_STOP_PATIENCE)

        num_parents = int(pop_size * parents_ratio)

        # start
        population = self.initialize_population(pop_size)
        best_params = None
        best_score = -np.inf
        no_improve_count = 0
        best_scores_history = []

        # GA
        for generation in tqdm(range(generations), desc="Genetic Optimization"):
            # 1
            fitness_scores = []
            for individual in population:
                params = dict(zip(self.param_keys, individual))
                score = self.evaluate_fitness(params, X_train, y_train)
                fitness_scores.append(score)

            fitness_scores = np.array(fitness_scores)

            # 2
            current_best_idx = np.argmax(fitness_scores)
            current_best_score = fitness_scores[current_best_idx]
            best_scores_history.append(current_best_score)

            if current_best_score > best_score:
                best_score = current_best_score
                best_params = dict(zip(self.param_keys, population[current_best_idx]))
                no_improve_count = 0
            else:
                no_improve_count += 1

            # 3
            if no_improve_count >= early_stop_patience:
                print(f"\nEarly stopping at generation {generation}")
                break

            # select
            parents = self.selection(population, fitness_scores, num_parents)

            # regeneration
            offspring_size = (pop_size - num_parents, population.shape[1])
            offspring = self.crossover(parents, offspring_size)

            # bianyi
            offspring = self.mutation(offspring)

            # new
            population[:num_parents] = parents
            population[num_parents:] = offspring

        print(f"\nBest CV R² Score: {best_score:.6f}")
        return best_params, best_scores_history


def load_and_split_data(filepath: str = None, encoding: str = None) -> Tuple:
    """data"""
    filepath = filepath if filepath is not None else Config.DATA_PATH
    encoding = encoding if encoding is not None else Config.DATA_ENCODING

    data = pd.read_csv(filepath, encoding=encoding)

    # last
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    print(f"Data shape: X={X.shape}, y={y.shape}")
    print(f"Features: {list(data.columns[:-1])}")
    print(f"Target: {data.columns[-1]}")

    # train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=Config.TEST_SIZE,
        random_state=Config.RANDOM_STATE,
        shuffle=True
    )

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    #
    train_indices = pd.DataFrame({
        'original_index': np.arange(len(y))[:len(y_train)]
    })
    test_indices = pd.DataFrame({
        'original_index': np.arange(len(y))[len(y_train):len(y_train) + len(y_test)]
    })

    return X_train, X_test, y_train, y_test, train_indices, test_indices


def train_final_model(X_train: np.ndarray, y_train: np.ndarray,
                      best_params: Dict) -> Tuple[xgb.XGBRegressor, Dict]:
    """final model"""
    #
    final_params = best_params.copy()
    for key in ['n_estimators', 'max_depth']:
        if key in final_params:
            final_params[key] = int(final_params[key])

    #
    final_params.update(Config.MODEL_FIXED_PARAMS)

    model = xgb.XGBRegressor(**final_params)
    model.fit(X_train, y_train)

    return model, final_params


def evaluate_model(model: xgb.XGBRegressor, X_test: np.ndarray,
                   y_test: np.ndarray) -> float:
    """evaluate model"""
    y_pred = model.predict(X_test)
    test_score = r2_score(y_test, y_pred)
    print(f"Test R² Score: {test_score:.6f}")
    return test_score


def save_results(model: xgb.XGBRegressor, best_params: Dict, train_indices: pd.DataFrame,
                 test_indices: pd.DataFrame, X_train: np.ndarray, X_test: np.ndarray,
                 y_train: np.ndarray, y_test: np.ndarray, output_dir: str = None) -> Path:
    """save"""
    output_dir = output_dir if output_dir is not None else Config.OUTPUT_DIR
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # 1. save model
    model_path = output_path / Config.MODEL_FILENAME
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to: {model_path}")

    # 2. save params
    params_path = output_path / Config.PARAMS_FILENAME
    with open(params_path, 'w', encoding='utf-8') as f:
        json.dump(best_params, f, indent=2, ensure_ascii=False)
    print(f"Parameters saved to: {params_path}")

    # 3. save train test data
    train_indices_path = output_path / Config.TRAIN_INDICES_FILENAME
    test_indices_path = output_path / Config.TEST_INDICES_FILENAME
    train_indices.to_csv(train_indices_path, index=False, encoding='gbk')
    test_indices.to_csv(test_indices_path, index=False, encoding='gbk')
    print(f"Train indices saved to: {train_indices_path}")
    print(f"Test indices saved to: {test_indices_path}")

    # 4. save data
    np.save(output_path / "X_train.npy", X_train)
    np.save(output_path / "X_test.npy", X_test)
    np.save(output_path / "y_train.npy", y_train)
    np.save(output_path / "y_test.npy", y_test)
    print(f"Data arrays saved to: {output_path}")

    return output_path


def main():
    """main"""
    print("=" * 50)
    print("XGBoost Regression with Genetic Algorithm Optimization")
    print("=" * 50)

    # 1. data
    print("\nStep 1: Loading and splitting data...")
    X_train, X_test, y_train, y_test, train_indices, test_indices = load_and_split_data()

    # 2. ga
    print("\nStep 2: Genetic algorithm optimization...")
    optimizer = GeneticXGBoostOptimizer()
    best_params, scores_history = optimizer.optimize(X_train, y_train)

    print("\nBest Parameters Found:")
    for key, value in best_params.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")

    # 3. train final model
    print("\nStep 3: Training final model with best parameters...")
    final_model, final_params = train_final_model(X_train, y_train, best_params)

    # 4. test evaluate
    print("\nStep 4: Evaluating on test set...")
    test_score = evaluate_model(final_model, X_test, y_test)

    # 5. save result
    print("\nStep 5: Saving results...")
    output_path = save_results(
        final_model, final_params, train_indices, test_indices,
        X_train, X_test, y_train, y_test
    )

    # 6. output
    print("\n" + "=" * 50)
    print("Optimization Complete!")
    print("=" * 50)
    print(f"Best CV R²: {np.max(scores_history):.6f}")
    print(f"Test R²: {test_score:.6f}")
    print(f"Optimization generations: {len(scores_history)}")
    print(f"All results saved to: {output_path}")

    # save
    history_path = output_path / "optimization_history.npy"
    np.save(history_path, np.array(scores_history))
    print(f"Optimization history saved to: {history_path}")

    print("=" * 50)


if __name__ == "__main__":
    main()