import os
import random
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

# ==================== CONSTANTS & PATHS ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, 'fruit_ripeness_dataset', 'archive (1)', 'dataset')
ARTIFACT_DIR = r"C:\Users\Lenovo\.gemini\antigravity\brain\c6bb5018-1bc5-4b63-a9a5-92c59051c6b9"

# Ensure artifact directory exists
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# Categories
CATEGORIES = {
    'rotten': 'rottenbanana',
    'unripe': 'unripe banana',
    'ripe': 'freshbanana'
}

# ==================== FEATURE EXTRACTION ====================
def extract_features(image_path):
    """
    Ekstraksi fitur warna gabungan (RGB + LAB):
    - Resize ke (100, 100)
    - Konversi ke RGB dan LAB
    - Rata-rata & Std Dev R, G, B (6 dimensi)
    - Rata-rata & Std Dev L, A, B (6 dimensi)
    Menghasilkan vektor 12-dimensi.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    img_rgb_res = cv2.resize(img_rgb, (100, 100))
    img_lab_res = cv2.resize(img_lab, (100, 100))
    
    # Hitung rata-rata dan standar deviasi untuk RGB
    mean_rgb = np.mean(img_rgb_res, axis=(0, 1))  # [mean_R, mean_G, mean_B]
    std_rgb = np.std(img_rgb_res, axis=(0, 1))    # [std_R, std_G, std_B]
    
    # Hitung rata-rata dan standar deviasi untuk LAB (Kecerahan & Kejenuhan)
    mean_lab = np.mean(img_lab_res, axis=(0, 1))  # [mean_L, mean_A, mean_B]
    std_lab = np.std(img_lab_res, axis=(0, 1))    # [std_L, std_A, std_B]
    
    # Gabungkan menjadi vektor 12-dimensi
    return np.concatenate([mean_rgb, std_rgb, mean_lab, std_lab])

def get_image_paths(category_name, split='train'):
    """Mendapatkan daftar semua file gambar di folder kategori"""
    folder_path = os.path.join(DATASET_DIR, split, CATEGORIES[category_name])
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
    paths = []
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    for file in os.listdir(folder_path):
        if file.lower().endswith(valid_extensions):
            paths.append(os.path.join(folder_path, file))
    return paths

# ==================== CLASSIFIER CLASSES ====================
class MinDistanceClassifier:
    """Opsi A: Minimum Distance Classifier (Euclidean Distance)"""
    def __init__(self):
        self.means = {}

    def fit(self, X_dict):
        for label, X in X_dict.items():
            self.means[label] = np.mean(X, axis=0)

    def predict_single(self, x):
        dists = {}
        for label, mean in self.means.items():
            dists[label] = np.linalg.norm(x - mean)
        pred = min(dists, key=dists.get)
        return pred, dists

    def predict(self, X):
        return np.array([self.predict_single(x)[0] for x in X])


class BayesClassifier:
    """Opsi B: Bayes Decision Classifier (Gaussian Likelihood with Prior)"""
    def __init__(self, reg=1e-5):
        self.means = {}
        self.covs = {}
        self.priors = {}
        self.reg = reg

    def fit(self, X_dict):
        total_samples = sum(len(X) for X in X_dict.values())
        for label, X in X_dict.items():
            self.means[label] = np.mean(X, axis=0)
            cov = np.cov(X, rowvar=False)
            # Regularisasi untuk menghindari matriks singular
            self.covs[label] = cov + self.reg * np.eye(cov.shape[0])
            self.priors[label] = len(X) / total_samples

    def predict_single(self, x):
        log_posteriors = {}
        for label in self.means.keys():
            mean = self.means[label]
            cov = self.covs[label]
            prior = self.priors[label]
            
            diff = x - mean
            sign, logdet = np.linalg.slogdet(cov)
            if sign <= 0:
                logdet = np.sum(np.log(np.diagonal(cov)))  # fallback
            
            inv_cov = np.linalg.inv(cov)
            mahalanobis_sq = diff.T @ inv_cov @ diff
            
            # log p(x|c) + log p(c)
            log_likelihood = -0.5 * logdet - 0.5 * mahalanobis_sq
            log_posteriors[label] = log_likelihood + np.log(prior)
            
        pred = max(log_posteriors, key=log_posteriors.get)
        return pred, log_posteriors

    def predict(self, X):
        return np.array([self.predict_single(x)[0] for x in X])


class MahalanobisClassifier:
    """Opsi C: Mahalanobis Distance Classifier"""
    def __init__(self, reg=1e-5):
        self.means = {}
        self.covs = {}
        self.reg = reg

    def fit(self, X_dict):
        for label, X in X_dict.items():
            self.means[label] = np.mean(X, axis=0)
            cov = np.cov(X, rowvar=False)
            self.covs[label] = cov + self.reg * np.eye(cov.shape[0])

    def predict_single(self, x):
        dists = {}
        for label in self.means.keys():
            mean = self.means[label]
            cov = self.covs[label]
            inv_cov = np.linalg.inv(cov)
            diff = x - mean
            dists[label] = np.sqrt(diff.T @ inv_cov @ diff)
        pred = min(dists, key=dists.get)
        return pred, dists

    def predict(self, X):
        return np.array([self.predict_single(x)[0] for x in X])

# ==================== MAIN PIPELINE ====================
def main():
    print("=== TAHAP 1: MEMUAT PATH DATASET APEL ===")
    
    # Ambil semua file path sekali saja
    all_train_paths = {}
    all_test_paths = {}
    for cat in CATEGORIES.keys():
        all_train_paths[cat] = get_image_paths(cat, 'train')
        all_test_paths[cat] = get_image_paths(cat, 'test')
        print(f"Kategori '{cat}': Train={len(all_train_paths[cat])} gambar, Test={len(all_test_paths[cat])} gambar")

    # Inisialisasi struktur penyimpanan metrik untuk 3 run
    runs_results = []
    
    # Menggunakan system randomness untuk menghasilkan 3 seed acak yang unik
    num_runs = 3
    random.seed(None)
    seeds = [random.randint(1, 1000000) for _ in range(num_runs)]
    print(f"\nSeed acak yang dihasilkan untuk 3 kali run: {seeds}")

    # Aggregator untuk confusion matrix
    cm_accumulators = {
        'Option A (Minimum Distance)': np.zeros((3, 3)),
        'Option B (Bayes Gaussian)': np.zeros((3, 3)),
        'Option C (Mahalanobis Distance)': np.zeros((3, 3))
    }

    # Untuk menyimpan data terakhir agar bisa digambar decision boundary-nya
    last_run_data = {}

    for run_idx, seed in enumerate(seeds):
        run_num = run_idx + 1
        print(f"\n=================== RUN KE-{run_num} (Seed: {seed}) ===================")
        random.seed(seed)
        np.random.seed(seed)

        # Ambil random 100 gambar per kategori untuk training
        train_samples = {cat: random.sample(all_train_paths[cat], 100) for cat in CATEGORIES.keys()}
        # Ambil random 100 gambar per kategori untuk testing (balanced evaluation)
        test_samples = {cat: random.sample(all_test_paths[cat], min(100, len(all_test_paths[cat]))) for cat in CATEGORIES.keys()}

        # Ekstraksi fitur (gabungan RGB + LAB)
        X_train_dict = {cat: [] for cat in CATEGORIES.keys()}
        X_test_dict = {cat: [] for cat in CATEGORIES.keys()}
        
        for cat in CATEGORIES.keys():
            for p in train_samples[cat]:
                X_train_dict[cat].append(extract_features(p))
            X_train_dict[cat] = np.array(X_train_dict[cat])
            
            for p in test_samples[cat]:
                X_test_dict[cat].append(extract_features(p))
            X_test_dict[cat] = np.array(X_test_dict[cat])

        # Satukan data untuk evaluasi sklearn
        X_train, y_train = [], []
        for cat in CATEGORIES.keys():
            X_train.extend(X_train_dict[cat])
            y_train.extend([cat] * len(X_train_dict[cat]))
        X_train = np.array(X_train)
        y_train = np.array(y_train)

        X_test, y_test = [], []
        for cat in CATEGORIES.keys():
            X_test.extend(X_test_dict[cat])
            y_test.extend([cat] * len(X_test_dict[cat]))
        X_test = np.array(X_test)
        y_test = np.array(y_test)

        # Simpan data run terakhir untuk visualisasi decision boundary
        if run_num == num_runs:
            last_run_data = {
                'X_train': X_train,
                'y_train': y_train,
                'X_test': X_test,
                'y_test': y_test,
                'X_train_dict': X_train_dict,
                'X_test_dict': X_test_dict
            }

        # Latih dan evaluasi
        classifiers = {
            'Option A (Minimum Distance)': MinDistanceClassifier(),
            'Option B (Bayes Gaussian)': BayesClassifier(),
            'Option C (Mahalanobis Distance)': MahalanobisClassifier()
        }
        
        run_metrics = {}
        for name, clf in classifiers.items():
            clf.fit(X_train_dict)
            y_pred = clf.predict(X_test)
            
            acc = accuracy_score(y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average=None, labels=list(CATEGORIES.keys()))
            macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro', labels=list(CATEGORIES.keys()))
            cm = confusion_matrix(y_test, y_pred, labels=list(CATEGORIES.keys()))
            
            # Akumulasikan confusion matrix
            cm_accumulators[name] += cm
            
            run_metrics[name] = {
                'accuracy': acc,
                'macro_precision': macro_prec,
                'macro_recall': macro_rec,
                'macro_f1': macro_f1,
                'by_class': {
                    cat: {
                        'precision': precision[i],
                        'recall': recall[i],
                        'f1': f1[i]
                    } for i, cat in enumerate(CATEGORIES.keys())
                }
            }
            print(f"  {name} -> Accuracy: {acc:.4f} | Macro F1: {macro_f1:.4f}")
        
        runs_results.append({
            'run': run_num,
            'seed': seed,
            'metrics': run_metrics
        })

    # Print mean statistics tiap kelas (dari run terakhir)
    print("\n=== STATISTIK RATA-RATA TIAP KELAS (TRAIN SET - RUN TERAKHIR) ===")
    for cat in CATEGORIES.keys():
        m = np.mean(last_run_data['X_train_dict'][cat], axis=0)
        print(f"Mean {cat.upper()}:")
        print(f"  RGB: R-Mean={m[0]:.2f}, G-Mean={m[1]:.2f}, B-Mean={m[2]:.2f} | R-Std={m[3]:.2f}, G-Std={m[4]:.2f}, B-Std={m[5]:.2f}")
        print(f"  LAB: L-Mean={m[6]:.2f}, A-Mean={m[7]:.2f}, B-Mean={m[8]:.2f} | L-Std={m[9]:.2f}, A-Std={m[10]:.2f}, B-Std={m[11]:.2f}")

    # ==================== EVALUASI AGREGAT (MEAN & STD DEV) ====================
    print("\n=================== RINGKASAN AGREGAT (3 RUNS) ===================")
    agg_results = {}
    for name in cm_accumulators.keys():
        accs = [r['metrics'][name]['accuracy'] for r in runs_results]
        f1s = [r['metrics'][name]['macro_f1'] for r in runs_results]
        precs = [r['metrics'][name]['macro_precision'] for r in runs_results]
        recs = [r['metrics'][name]['macro_recall'] for r in runs_results]
        
        mean_acc, std_acc = np.mean(accs), np.std(accs)
        mean_f1, std_f1 = np.mean(f1s), np.std(f1s)
        mean_prec, std_prec = np.mean(precs), np.std(precs)
        mean_rec, std_rec = np.mean(recs), np.std(recs)
        
        agg_results[name] = {
            'accuracy': {'mean': mean_acc, 'std': std_acc, 'values': accs},
            'macro_f1': {'mean': mean_f1, 'std': std_f1, 'values': f1s},
            'macro_precision': {'mean': mean_prec, 'std': std_prec, 'values': precs},
            'macro_recall': {'mean': mean_rec, 'std': std_rec, 'values': recs},
            'avg_confusion_matrix': (cm_accumulators[name] / num_runs).tolist()
        }
        
        print(f"\n{name} Agregat:")
        print(f"  Akurasi      : {mean_acc:.4f} ± {std_acc:.4f}")
        print(f"  Macro F1-skor: {mean_f1:.4f} ± {std_f1:.4f}")
        print(f"  Macro Presisi: {mean_prec:.4f} ± {std_prec:.4f}")
        print(f"  Macro Recall : {mean_rec:.4f} ± {std_rec:.4f}")

    # Simpan data metrik agregat ke JSON
    output_json = {
        'seeds': seeds,
        'runs': runs_results,
        'aggregates': agg_results
    }
    with open(os.path.join(ARTIFACT_DIR, 'evaluation_metrics.json'), 'w') as f:
        json.dump(output_json, f, indent=4)

    # ==================== VISUALISASI 1: CONFUSION MATRIX AGREGAT ====================
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for idx, name in enumerate(cm_accumulators.keys()):
        ax = axes[idx]
        cm_avg = cm_accumulators[name] / num_runs
        
        im = ax.imshow(cm_avg, interpolation='nearest', cmap=plt.cm.Blues)
        ax.set_title(f"{name}\nRata-rata Akurasi: {agg_results[name]['accuracy']['mean']:.2%}")
        
        # Add labels
        tick_marks = np.arange(len(CATEGORIES))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(list(CATEGORIES.keys()))
        ax.set_yticklabels(list(CATEGORIES.keys()))
        
        # Add text values
        thresh = cm_avg.max() / 2.
        for i in range(cm_avg.shape[0]):
            for j in range(cm_avg.shape[1]):
                ax.text(j, i, f"{cm_avg[i, j]:.1f}",
                        ha="center", va="center",
                        color="white" if cm_avg[i, j] > thresh else "black")
                
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
    
    plt.tight_layout()
    cm_path = os.path.join(ARTIFACT_DIR, 'confusion_matrices.png')
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"\nAveraged confusion matrices plot saved to: {cm_path}")

    # ==================== VISUALISASI 2: DECISION BOUNDARY (DARI RUN TERAKHIR) ====================
    print("\n=== TAHAP 4: TRAINING CLASSIFIERS (2D PCA PROJECTION) UNTUK BATAS KEPUTUSAN (RUN TERAKHIR) ===")
    X_train_last = last_run_data['X_train']
    X_train_dict_last = last_run_data['X_train_dict']

    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train_last)

    # Re-group PCA training data by category
    X_train_pca_dict = {}
    idx = 0
    for cat in CATEGORIES.keys():
        n_samples = len(X_train_dict_last[cat])
        X_train_pca_dict[cat] = X_train_pca[idx : idx + n_samples]
        idx += n_samples

    # Fit classifiers in 2D
    clf_A_2d = MinDistanceClassifier()
    clf_A_2d.fit(X_train_pca_dict)
    
    clf_B_2d = BayesClassifier()
    clf_B_2d.fit(X_train_pca_dict)
    
    clf_C_2d = MahalanobisClassifier()
    clf_C_2d.fit(X_train_pca_dict)

    # Grid for contour plot
    x_min, x_max = X_train_pca[:, 0].min() - 10, X_train_pca[:, 0].max() + 10
    y_min, y_max = X_train_pca[:, 1].min() - 10, X_train_pca[:, 1].max() + 10
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.5), np.arange(y_min, y_max, 0.5))
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # Predictions on grid
    pred_A_grid = clf_A_2d.predict(grid_points)
    pred_B_grid = clf_B_2d.predict(grid_points)
    pred_C_grid = clf_C_2d.predict(grid_points)

    # Map labels to integers for contour plotting
    label_to_int = {'rotten': 0, 'unripe': 1, 'ripe': 2}
    Z_A = np.array([label_to_int[p] for p in pred_A_grid]).reshape(xx.shape)
    Z_B = np.array([label_to_int[p] for p in pred_B_grid]).reshape(xx.shape)
    Z_C = np.array([label_to_int[p] for p in pred_C_grid]).reshape(xx.shape)

    # Plot subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    titles = [
        'Option A: Minimum Distance (Euclidean)',
        'Option B: Bayes Gaussian Classifier',
        'Option C: Mahalanobis Distance'
    ]
    grids = [Z_A, Z_B, Z_C]
    
    # Custom colors
    from matplotlib.colors import ListedColormap
    bg_cmap = ListedColormap(['#E5D4C0', '#D1E7DD', '#FCE8E6'])
    point_colors = {'rotten': '#8B5A2B', 'unripe': '#1E4620', 'ripe': '#B03A2E'}
    
    for i in range(3):
        ax = axes[i]
        ax.contourf(xx, yy, grids[i], alpha=0.6, cmap=bg_cmap)
        
        # Plot training data points
        for cat in CATEGORIES.keys():
            pts = X_train_pca_dict[cat]
            ax.scatter(pts[:, 0], pts[:, 1], c=point_colors[cat], label=cat, edgecolors='black', s=40)
            
        ax.set_title(titles[i])
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        if i == 0:
            ax.legend(loc='lower left')

    plt.tight_layout()
    db_path = os.path.join(ARTIFACT_DIR, 'decision_boundaries.png')
    plt.savefig(db_path, dpi=150)
    plt.close()
    
    print(f"Decision boundaries plot saved to: {db_path}")
    print("=== FINISHED PIPELINE ===")

if __name__ == '__main__':
    main()
